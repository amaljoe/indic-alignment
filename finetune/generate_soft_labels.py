"""
Generate soft-label training data using Qwen3-30B-A3B teacher.

For each dataset, the teacher sees a rich cultural-context system prompt and
generates a response.  We then do a forward pass over (prompt + teacher_response)
to capture the top-256 log-prob distribution at every response token position.

Those sparse distributions become the KL supervision signal in train_kl.py.
The student messages intentionally omit the cultural context — this is the
context-distillation trick: the student learns to answer without explicit context.

Train/test splits enforced
  NormAd        train = india/pakistan/bangladesh   test = nepal/sri_lanka
  BhED          80/20 random (seed 42)
  MILU          use dataset "train" split
  GlobalOpinion 80/20 by index (seed 42)

Outputs
  finetune/data/{source}_soft.jsonl   — training examples with teacher_logprobs
  finetune/data/{source}_test.jsonl   — held-out test examples (messages only)

Each soft-label record:
  {
    "messages":        [ student system, user, assistant-response ],
    "teacher_logprobs":[ [{token_id, log_prob}, ...] × R ],   # R = resp token count
    "source":          str,
    "gold_label":      str | null
  }
"""

import argparse
import ast
import json
import os
import random
import re
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-30B-A3B"
DATA_DIR   = "finetune/data"
TOP_K      = 256
SEED       = 42
MAX_NEW_TOKENS = 512
MAX_INPUT_LEN  = 1024          # max prompt tokens; skip if longer

NORMAD_TRAIN_COUNTRIES = {"india", "pakistan", "bangladesh"}
NORMAD_TEST_COUNTRIES  = {"nepal", "sri_lanka"}
INDIC_COUNTRIES        = NORMAD_TRAIN_COUNTRIES | NORMAD_TEST_COUNTRIES
INDIA_KEY              = "India (Current national sample)"
LETTERS                = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

random.seed(SEED)

# ── System prompts ─────────────────────────────────────────────────────────────

TEACHER_SYS = {
    "normad": (
        "You are a cultural etiquette expert with deep knowledge of South Asian social norms. "
        "You will be given cultural background about a specific country and a story. "
        "First think through the cultural context carefully inside <think> tags, then decide "
        "whether the action described is socially acceptable in that culture. "
        "End your response with exactly one word on its own line: yes, no, or neutral."
    ),
    "milu": (
        "You are an expert tutor on Indian history, culture, science, geography, and current affairs. "
        "For the following multiple-choice question, think step by step inside <think> tags, drawing "
        "on specific cultural and historical knowledge. "
        "End your response with 'Answer: X' where X is the letter of the correct option."
    ),
    "bhed": (
        "You are a fairness researcher studying social biases in South Asian contexts. "
        "When presented with a sentence containing a MASK placeholder and two options, "
        "consider the social stereotypes associated with each group carefully, then choose "
        "the option that LEAST reinforces harmful stereotypes. "
        "Explain briefly inside <think> tags, then end with 'Answer: A' or 'Answer: B'."
    ),
    "globalopinion": (
        "You are a thoughtful, well-informed Indian citizen deeply familiar with Indian society, "
        "culture, politics, and current events. For the following survey question, reason carefully "
        "from an Indian cultural and social perspective. "
        "End with 'Answer: X' where X is the option letter."
    ),
}

STUDENT_SYS = {
    "normad": (
        "You are an expert on South Asian cultural norms and social etiquette. "
        "Given a story, decide whether the described action is socially acceptable. "
        "End your response with exactly one word on its own line: yes, no, or neutral."
    ),
    "milu": (
        "You are an expert on Indian history, culture, science, geography, and current affairs. "
        "For the following multiple-choice question, think step by step and explain your reasoning. "
        "End with 'Answer: X' where X is the letter of the correct option."
    ),
    "bhed": (
        "You are a helpful assistant. For the following sentence with a MASK placeholder, "
        "choose the option that LEAST reinforces harmful social stereotypes. "
        "End with 'Answer: A' or 'Answer: B'."
    ),
    "globalopinion": (
        "You are an expert on Indian society, culture, and public opinion. "
        "For the following survey question about global perspectives, select the option "
        "that best represents an Indian viewpoint. End with 'Answer: X' where X is the option letter."
    ),
}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_teacher(model_id=MODEL_ID):
    print(f"Loading teacher: {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Teacher loaded.", flush=True)
    return tok, model


# ── Core extraction ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_and_extract(tok, model, teacher_msgs, student_msgs, max_new=MAX_NEW_TOKENS):
    """
    1. Generate teacher response for teacher_msgs.
    2. Forward pass over (teacher_msgs + response) to get per-token distributions.
    3. Return (response_text, top_k_logprobs_list).

    top_k_logprobs_list[i] = list of {"token_id": int, "log_prob": float}  (len=TOP_K)
    """
    # Tokenize teacher prompt
    prompt_text = tok.apply_chat_template(
        teacher_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    if prompt_ids.shape[1] > MAX_INPUT_LEN:
        return None, None

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)

    # Generate (greedy for determinism)
    gen_out = model.generate(
        prompt_ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    new_ids = gen_out[0, prompt_ids.shape[1]:]   # response token ids
    if new_ids.numel() == 0:
        return None, None

    response_text = tok.decode(new_ids, skip_special_tokens=True)

    # Full forward pass over prompt+response to get exact next-token distributions
    full_ids = gen_out  # [1, L_full]
    logits = model(input_ids=full_ids).logits[0]  # [L_full, V]

    # Response positions: predict new_ids[i] from position (prompt_len - 1 + i)
    resp_start = prompt_ids.shape[1] - 1
    resp_logits = logits[resp_start : resp_start + new_ids.shape[0]]  # [R, V]

    log_probs = F.log_softmax(resp_logits.float(), dim=-1)  # [R, V]
    topk_lp, topk_id = torch.topk(log_probs, k=TOP_K, dim=-1)  # [R, K]

    topk_lp  = topk_lp.cpu().float().numpy()
    topk_id  = topk_id.cpu().numpy()

    teacher_logprobs = [
        [{"token_id": int(topk_id[i, j]), "log_prob": float(topk_lp[i, j])}
         for j in range(TOP_K)]
        for i in range(new_ids.shape[0])
    ]

    return response_text, teacher_logprobs


def make_record(student_msgs, teacher_logprobs, source, gold_label=None):
    return {
        "messages": student_msgs,
        "teacher_logprobs": teacher_logprobs,
        "source": source,
        "gold_label": gold_label,
    }


# ── Dataset helpers ────────────────────────────────────────────────────────────

def parse_selections(sel_raw):
    if isinstance(sel_raw, dict):
        return sel_raw
    if isinstance(sel_raw, str):
        cleaned = sel_raw.replace("<class 'list'>", "list")
        try:
            return eval(cleaned, {"defaultdict": defaultdict, "list": list, "__builtins__": {}})
        except Exception:
            pass
    return {}


def parse_options(s):
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except Exception:
            return [s]
    return []


def extract_label_normad(text):
    text = text.strip().lower()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    for lbl in ("yes", "no", "neutral"):
        if text.startswith(lbl):
            return lbl
    tail = text[-300:]
    for lbl in ("yes", "no", "neutral"):
        if re.search(rf"\b{lbl}\b", tail):
            return lbl
    return None


# ── Per-dataset generators ─────────────────────────────────────────────────────

def process_normad(tok, model, args):
    ds = load_dataset("akhilayerukola/NormAd", split="test")
    train_recs, test_recs = [], []

    for row in tqdm(ds, desc="NormAd"):
        country = str(row.get("Country", "")).lower().replace(" ", "_")
        if country not in INDIC_COUNTRIES:
            continue

        gold = str(row.get("Gold Label", "")).strip().lower()
        if gold not in ("yes", "no", "neutral"):
            continue

        background = row.get("Background", "")
        story      = row.get("Story", "")
        subaxis    = row.get("Subaxis", "")

        teacher_user = (
            f"Country: {country.replace('_', ' ').title()}\n"
            f"Cultural domain: {subaxis}\n\n"
            f"Cultural background:\n{background}\n\n"
            f"Story:\n{story}"
        )
        student_user = f"Story:\n{story}"

        teacher_msgs = [
            {"role": "system",    "content": TEACHER_SYS["normad"]},
            {"role": "user",      "content": teacher_user},
        ]
        student_msgs_base = [
            {"role": "system",    "content": STUDENT_SYS["normad"]},
            {"role": "user",      "content": student_user},
        ]

        resp, tl = generate_and_extract(tok, model, teacher_msgs, student_msgs_base)
        if resp is None:
            continue

        # Only keep examples where teacher prediction == gold (filter noisy)
        pred = extract_label_normad(resp)
        if args.filter_correct and pred != gold:
            continue

        student_msgs = student_msgs_base + [{"role": "assistant", "content": resp}]
        rec = make_record(student_msgs, tl, "normad", gold)

        if country in NORMAD_TRAIN_COUNTRIES:
            train_recs.append(rec)
        else:
            # Test split: messages only (no teacher_logprobs needed)
            test_recs.append({"messages": student_msgs_base, "source": "normad", "gold_label": gold})

    return train_recs, test_recs


def process_milu(tok, model, lang="en"):
    source = f"milu_{lang}"
    config = "hi" if lang == "hi" else "en"
    try:
        ds = load_dataset("sarvamai/MILU", config, split="train")
    except Exception:
        ds = load_dataset("sarvamai/MILU", split="train")

    train_recs = []
    for row in tqdm(ds, desc=f"MILU-{lang}"):
        question = row.get("question", "")
        options  = row.get("options", [])
        if isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except Exception:
                options = [options]
        answer_idx = row.get("answer", None)

        if not options or answer_idx is None:
            continue

        opts_str = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options) if i < 26)
        try:
            gold = LETTERS[int(answer_idx)]
        except (ValueError, IndexError):
            gold = str(answer_idx).strip().upper()
            if gold not in LETTERS:
                continue

        user_q = f"{question}\n\n{opts_str}"
        teacher_msgs = [
            {"role": "system", "content": TEACHER_SYS["milu"]},
            {"role": "user",   "content": user_q},
        ]
        student_msgs_base = [
            {"role": "system", "content": STUDENT_SYS["milu"]},
            {"role": "user",   "content": user_q},
        ]

        resp, tl = generate_and_extract(tok, model, teacher_msgs, student_msgs_base)
        if resp is None:
            continue

        student_msgs = student_msgs_base + [{"role": "assistant", "content": resp}]
        train_recs.append(make_record(student_msgs, tl, source, gold))

    return train_recs, []   # MILU has no separate test split here; eval on val


def process_bhed(tok, model):
    import io
    import urllib.request

    CASTE_URL    = "https://raw.githubusercontent.com/khyatikhandelwal/Indian-LLMs-Bias/main/Data/Caste.csv"
    RELIGION_URL = "https://raw.githubusercontent.com/khyatikhandelwal/Indian-LLMs-Bias/main/Data/India_Religious.csv"

    def load_csv(url):
        with urllib.request.urlopen(url) as r:
            return r.read().decode("utf-8")

    import csv
    all_rows = []
    for url, cat in [(CASTE_URL, "caste"), (RELIGION_URL, "religion")]:
        reader = csv.DictReader(io.StringIO(load_csv(url)))
        for row in reader:
            row["_category"] = cat
            all_rows.append(row)

    random.shuffle(all_rows)
    n_test  = max(1, int(len(all_rows) * 0.20))
    test_rows  = all_rows[:n_test]
    train_rows = all_rows[n_test:]

    train_recs, test_recs = [], []

    def process_rows(rows, split):
        recs = []
        for row in tqdm(rows, desc=f"BhED-{split}"):
            stereo     = row.get("Target_Stereotypical", "").strip()
            anti_stereo = row.get("Target_Anti-Stereotypical", "").strip()
            sentence   = row.get("Sentence", "").strip()
            if not stereo or not anti_stereo or not sentence or "MASK" not in sentence:
                continue

            # Randomise A/B assignment to prevent position bias
            stereo_is_a = random.random() < 0.5
            if stereo_is_a:
                opt_a, opt_b = stereo, anti_stereo
                gold = "B"   # anti-stereotypical is correct
            else:
                opt_a, opt_b = anti_stereo, stereo
                gold = "A"   # anti-stereotypical is correct

            user_q = (
                f"Sentence: {sentence}\n\n"
                f"A. {opt_a}\n"
                f"B. {opt_b}\n\n"
                "Which option LEAST reinforces harmful stereotypes?"
            )
            teacher_msgs = [
                {"role": "system", "content": TEACHER_SYS["bhed"]},
                {"role": "user",   "content": user_q},
            ]
            student_msgs_base = [
                {"role": "system", "content": STUDENT_SYS["bhed"]},
                {"role": "user",   "content": user_q},
            ]

            if split == "train":
                resp, tl = generate_and_extract(tok, model, teacher_msgs, student_msgs_base)
                if resp is None:
                    continue
                student_msgs = student_msgs_base + [{"role": "assistant", "content": resp}]
                recs.append(make_record(student_msgs, tl, "bhed", gold))
            else:
                recs.append({
                    "messages": student_msgs_base,
                    "source": "bhed",
                    "gold_label": gold,
                    "_stereo_is_a": stereo_is_a,
                    "_category": row["_category"],
                })
        return recs

    train_recs = process_rows(train_rows, "train")
    test_recs  = process_rows(test_rows,  "test")
    return train_recs, test_recs


def process_globalopinion(tok, model):
    ds = load_dataset("Anthropic/llm_global_opinions", split="train")

    all_rows = [r for r in ds
                if INDIA_KEY in parse_selections(r.get("selections", {}))]
    random.shuffle(all_rows)
    n_test  = max(1, int(len(all_rows) * 0.20))
    test_rows  = all_rows[:n_test]
    train_rows = all_rows[n_test:]

    train_recs, test_recs = [], []

    def process_rows(rows, split):
        recs = []
        for row in tqdm(rows, desc=f"GlobalOpinion-{split}"):
            question = row.get("question", "")
            options  = parse_options(row.get("options", []))
            if not options:
                continue
            n = len(options)
            if n > 26:
                continue

            opts_str = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
            user_q = f"Survey question: {question}\n\nOptions:\n{opts_str}"

            teacher_msgs = [
                {"role": "system", "content": TEACHER_SYS["globalopinion"]},
                {"role": "user",   "content": user_q},
            ]
            student_msgs_base = [
                {"role": "system", "content": STUDENT_SYS["globalopinion"]},
                {"role": "user",   "content": user_q},
            ]

            if split == "train":
                resp, tl = generate_and_extract(tok, model, teacher_msgs, student_msgs_base)
                if resp is None:
                    continue
                student_msgs = student_msgs_base + [{"role": "assistant", "content": resp}]
                recs.append(make_record(student_msgs, tl, "globalopinion"))
            else:
                recs.append({"messages": student_msgs_base, "source": "globalopinion"})
        return recs

    train_recs = process_rows(train_rows, "train")
    test_recs  = process_rows(test_rows,  "test")
    return train_recs, test_recs


# ── Main ───────────────────────────────────────────────────────────────────────

SOURCES = ["normad", "milu_en", "milu_hi", "bhed", "globalopinion"]


def save_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default=MODEL_ID)
    parser.add_argument("--sources",        nargs="+", default=SOURCES,
                        help="Subset of sources to process")
    parser.add_argument("--filter-correct", action="store_true", default=True,
                        help="NormAd: keep only examples where teacher prediction == gold")
    parser.add_argument("--no-filter-correct", dest="filter_correct", action="store_false")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok, model = load_teacher(args.model or MODEL_ID)

    for src in args.sources:
        print(f"\n=== {src} ===")
        if src == "normad":
            train, test = process_normad(tok, model, args)
        elif src == "milu_en":
            train, test = process_milu(tok, model, "en")
        elif src == "milu_hi":
            train, test = process_milu(tok, model, "hi")
        elif src == "bhed":
            train, test = process_bhed(tok, model)
        elif src == "globalopinion":
            train, test = process_globalopinion(tok, model)
        else:
            print(f"Unknown source: {src}")
            continue

        save_jsonl(os.path.join(DATA_DIR, f"{src}_soft.jsonl"),  train)
        if test:
            save_jsonl(os.path.join(DATA_DIR, f"{src}_test.jsonl"), test)

    print("\nDone.")


if __name__ == "__main__":
    main()
