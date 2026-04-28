"""
Context distillation data generation.

For each dataset, prompt the teacher model WITH full context (cultural background,
hints, CoT instruction) and store (student_prompt, teacher_response) pairs.
The student prompt deliberately omits the context that the teacher saw — this is
the distillation step.

Teacher: Qwen2.5-7B-Instruct on port 8003
Output:  finetune/data/{normad,milu,bhed,globalopinion}_teacher.jsonl
"""

import argparse
import ast
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# ── Teachers ──────────────────────────────────────────────────────────────────

TEACHER_SYSTEM_NORMAD = (
    "You are a cultural etiquette expert with deep knowledge of South Asian social norms. "
    "You will be given a cultural background and a story. "
    "First think through the cultural context carefully, then decide whether the action is "
    "socially acceptable. End your response with exactly one word on its own line: yes, no, or neutral."
)

TEACHER_SYSTEM_MILU = (
    "You are an expert tutor on Indian history, culture, science, and current affairs. "
    "For the following multiple-choice question, think step by step and explain your reasoning. "
    "End with 'Answer: X' where X is the letter of the correct option."
)

TEACHER_SYSTEM_BHED = (
    "You are a fairness researcher studying social stereotypes. "
    "For the following sentence with a MASK placeholder, explain why either group could fill the role, "
    "then choose the option that LEAST reinforces harmful stereotypes. "
    "End with 'Answer: A' or 'Answer: B'."
)

TEACHER_SYSTEM_OPINION = (
    "You are a thoughtful Indian citizen who closely follows news and social issues. "
    "For the following survey question, reason from an Indian cultural and social perspective "
    "and select the most appropriate option. End with 'Answer: X' where X is the option letter."
)

INDIC_COUNTRIES = ["india", "pakistan", "bangladesh", "nepal", "sri_lanka"]
INDIA_KEY = "India (Current national sample)"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def call_teacher(client, system, user, max_tokens=512):
    resp = client.chat.completions.create(
        model="qwen2-5-7b-instruct",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


def extract_label(text, labels):
    """Extract first occurrence of a label from text."""
    text_lower = text.strip().lower()
    lines = text_lower.split("\n")
    # Check last few lines first (final answer usually at end)
    for line in reversed(lines[-3:]):
        line = line.strip()
        for lbl in labels:
            if line == lbl or line.endswith(f": {lbl}") or line.startswith(lbl):
                return lbl
    # Fallback: scan full text
    for lbl in labels:
        if lbl in text_lower:
            return lbl
    return None


def extract_letter(text, n_options):
    valid = set(LETTERS[:n_options])
    # Look for "Answer: X" pattern
    m = re.search(r"answer[:\s]+([A-Z])", text, re.IGNORECASE)
    if m and m.group(1) in valid:
        return m.group(1)
    # Fallback: last uppercase letter in valid set
    for ch in reversed(text.upper()):
        if ch in valid:
            return ch
    return None


# ── NormAd ────────────────────────────────────────────────────────────────────

def generate_normad(client, output_path, countries=None, max_workers=8):
    if countries is None:
        countries = INDIC_COUNTRIES

    print("Loading NormAd...")
    ds = load_dataset("akhilayerukola/NormAd")["train"]
    rows = [r for r in ds if r["Country"] in countries]
    print(f"  {len(rows)} rows for countries: {countries}")

    records = []
    errors = 0

    with tqdm(total=len(rows), desc="NormAd teacher") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for row in rows:
                country = row["Country"].replace("_", " ").title()
                teacher_user = (
                    f"Country: {country}\n\n"
                    f"Cultural Background:\n{row['Background'].strip()}\n\n"
                    f"Story: {row['Story'].strip()}\n\n"
                    f"Is this action socially acceptable? Think step by step, "
                    f"then answer with exactly one word on the final line: yes, no, or neutral."
                )
                student_user = (
                    f"Country: {country}\n\n"
                    f"Story: {row['Story'].strip()}\n\n"
                    f"Is this action socially acceptable? Answer yes, no, or neutral."
                )
                fut = pool.submit(call_teacher, client, TEACHER_SYSTEM_NORMAD, teacher_user, 512)
                futures[fut] = (row, student_user)

            for fut in as_completed(futures):
                row, student_user = futures[fut]
                try:
                    response = fut.result()
                    label = extract_label(response, ["yes", "no", "neutral"])
                    gold = row["Gold Label"]
                    if label == gold:   # only keep correct teacher predictions
                        records.append({
                            "source": "normad",
                            "country": row["Country"],
                            "gold": gold,
                            "teacher_correct": True,
                            "messages": [
                                {"role": "user",      "content": student_user},
                                {"role": "assistant", "content": response.strip()},
                            ],
                        })
                    else:
                        errors += 1  # teacher wrong; skip
                except Exception:
                    errors += 1
                pbar.update(1)

    print(f"  NormAd: {len(records)} correct / {len(rows)} total (teacher errors/wrong: {errors})")
    _write_jsonl(records, output_path)
    return records


# ── MILU ──────────────────────────────────────────────────────────────────────

MILU_OPTION_KEYS = ["option1", "option2", "option3", "option4"]
MILU_LETTERS = ["A", "B", "C", "D"]
MILU_TARGET_TO_LETTER = {k: v for k, v in zip(MILU_OPTION_KEYS, MILU_LETTERS)}


def generate_milu(client, output_path, language="English", num_samples=500, max_workers=16):
    print(f"Loading MILU [{language}]...")
    ds_val  = load_dataset("ai4bharat/MILU", language, split="validation")
    ds_test = load_dataset("ai4bharat/MILU", language, split="test")
    all_rows = list(ds_val) + list(ds_test)
    random.shuffle(all_rows)
    rows = all_rows[:num_samples]
    print(f"  {len(rows)} rows")

    records = []
    errors = 0

    with tqdm(total=len(rows), desc=f"MILU {language} teacher") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for ex in rows:
                gold_letter = MILU_TARGET_TO_LETTER.get(ex["target"])
                if not gold_letter:
                    pbar.update(1)
                    continue
                opts = "\n".join(f"{l}. {ex[k]}" for l, k in zip(MILU_LETTERS, MILU_OPTION_KEYS))
                teacher_user = (
                    f"Question: {ex['question']}\n{opts}\n\n"
                    f"Think through this India-specific question carefully. "
                    f"The correct answer is {gold_letter}. Explain why {gold_letter} is correct, "
                    f"then end with 'Answer: {gold_letter}'."
                )
                student_user = f"Question: {ex['question']}\n{opts}\nAnswer:"
                fut = pool.submit(call_teacher, client, TEACHER_SYSTEM_MILU, teacher_user, 400)
                futures[fut] = (ex, student_user, gold_letter)

            for fut in as_completed(futures):
                ex, student_user, gold_letter = futures[fut]
                try:
                    response = fut.result()
                    # Append the clean answer if not already at end
                    if not response.strip().upper().endswith(gold_letter):
                        response = response.strip() + f"\n\nAnswer: {gold_letter}"
                    records.append({
                        "source": f"milu_{language.lower()}",
                        "domain": ex.get("domain", ""),
                        "gold": gold_letter,
                        "messages": [
                            {"role": "user",      "content": student_user},
                            {"role": "assistant", "content": response.strip()},
                        ],
                    })
                except Exception:
                    errors += 1
                pbar.update(1)

    print(f"  MILU {language}: {len(records)} records (errors: {errors})")
    _write_jsonl(records, output_path)
    return records


# ── BhED ──────────────────────────────────────────────────────────────────────

def generate_bhed(client, output_path, max_workers=8):
    import requests
    from io import StringIO
    import pandas as pd

    base = "https://raw.githubusercontent.com/khyatikhandelwal/Indian-LLMs-Bias/main/Data/"
    print("Downloading BhED CSVs from GitHub...")
    caste_df    = pd.read_csv(StringIO(requests.get(base + "Caste.csv",         timeout=30).text))
    religion_df = pd.read_csv(StringIO(requests.get(base + "India_Religious.csv", timeout=30).text))
    all_rows = [("caste", r) for r in caste_df.to_dict("records")] + \
               [("religion", r) for r in religion_df.to_dict("records")]
    print(f"  {len(all_rows)} rows total")

    def parse_list(s):
        try: return ast.literal_eval(s)
        except: return [str(s)]

    records = []
    errors = 0

    with tqdm(total=len(all_rows), desc="BhED teacher") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for cat, row in all_rows:
                stereo = parse_list(row["Target_Stereotypical"])[0]
                anti   = parse_list(row["Target_Anti-Stereotypical"])[0]
                sentence = row["Sentence"].strip()
                # Randomize A/B to counter position bias in training data
                if random.random() < 0.5:
                    opt_a, opt_b, stereo_is_a = stereo, anti, True
                else:
                    opt_a, opt_b, stereo_is_a = anti, stereo, False
                anti_letter = "B" if stereo_is_a else "A"
                teacher_user = (
                    f'Sentence: "{sentence}"\n\n'
                    f"A. {opt_a}\nB. {opt_b}\n\n"
                    f"Which group fills MASK without reinforcing harmful stereotypes? "
                    f"Explain your reasoning, then answer with 'Answer: {anti_letter}' "
                    f"(the less stereotypical choice)."
                )
                student_user = (
                    f'Sentence: "{sentence}"\n\n'
                    f"Which group most naturally fills MASK in this sentence?\n"
                    f"A. {opt_a}\nB. {opt_b}\n\nAnswer with only A or B."
                )
                fut = pool.submit(call_teacher, client, TEACHER_SYSTEM_BHED, teacher_user, 350)
                futures[fut] = (row, student_user, anti_letter, cat)

            for fut in as_completed(futures):
                row, student_user, anti_letter, cat = futures[fut]
                try:
                    response = fut.result()
                    if not response.strip().upper().endswith(anti_letter):
                        response = response.strip() + f"\n\nAnswer: {anti_letter}"
                    records.append({
                        "source": f"bhed_{cat}",
                        "messages": [
                            {"role": "user",      "content": student_user},
                            {"role": "assistant", "content": response.strip()},
                        ],
                    })
                except Exception:
                    errors += 1
                pbar.update(1)

    print(f"  BhED: {len(records)} records (errors: {errors})")
    _write_jsonl(records, output_path)
    return records


# ── Global Opinion QA ─────────────────────────────────────────────────────────

def parse_selections(sel_raw):
    if isinstance(sel_raw, dict): return sel_raw
    if isinstance(sel_raw, str):
        cleaned = sel_raw.replace("<class 'list'>", "list")
        try: return eval(cleaned, {"defaultdict": defaultdict, "list": list, "__builtins__": {}})
        except: pass
    return {}


def parse_options(s):
    if isinstance(s, list): return s
    if isinstance(s, str):
        try: return ast.literal_eval(s)
        except: return [s]
    return []


def generate_globalopinion(client, output_path, num_samples=300, max_workers=16):
    print("Loading Global Opinion QA...")
    ds = load_dataset("Anthropic/llm_global_opinions", split="train")

    india_rows = []
    for row in ds:
        sels = parse_selections(row["selections"])
        if INDIA_KEY in sels:
            india_dist = sels[INDIA_KEY]
            options = parse_options(row["options"])
            if isinstance(india_dist, list) and len(india_dist) == len(options) > 0:
                # Find the option India most prefers
                total = sum(india_dist)
                if total == 0: continue
                best_idx = max(range(len(india_dist)), key=lambda i: india_dist[i])
                india_rows.append({
                    "question": row["question"],
                    "options": options,
                    "india_dist": india_dist,
                    "best_idx": best_idx,
                    "source": row.get("source", ""),
                })

    print(f"  {len(india_rows)} rows with India data; sampling {num_samples}")
    rows = random.sample(india_rows, min(num_samples, len(india_rows)))

    records = []
    errors = 0

    with tqdm(total=len(rows), desc="GlobalOpinion teacher") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for row in rows:
                opts_str = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
                india_letter = LETTERS[row["best_idx"]]
                teacher_user = (
                    f"Question: {row['question']}\n\n{opts_str}\n\n"
                    f"From an Indian cultural and social perspective, explain why "
                    f"option {india_letter} ('{row['options'][row['best_idx']]}') "
                    f"best represents Indian public opinion. Then end with 'Answer: {india_letter}'."
                )
                student_user = (
                    f"Question: {row['question']}\n\n{opts_str}\n\nYour answer (single letter):"
                )
                fut = pool.submit(call_teacher, client, TEACHER_SYSTEM_OPINION, teacher_user, 400)
                futures[fut] = (row, student_user, india_letter)

            for fut in as_completed(futures):
                row, student_user, india_letter = futures[fut]
                try:
                    response = fut.result()
                    if not response.strip().upper().endswith(india_letter):
                        response = response.strip() + f"\n\nAnswer: {india_letter}"
                    records.append({
                        "source": "globalopinion",
                        "india_letter": india_letter,
                        "messages": [
                            {"role": "user",      "content": student_user},
                            {"role": "assistant", "content": response.strip()},
                        ],
                    })
                except Exception:
                    errors += 1
                pbar.update(1)

    print(f"  GlobalOpinion: {len(records)} records (errors: {errors})")
    _write_jsonl(records, output_path)
    return records


# ── Utilities ─────────────────────────────────────────────────────────────────

def _write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-url",    default="http://localhost:8003/v1")
    parser.add_argument("--data-dir",       default="finetune/data")
    parser.add_argument("--tasks",          nargs="+",
                        default=["normad", "milu_en", "milu_hi", "bhed", "globalopinion"])
    parser.add_argument("--milu-samples",   type=int, default=500)
    parser.add_argument("--opinion-samples",type=int, default=300)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--hf-token",       default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    if args.hf_token:
        import os
        os.environ["HF_TOKEN"] = args.hf_token

    client = OpenAI(base_url=args.teacher_url, api_key="dummy")

    # Verify teacher is up
    try:
        models = client.models.list()
        print(f"Teacher model: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Teacher not available at {args.teacher_url}: {e}")
        return

    if "normad" in args.tasks:
        generate_normad(client, f"{args.data_dir}/normad_teacher.jsonl")

    if "milu_en" in args.tasks:
        generate_milu(client, f"{args.data_dir}/milu_en_teacher.jsonl",
                      language="English", num_samples=args.milu_samples)

    if "milu_hi" in args.tasks:
        generate_milu(client, f"{args.data_dir}/milu_hi_teacher.jsonl",
                      language="Hindi", num_samples=args.milu_samples)

    if "bhed" in args.tasks:
        generate_bhed(client, f"{args.data_dir}/bhed_teacher.jsonl")

    if "globalopinion" in args.tasks:
        generate_globalopinion(client, f"{args.data_dir}/globalopinion_teacher.jsonl",
                               num_samples=args.opinion_samples)

    print("\nAll teacher data generated.")


if __name__ == "__main__":
    main()
