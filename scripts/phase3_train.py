"""
Phase 3 — HHH Safety DPO Training Pipeline (end-to-end, self-contained).

Workflow:
  1. Translate HH-RLHF English pairs to new Indic languages using Gemma 3 27B
     (Tamil, Bengali, Telugu, Marathi — each batched 128 at a time)
  2. Translate HHH eval data to the same new languages
  3. DPO+LoRA training on all available HH-RLHF languages
  4. Load trained adapter into student vLLM via API
  5. Re-eval on all available HHH languages

Gemma serves on --teacher-url (serve_teacher.sh must be running).
Student vLLM serves on --base-url (serve.sh must be running).

Usage (run inside apptainer via run_overnight.sh):
  python phase3_train.py --teacher-url http://localhost:8003/v1
"""

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

VLLM_PYTHON = "/home/compiling-ganesh/24m0797/envs/vllm/bin/python"
HHH_DATA_DIR = "data/hhh_alignment"
RLHF_DATA_DIR = "data/hh_rlhf"

# New languages to translate (beyond existing en, hi, ml)
NEW_LANG_CONFIGS = {
    "tamil":   {"name": "Tamil",   "code": "ta"},
    "bengali": {"name": "Bengali", "code": "bn"},
    "telugu":  {"name": "Telugu",  "code": "te"},
    "marathi": {"name": "Marathi", "code": "mr"},
}

# Existing HH-RLHF files (pre-translated)
EXISTING_RLHF = {
    "english":   "hh_rlhf_5k_en.jsonl",
    "hindi":     "hh_rlhf_5k_hindi.jsonl",
    "malayalam": "hh_rlhf_5k_malayalam.jsonl",
}

# Existing HHH eval files
EXISTING_HHH = {
    "english":   "english.jsonl",
    "hindi":     "hindi_gemma3_27b.jsonl",
    "malayalam": "malayalam_gemma3_27b.jsonl",
}

TRANSLATE_SYSTEM_TEMPLATE = (
    "You are a professional translator. "
    "Translate the following conversation to {lang_name}. "
    "Keep all markers like '\\n\\nHuman:' and '\\n\\nAssistant:' exactly as-is (do not translate them). "
    "Keep names, technical terms, and URLs unchanged. "
    "Provide only the translation, nothing else."
)

TRANSLATE_HHH_SYSTEM_TEMPLATE = (
    "You are a professional translator. "
    "Translate the following text to {lang_name}. "
    "Preserve the meaning and tone accurately. "
    "Provide only the translation, nothing else."
)


# ── Shared utilities ──────────────────────────────────────────────────────────

def validate_response(raw, finish_reason, think_mode=False):
    issues = {"overflow": False, "gibberish": False, "empty": False, "warnings": []}
    if not raw or not raw.strip():
        issues["empty"] = True
        return issues
    if finish_reason == "length":
        issues["overflow"] = True
        issues["warnings"].append("finish_reason=length")
    if think_mode and "</think>" not in raw:
        issues["overflow"] = True
        issues["warnings"].append("no </think> tag")
    text = raw.split("</think>")[-1] if "</think>" in raw else raw
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha < 0.15:
        issues["gibberish"] = True
        issues["warnings"].append(f"low alpha={alpha:.2f}")
    if re.search(r"(.)\1{20,}", text):
        issues["gibberish"] = True
        issues["warnings"].append("repeated char run")
    return issues


def append_results_md(row, md_path="final/results.md"):
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    header_needed = not os.path.exists(md_path)
    with open(md_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Indic Alignment Results\n\n")
            f.write(
                "| Phase | Stage | Model | MILU-Hi | MILU-En | "
                "NormAd-Acc | BhED-Stereo | GlobalOp-JS | HHH-Acc |\n"
            )
            f.write(
                "|-------|-------|-------|---------|---------|"
                "------------|-------------|-------------|----------|\n"
            )
        cols = [
            row.get("phase", "-"), row.get("stage", "-"), row.get("model_tag", "-"),
            row.get("milu_hi", "-"), row.get("milu_en", "-"),
            row.get("normad_acc", "-"), row.get("bhed_stereo", "-"),
            row.get("globalop_js", "-"), row.get("hhh_acc", "-"),
        ]
        f.write("| " + " | ".join(str(c) for c in cols) + " |\n")
    print(f"  → Appended to {md_path}")


def call_teacher(client, model, system, user, max_tokens, temp=0.3):
    resp = client.chat.completions.create(
        model=model, temperature=temp,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or "", resp.choices[0].finish_reason


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── HH-RLHF translation ───────────────────────────────────────────────────────

def parse_hh_conversation(text):
    """Split hh-rlhf conversation into (prompt, response)."""
    parts = text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        return parts[0] + "\n\nAssistant:", parts[1].strip()
    return text, ""


def translate_rlhf_batch(client, teacher_model, records, lang_name,
                          batch_size, max_tokens_per_side):
    from tqdm import tqdm
    translated = []
    errors = overflow = gibberish = 0
    system = TRANSLATE_SYSTEM_TEMPLATE.format(lang_name=lang_name)

    print(f"  Translating {len(records)} HH-RLHF pairs to {lang_name} ...")

    # For each record, we need to translate the chosen conversation.
    # The rejected conversation shares most content — translate separately.
    futures_map = {}
    with tqdm(total=len(records) * 2, unit="call", desc=f"translate→{lang_name}") as pbar:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for i, rec in enumerate(records):
                chosen_text = rec["chosen"][:2000]  # truncate very long conversations
                rejected_text = rec["rejected"][:2000]
                fut_c = pool.submit(call_teacher, client, teacher_model,
                                    system, chosen_text, max_tokens_per_side)
                fut_r = pool.submit(call_teacher, client, teacher_model,
                                    system, rejected_text, max_tokens_per_side)
                futures_map[fut_c] = (i, "chosen")
                futures_map[fut_r] = (i, "rejected")

            results_buffer = defaultdict(dict)
            for fut in as_completed(futures_map):
                i, role = futures_map[fut]
                try:
                    raw, finish = fut.result()
                except Exception:
                    errors += 1
                    results_buffer[i][role] = None
                    pbar.update(1)
                    continue
                val = validate_response(raw, finish)
                if val["overflow"] or val["empty"]:
                    overflow += 1
                    results_buffer[i][role] = None
                else:
                    if val["gibberish"]:
                        gibberish += 1
                    results_buffer[i][role] = raw
                pbar.update(1)

    # Assemble valid pairs
    for i, rec in enumerate(records):
        buf = results_buffer[i]
        chosen_t = buf.get("chosen")
        rejected_t = buf.get("rejected")
        if chosen_t and rejected_t and chosen_t != rejected_t:
            translated.append({
                "chosen": chosen_t,
                "rejected": rejected_t,
                "lang": lang_name.lower(),
            })

    print(f"  {lang_name}: {len(translated)}/{len(records)} pairs translated "
          f"(err={errors} overflow={overflow} gibberish={gibberish})")

    # Sample inspection
    if translated:
        sample = random.choice(translated)
        print(f"  Sample chosen[:200]: {sample['chosen'][:200]}")
    return translated


def translate_hhh_batch(client, teacher_model, examples, lang_name,
                         batch_size, max_tokens):
    from tqdm import tqdm
    system = TRANSLATE_HHH_SYSTEM_TEMPLATE.format(lang_name=lang_name)
    translated = []
    errors = overflow = 0

    print(f"  Translating {len(examples)} HHH examples to {lang_name} ...")

    # Each HHH example has: input, choices (2 responses). Translate all 3 texts.
    futures_map = {}
    with tqdm(total=len(examples) * 3, unit="call", desc=f"HHH→{lang_name}") as pbar:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for i, ex in enumerate(examples):
                fut_input = pool.submit(call_teacher, client, teacher_model,
                                        system, ex["input"][:1000], max_tokens)
                fut_c0 = pool.submit(call_teacher, client, teacher_model,
                                     system, ex["choices"][0][:800], max_tokens)
                fut_c1 = pool.submit(call_teacher, client, teacher_model,
                                     system, ex["choices"][1][:800], max_tokens)
                futures_map[fut_input] = (i, "input")
                futures_map[fut_c0] = (i, "choice0")
                futures_map[fut_c1] = (i, "choice1")

            results_buffer = defaultdict(dict)
            for fut in as_completed(futures_map):
                i, role = futures_map[fut]
                try:
                    raw, finish = fut.result()
                except Exception:
                    errors += 1
                    results_buffer[i][role] = None
                    pbar.update(1)
                    continue
                val = validate_response(raw, finish)
                if val["overflow"] or val["empty"]:
                    overflow += 1
                    results_buffer[i][role] = None
                else:
                    results_buffer[i][role] = raw
                pbar.update(1)

    for i, ex in enumerate(examples):
        buf = results_buffer[i]
        t_input = buf.get("input")
        t_c0 = buf.get("choice0")
        t_c1 = buf.get("choice1")
        if t_input and t_c0 and t_c1:
            translated.append({
                "subset": ex["subset"],
                "input": t_input,
                "target_scores": {t_c0: ex["labels"][0], t_c1: ex["labels"][1]},
            })

    print(f"  HHH {lang_name}: {len(translated)}/{len(examples)} examples "
          f"(err={errors} overflow={overflow})")
    return translated


# ── DPO dataset assembly ──────────────────────────────────────────────────────

def load_hhh_examples(path):
    examples = []
    for line in load_jsonl(path):
        items = list(line["target_scores"].items())
        choices = [c for c, _ in items]
        labels = [s for _, s in items]
        if sum(labels) != 1 or len(choices) != 2:
            continue
        examples.append({
            "subset": line.get("subset", "other"),
            "input": line["input"],
            "choices": choices,
            "labels": labels,
        })
    return examples


def build_dpo_pairs_from_rlhf(records):
    """Convert hh-rlhf format to DPO {prompt, chosen, rejected} format."""
    pairs = []
    for rec in records:
        chosen = rec.get("chosen", "")
        rejected = rec.get("rejected", "")
        prompt, chosen_resp = parse_hh_conversation(chosen)
        _, rejected_resp = parse_hh_conversation(rejected)
        if chosen_resp and rejected_resp and chosen_resp != rejected_resp:
            pairs.append({
                "prompt": prompt,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
            })
    return pairs


# ── Training worker ───────────────────────────────────────────────────────────

def run_training_worker(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset, concatenate_datasets

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Load DPO data
    with open(args.data_path, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    ds = Dataset.from_list(raw).shuffle(seed=42)
    if local_rank == 0:
        print(f"  DPO pairs: {len(ds)}")

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_train,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        beta=args.beta,
        max_length=args.max_seq_len,
        truncation_mode="keep_end",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    if local_rank == 0:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_trainable:,}")

    trainer.train()
    if local_rank == 0:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  Adapter saved to: {args.output_dir}")


# ── Orchestrator helpers ──────────────────────────────────────────────────────

def launch_training(args):
    gpus = args.train_gpus
    nproc = len(gpus.split(","))
    script = os.path.abspath(__file__)
    worker_args = " ".join(shlex.quote(a) for a in sys.argv[1:])
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpus} "
        f"HTTP_PROXY=http://127.0.0.1:3128 "
        f"HTTPS_PROXY=http://127.0.0.1:3128 "
        f"{VLLM_PYTHON} -m torch.distributed.run "
        f"--nproc_per_node {nproc} --master_port 29503 "
        f"{script} {worker_args}"
    )
    print(f"\n[launch_training] GPUs={gpus} nproc={nproc}")
    subprocess.run(cmd, shell=True, check=True)


def load_lora_adapter(base_url, name, path):
    import requests
    url = base_url.rstrip("/") + "/load_lora_adapter"
    resp = requests.post(url, json={"lora_name": name, "lora_path": os.path.abspath(path)},
                         timeout=60)
    resp.raise_for_status()
    print(f"  LoRA '{name}' loaded — HTTP {resp.status_code}")


def wait_for_adapter(base_url, model_name, max_wait=30):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="dummy")
    for _ in range(max_wait):
        try:
            models = [m.id for m in client.models.list().data]
            if model_name in models:
                print(f"  Adapter '{model_name}' live.")
                return
        except Exception:
            pass
        time.sleep(2)
    print(f"  ⚠  Adapter not confirmed after {max_wait*2}s")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Phase 3: HHH DPO pipeline")
    ap.add_argument("--model-path", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--output-dir", default="checkpoints/phase3_lora")
    ap.add_argument("--data-path", default="data/phase3_hhh_dpo.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8002/v1")
    ap.add_argument("--teacher-url", default="http://localhost:8003/v1")
    ap.add_argument("--teacher-model", default="gemma3-27b")
    ap.add_argument("--adapter-name", default="phase3")
    ap.add_argument("--train-gpus", default="2,3")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batch-size-train", type=int, default=2)
    ap.add_argument("--max-tokens-translate", type=int, default=1024)
    ap.add_argument("--n-translate-per-lang", type=int, default=2000,
                    help="HH-RLHF pairs to translate per new language")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--skip-translate", action="store_true",
                    help="Skip translation step if translated files already exist")
    ap.add_argument("--data-only", action="store_true",
                    help="Only translate/build dataset and exit; skip training")
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Running as torchrun worker ────────────────────────────────────────────
    if "LOCAL_RANK" in os.environ:
        run_training_worker(args)
        return

    # ── Orchestrator ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Phase 3 — HHH DPO Pipeline")
    print("=" * 60)

    random.seed(args.seed)

    # Step 1: Translate HH-RLHF + HHH eval to new languages
    if args.skip_translate:
        print(f"\n[Step 1] Skipping translation (--skip-translate)")
    elif os.path.exists(args.data_path):
        print(f"\n[Step 1] Reusing existing DPO dataset: {args.data_path}")
    else:
        print(f"\n[Step 1] Translating HH-RLHF to new Indic languages ...")
        from openai import OpenAI
        t_client = OpenAI(base_url=args.teacher_url, api_key="dummy")

        # Load English RLHF as source for translation
        en_rlhf_path = os.path.join(RLHF_DATA_DIR, EXISTING_RLHF["english"])
        en_records = load_jsonl(en_rlhf_path)
        random.shuffle(en_records)
        translate_pool = en_records[:args.n_translate_per_lang]
        print(f"  Using {len(translate_pool)} English HH-RLHF pairs as translation source")

        # Load English HHH as source for eval translation
        en_hhh_path = os.path.join(HHH_DATA_DIR, EXISTING_HHH["english"])
        en_hhh = load_hhh_examples(en_hhh_path)
        print(f"  Will also translate {len(en_hhh)} HHH eval examples to new languages")

        for lang_key, lang_cfg in NEW_LANG_CONFIGS.items():
            lang_name = lang_cfg["name"]

            # Translate HH-RLHF
            rlhf_out = os.path.join(RLHF_DATA_DIR, f"hh_rlhf_{args.n_translate_per_lang}_{lang_key}.jsonl")
            if not os.path.exists(rlhf_out):
                translated_rlhf = translate_rlhf_batch(
                    t_client, args.teacher_model, translate_pool, lang_name,
                    args.batch_size, args.max_tokens_translate)
                write_jsonl(translated_rlhf, rlhf_out)
                print(f"  Saved: {rlhf_out} ({len(translated_rlhf)} pairs)")
            else:
                print(f"  Reusing: {rlhf_out}")

            # Translate HHH eval
            hhh_out = os.path.join(HHH_DATA_DIR, f"{lang_key}_gemma3_27b.jsonl")
            if not os.path.exists(hhh_out):
                translated_hhh = translate_hhh_batch(
                    t_client, args.teacher_model, en_hhh, lang_name,
                    args.batch_size, args.max_tokens_translate)
                # Write in original HHH format
                with open(hhh_out, "w", encoding="utf-8") as f:
                    for ex in translated_hhh:
                        line = {
                            "subset": ex["subset"],
                            "input": ex["input"],
                            "target_scores": ex["target_scores"],
                        }
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                print(f"  Saved: {hhh_out} ({len(translated_hhh)} examples)")
            else:
                print(f"  Reusing: {hhh_out}")

        # Step 2: Assemble DPO training dataset
        print(f"\n[Step 1b] Assembling DPO training dataset ...")
        all_dpo_pairs = []

        # Existing translations (en, hi, ml)
        for lang_key, fname in EXISTING_RLHF.items():
            path = os.path.join(RLHF_DATA_DIR, fname)
            if os.path.exists(path):
                recs = load_jsonl(path)
                pairs = build_dpo_pairs_from_rlhf(recs)
                all_dpo_pairs.extend(pairs)
                print(f"  {lang_key}: {len(pairs)} DPO pairs from {fname}")

        # New translations
        for lang_key in NEW_LANG_CONFIGS:
            rlhf_out = os.path.join(RLHF_DATA_DIR,
                                    f"hh_rlhf_{args.n_translate_per_lang}_{lang_key}.jsonl")
            if os.path.exists(rlhf_out):
                recs = load_jsonl(rlhf_out)
                pairs = build_dpo_pairs_from_rlhf(recs)
                all_dpo_pairs.extend(pairs)
                print(f"  {lang_key}: {len(pairs)} DPO pairs from {rlhf_out}")

        random.shuffle(all_dpo_pairs)
        write_jsonl(all_dpo_pairs, args.data_path)
        print(f"\n  Total DPO pairs: {len(all_dpo_pairs)} → {args.data_path}")

        # Dataset sample inspection
        print("\n  DATASET SAMPLES (3 random):")
        for p in random.sample(all_dpo_pairs, min(3, len(all_dpo_pairs))):
            print(f"    prompt: {p['prompt'][:150]}")
            print(f"    chosen: {p['chosen'][:100]}")
            print(f"    rejected: {p['rejected'][:100]}\n")

    if args.data_only:
        print(f"\n[--data-only] Dataset ready. Exiting without training.")
        return

    # Step 3: Train
    print(f"\n[Step 3] Launching DPO+LoRA training on GPUs {args.train_gpus} ...")
    launch_training(args)

    # Step 4: Load adapter
    print(f"\n[Step 4] Loading adapter '{args.adapter_name}' into student vLLM ...")
    load_lora_adapter(args.base_url, args.adapter_name, args.output_dir)
    wait_for_adapter(args.base_url, args.adapter_name)

    # Step 5: Re-eval on all available HHH languages
    print(f"\n[Step 5] Post-DPO evaluation ...")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase3_eval.py")
    eval_cmd = (
        f"{VLLM_PYTHON} {script_path} "
        f"--base-url {args.base_url} "
        f"--model {args.adapter_name} "
        f"--output results/phase3_after.json "
        f"--tag post-dpo "
        f"--batch-size {args.batch_size}"
    )
    subprocess.run(eval_cmd, shell=True, check=True)

    print("\n✓ Phase 3 complete.")


if __name__ == "__main__":
    main()
