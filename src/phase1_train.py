"""
Phase 1 — MILU SFT Training Pipeline (end-to-end, self-contained).

Workflow:
  1. Build 5k MILU SFT dataset (2.5k Hindi + 2.5k English, gold labels)
  2. Launch SFT+LoRA training via torchrun inside the same apptainer env
  3. Load trained adapter into running student vLLM via API
  4. Re-eval on MILU and append results to final/results.md

Usage (always run inside apptainer via run_overnight.sh):
  python phase1_train.py [--model-path ...] [--output-dir ...]

When run as torchrun worker (LOCAL_RANK set), executes the training code directly.
"""

import argparse
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

VLLM_PYTHON = "/home/compiling-ganesh/24m0797/envs/vllm/bin/python"
CHOICES = ["A", "B", "C", "D"]
OPTION_KEYS = ["option1", "option2", "option3", "option4"]
GOLD_MAP = dict(zip(OPTION_KEYS, CHOICES))

SYSTEM_NO_THINK = (
    "You are a helpful assistant. Answer the following multiple-choice question. "
    "Do NOT show any reasoning. "
    "Respond with ONLY the single letter of the correct answer (A, B, C, or D)."
)


# ── Shared utilities (duplicated for self-containedness) ──────────────────────

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


# ── Dataset building ─────────────────────────────────────────────────────────

def build_milu_sft_dataset(n_hindi: int, n_english: int, seed: int,
                            output_path: str) -> int:
    from datasets import load_dataset as _load

    random.seed(seed)
    records = []

    for lang, n in [("Hindi", n_hindi), ("English", n_english)]:
        print(f"  Loading MILU {lang} train split ...")
        # MILU provides train+validation splits; use both for maximum coverage
        splits = []
        for split_name in ["train", "validation"]:
            try:
                splits.append(list(_load("ai4bharat/MILU", lang, split=split_name)))
            except Exception:
                pass
        pool = [ex for s in splits for ex in s]
        print(f"    {lang}: {len(pool)} available → sampling {n}")
        if len(pool) < n:
            print(f"    ⚠  Only {len(pool)} available for {lang}")
            n = len(pool)
        sample = random.sample(pool, n)

        for ex in sample:
            gold_key = ex.get("target", "")
            gold_letter = GOLD_MAP.get(gold_key)
            if not gold_letter:
                continue
            question_text = (
                f"Question: {ex['question']}\n"
                + "".join(f"{letter}. {ex[key]}\n"
                          for key, letter in zip(OPTION_KEYS, CHOICES))
                + "Answer:"
            )
            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_NO_THINK},
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": gold_letter},
                ]
            })

    random.shuffle(records)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  SFT dataset: {len(records)} records → {output_path}")
    return len(records)


# ── Training worker (runs inside torchrun) ───────────────────────────────────

def run_training_worker(args):
    """Executed by each torchrun worker (LOCAL_RANK is set)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    print(f"[rank {local_rank}] Loading tokenizer from {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[rank {local_rank}] Loading model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # Load data
    with open(args.data_path, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    n_val = max(1, int(len(raw) * 0.05))
    train_ds = Dataset.from_list(raw[n_val:])
    val_ds = Dataset.from_list(raw[:n_val])
    if local_rank == 0:
        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_train,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        max_length=args.max_seq_len,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if local_rank == 0:
        print(f"  Trainable params: {n_trainable:,}")

    trainer.train()
    if local_rank == 0:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  LoRA adapter saved to: {args.output_dir}")


# ── Orchestrator helpers ──────────────────────────────────────────────────────

def launch_training(args):
    """Launch torchrun workers (runs inside apptainer already)."""
    gpus = args.train_gpus
    nproc = len(gpus.split(","))
    script = os.path.abspath(__file__)
    worker_args = " ".join(shlex.quote(a) for a in sys.argv[1:])

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpus} "
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"HTTP_PROXY=http://127.0.0.1:3128 "
        f"HTTPS_PROXY=http://127.0.0.1:3128 "
        f"{VLLM_PYTHON} -m torch.distributed.run "
        f"--nproc_per_node {nproc} "
        f"--master_port 29501 "
        f"{script} {worker_args}"
    )
    print(f"\n[launch_training] GPUs={gpus} nproc={nproc}")
    subprocess.run(cmd, shell=True, check=True)


def load_lora_adapter(base_url: str, name: str, path: str) -> None:
    import requests
    url = base_url.rstrip("/") + "/load_lora_adapter"
    # base_url is like http://localhost:8002/v1, so final URL is /v1/load_lora_adapter
    resp = requests.post(url, json={"lora_name": name, "lora_path": os.path.abspath(path)},
                         timeout=60)
    resp.raise_for_status()
    print(f"  LoRA adapter '{name}' loaded from {path} — HTTP {resp.status_code}")


def wait_for_adapter(base_url: str, model_name: str, max_wait: int = 30) -> None:
    """Poll /v1/models until the adapter name appears."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="dummy")
    for _ in range(max_wait):
        try:
            models = [m.id for m in client.models.list().data]
            if model_name in models:
                print(f"  Adapter '{model_name}' is live. Models: {models}")
                return
        except Exception:
            pass
        time.sleep(2)
    print(f"  ⚠  Adapter '{model_name}' not confirmed after {max_wait*2}s — proceeding anyway")


# ── Post-SFT eval (same logic as phase1_eval.py) ─────────────────────────────

def format_question(ex):
    q = f"Question: {ex['question']}\n"
    for key, letter in zip(OPTION_KEYS, CHOICES):
        q += f"{letter}. {ex[key]}\n"
    q += "Answer:"
    return q


def extract_answer(raw):
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    raw = raw.strip()
    if raw and raw[0].upper() in CHOICES:
        return raw[0].upper()
    for c in raw:
        if c.upper() in CHOICES:
            return c.upper()
    return None


def call_model(client, model, system, user, max_tokens):
    from openai import OpenAI as _OAI
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or "", resp.choices[0].finish_reason


def run_milu_eval(client, model, examples, lang, batch_size, max_tokens):
    from tqdm import tqdm as _tqdm
    results = [None] * len(examples)
    correct = errors = overflow = gibberish = 0

    print(f"\n{'─'*60}")
    print(f"  MILU {lang} — model={model} — {len(examples)} samples")
    print(f"{'─'*60}")

    with _tqdm(total=len(examples), unit="q", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {
                pool.submit(call_model, client, model, SYSTEM_NO_THINK,
                            format_question(ex), max_tokens): (i, ex)
                for i, ex in enumerate(examples)
            }
            for fut in as_completed(futures):
                i, ex = futures[fut]
                gold = GOLD_MAP.get(ex.get("target", ""))
                try:
                    raw, finish = fut.result()
                except Exception:
                    raw, finish = "", "error"
                    errors += 1
                val = validate_response(raw, finish)
                if val["overflow"]: overflow += 1
                if val["gibberish"]: gibberish += 1
                pred = extract_answer(raw)
                is_correct = pred == gold if (pred and gold) else False
                if is_correct: correct += 1
                results[i] = {
                    "idx": i, "question": ex["question"][:200], "gold": gold,
                    "predicted": pred, "raw": raw[:400], "correct": is_correct,
                    "domain": ex.get("domain", ""), "validation": val,
                }
                pbar.update(1)
                done = sum(1 for r in results if r is not None)
                pbar.set_postfix(acc=f"{correct/done*100:.1f}%", err=errors, overflow=overflow)
                if done > 20 and overflow / done > 0.20:
                    print(f"\n  ⚠ ALERT: overflow {overflow/done*100:.1f}% > 20%!")

    total = len(examples)
    acc = correct / total * 100
    z = (correct / total - 0.25) / math.sqrt(0.25 * 0.75 / total)
    pred_dist = Counter(r["predicted"] for r in results)
    print(f"\n  Accuracy: {acc:.2f}%  ({correct}/{total})  z={z:+.2f}")
    print(f"  Errors={errors}  Overflow={overflow}  Gibberish={gibberish}")

    # Sample inspection
    samples = random.sample([r for r in results if r], min(5, len(results)))
    print(f"\n  SAMPLE INSPECTION ({len(samples)} random):")
    for r in samples:
        status = "✓" if r["correct"] else "✗"
        print(f"    {status} gold={r['gold']} pred={r['predicted']} "
              f"domain={r['domain']} raw={repr(r['raw'][:100])}")

    return {
        "language": lang, "accuracy": round(acc, 4), "correct": correct,
        "total": total, "errors": errors, "overflow": overflow, "z_score": round(z, 3),
        "pred_distribution": dict(pred_dist), "results": results,
    }


def run_post_sft_eval(args, adapter_name: str) -> None:
    from datasets import load_dataset as _load
    from openai import OpenAI

    random.seed(args.seed)
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    print(f"\nPost-SFT eval — model={adapter_name}")
    ds_hi = list(_load("ai4bharat/MILU", "Hindi", split="test"))
    ds_en = list(_load("ai4bharat/MILU", "English", split="test"))
    hi_sub = random.sample(ds_hi, min(args.n_eval, len(ds_hi)))
    en_sub = random.sample(ds_en, min(args.n_eval, len(ds_en)))

    hi_res = run_milu_eval(client, adapter_name, hi_sub, "Hindi",
                           args.batch_size, args.max_tokens)
    en_res = run_milu_eval(client, adapter_name, en_sub, "English",
                           args.batch_size, args.max_tokens)

    output = {
        "model": adapter_name, "tag": "post-sft",
        "hindi": hi_res, "english": en_res,
    }
    out_path = "results/phase1_after.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nPost-SFT results saved to: {out_path}")

    # Update detailed results.md
    script_dir = os.path.dirname(os.path.abspath(__file__))
    update_script = os.path.join(script_dir, "update_results.py")
    subprocess.run(f"{VLLM_PYTHON} {update_script} phase1 {out_path}", shell=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Phase 1: MILU SFT pipeline")
    ap.add_argument("--model-path", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--output-dir", default="checkpoints/phase1_lora")
    ap.add_argument("--data-path", default="data/phase1_milu_sft.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8002/v1")
    ap.add_argument("--adapter-name", default="phase1")
    ap.add_argument("--train-gpus", default="2")
    ap.add_argument("--n-hindi", type=int, default=2500)
    ap.add_argument("--n-english", type=int, default=2500)
    ap.add_argument("--n-eval", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size-train", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    return ap.parse_args()


def main():
    args = parse_args()
    # batch_size is for eval (128 concurrent); batch_size_train is for training (4)

    # ── Running as torchrun worker ────────────────────────────────────────────
    if "LOCAL_RANK" in os.environ:
        run_training_worker(args)
        return

    # ── Orchestrator path ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  Phase 1 — MILU SFT Pipeline")
    print("=" * 60)

    # Step 1: Build dataset
    print(f"\n[Step 1] Building MILU SFT dataset ({args.n_hindi}+{args.n_english} samples)...")
    build_milu_sft_dataset(args.n_hindi, args.n_english, args.seed, args.data_path)

    # Step 2: Train
    print(f"\n[Step 2] Launching SFT+LoRA training on GPUs {args.train_gpus}...")
    launch_training(args)

    # Step 3: Load adapter
    print(f"\n[Step 3] Loading LoRA adapter '{args.adapter_name}' into vLLM...")
    load_lora_adapter(args.base_url, args.adapter_name, args.output_dir)
    wait_for_adapter(args.base_url, args.adapter_name)

    # Step 4: Re-eval
    print(f"\n[Step 4] Post-SFT evaluation...")
    run_post_sft_eval(args, args.adapter_name)

    print("\n✓ Phase 1 complete.")


if __name__ == "__main__":
    main()
