"""
MILU alignment evaluation — parallel inference with progress bar.

Modes: zero-shot/few-shot × with-thinking/no-thinking
"""

import argparse
import json
import random
import math
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

CHOICES = ["A", "B", "C", "D"]
OPTION_KEYS = ["option1", "option2", "option3", "option4"]
TARGET_TO_LETTER = {k: v for k, v in zip(OPTION_KEYS, CHOICES)}

SYSTEM_WITH_THINK = (
    "You are a helpful assistant. Answer the following multiple-choice question. "
    "Think step by step, then respond with only the letter of the correct answer "
    "(A, B, C, or D) on the final line."
)
SYSTEM_NO_THINK = (
    "You are a helpful assistant. Answer the following multiple-choice question. "
    "Do NOT show any reasoning. "
    "Respond with ONLY the single letter of the correct answer (A, B, C, or D)."
)


def format_question(ex, include_answer=False):
    prompt = f"Question: {ex['question']}\n"
    for key, letter in zip(OPTION_KEYS, CHOICES):
        prompt += f"{letter}. {ex[key]}\n"
    prompt += "Answer:"
    if include_answer:
        prompt += f" {TARGET_TO_LETTER[ex['target']]}"
    return prompt


def build_few_shot_prefix(val_examples, n):
    shots = random.sample(val_examples, min(n, len(val_examples)))
    return "\n\n".join(format_question(ex, include_answer=True) for ex in shots) + "\n\n"


def extract_answer(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    if text and text[0].upper() in CHOICES:
        return text[0].upper()
    for ch in text:
        if ch.upper() in CHOICES:
            return ch.upper()
    return None


def call_model(client, model, system_prompt, user_content, max_tokens):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def run_eval(client, subset, few_shot_prefix, system_prompt, model,
             max_tokens, label, batch_size):
    results = [None] * len(subset)
    correct = 0
    errors = 0

    print(f"\n{'─'*60}")
    print(f"  Mode: {label}")
    print(f"{'─'*60}")

    with tqdm(total=len(subset), unit="q", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for i, ex in enumerate(subset):
                user_content = few_shot_prefix + format_question(ex)
                fut = pool.submit(call_model, client, model, system_prompt,
                                  user_content, max_tokens)
                futures[fut] = (i, ex)

            for fut in as_completed(futures):
                i, ex = futures[fut]
                gold = TARGET_TO_LETTER.get(ex["target"])
                try:
                    raw = fut.result()
                    pred = extract_answer(raw)
                except Exception as e:
                    raw, pred = "", None
                    errors += 1

                is_correct = pred == gold if (pred and gold) else False
                results[i] = {
                    "idx": i,
                    "question": ex["question"],
                    "gold": gold,
                    "predicted": pred,
                    "raw_response": raw,
                    "correct": is_correct,
                    "domain": ex.get("domain", ""),
                    "subject": ex.get("subject", ""),
                }
                if is_correct:
                    correct += 1
                pbar.update(1)
                done = sum(1 for r in results if r is not None)
                pbar.set_postfix(acc=f"{correct/done*100:.1f}%", err=errors)

    total = len(subset)
    accuracy = correct / total * 100
    z = (correct/total - 0.25) / math.sqrt(0.25 * 0.75 / total)
    sig = "✓ sig(p<0.05)" if z > 1.65 else "✗ n.s."
    print(f"  Accuracy: {accuracy:.2f}%  ({correct}/{total})  z={z:+.2f} {sig}  errors={errors}")

    subj = defaultdict(lambda: {"c": 0, "t": 0})
    dom  = defaultdict(lambda: {"c": 0, "t": 0})
    pred_dist = Counter(r["predicted"] for r in results)
    for r in results:
        s, d = r["subject"] or "unknown", r["domain"] or "unknown"
        subj[s]["t"] += 1; subj[s]["c"] += r["correct"]
        dom[d]["t"]  += 1; dom[d]["c"]  += r["correct"]

    print(f"  Pred dist: {dict(sorted(pred_dist.items(), key=lambda kv: (kv[0] is None, str(kv[0]))))}")

    return {
        "label": label,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "errors": errors,
        "z_score": round(z, 3),
        "pred_distribution": dict(pred_dist),
        "per_domain":  {k: {"correct": v["c"], "total": v["t"],
                            "accuracy": round(v["c"]/v["t"]*100, 2)} for k, v in dom.items()},
        "per_subject": {k: {"correct": v["c"], "total": v["t"],
                            "accuracy": round(v["c"]/v["t"]*100, 2)} for k, v in subj.items()},
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="deepseek-r1")
    parser.add_argument("--base-url",    default="http://localhost:8002/v1")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--language",    default="Hindi")
    parser.add_argument("--few-shot-n",  type=int, default=5)
    parser.add_argument("--batch-size",  type=int, default=128,
                        help="Parallel requests to vLLM (server queues automatically)")
    parser.add_argument("--output",      default="milu_results.json")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading ai4bharat/MILU [{args.language}] ...")
    ds_test = load_dataset("ai4bharat/MILU", args.language, split="test")
    ds_val  = load_dataset("ai4bharat/MILU", args.language, split="validation")
    print(f"Test: {len(ds_test)}  Val: {len(ds_val)}")

    subset        = ds_test.select(range(min(args.num_samples, len(ds_test))))
    val_list      = list(ds_val)
    few_shot_prefix = build_few_shot_prefix(val_list, args.few_shot_n)

    client = OpenAI(base_url=args.base_url, api_key="dummy")

    configs = [
        ("zero-shot + thinking",    "",               SYSTEM_WITH_THINK, 512),
        ("zero-shot + no-thinking", "",               SYSTEM_NO_THINK,    16),
        ("few-shot  + thinking",    few_shot_prefix,  SYSTEM_WITH_THINK, 512),
        ("few-shot  + no-thinking", few_shot_prefix,  SYSTEM_NO_THINK,    16),
    ]

    all_results = {}
    for label, prefix, sys_prompt, max_tok in configs:
        all_results[label] = run_eval(
            client, subset, prefix, sys_prompt,
            args.model, max_tok, label, args.batch_size
        )

    print(f"\n{'='*60}")
    print(f"  MILU {args.language} — {args.model} — {args.num_samples} samples")
    print(f"{'='*60}")
    print(f"  {'Mode':<35} {'Acc':>7}  {'z':>6}  Pred dist")
    print(f"  {'─'*56}")
    for label, res in all_results.items():
        sig = "✓" if res["z_score"] > 1.65 else "✗"
        print(f"  {label:<35} {res['accuracy']:6.2f}%  {res['z_score']:+5.2f}{sig}  {res['pred_distribution']}")
    print(f"{'='*60}\n")

    output = {
        "model": args.model,
        "language": args.language,
        "num_samples": args.num_samples,
        "few_shot_n": args.few_shot_n,
        "batch_size": args.batch_size,
        "modes": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
