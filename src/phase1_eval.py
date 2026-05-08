"""
Phase 1 — MILU Evaluation.

No-think, 0-shot. 250 Hindi + 250 English from test split.
Expects student vLLM running at --base-url (serve.sh).

Usage:
  python phase1_eval.py --model deepseek-r1-8b --tag baseline
  python phase1_eval.py --model phase1 --tag post-sft --output results/phase1_after.json
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

CHOICES = ["A", "B", "C", "D"]
OPTION_KEYS = ["option1", "option2", "option3", "option4"]
GOLD_MAP = dict(zip(OPTION_KEYS, CHOICES))

SYSTEM_NO_THINK = (
    "You are a helpful assistant. Answer the following multiple-choice question. "
    "Do NOT show any reasoning. "
    "Respond with ONLY the single letter of the correct answer (A, B, C, or D)."
)


# ── Shared utilities ─────────────────────────────────────────────────────────

def validate_response(raw: str, finish_reason: str, think_mode: bool = False) -> dict:
    issues = {"overflow": False, "gibberish": False, "empty": False, "warnings": []}
    if not raw or not raw.strip():
        issues["empty"] = True
        return issues
    if finish_reason == "length":
        issues["overflow"] = True
        issues["warnings"].append("finish_reason=length")
    if think_mode and "</think>" not in raw:
        issues["overflow"] = True
        issues["warnings"].append("no </think> tag in think mode")
    text = raw.split("</think>")[-1] if "</think>" in raw else raw
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha < 0.15:
        issues["gibberish"] = True
        issues["warnings"].append(f"low alpha ratio={alpha:.2f}")
    if re.search(r"(.)\1{20,}", text):
        issues["gibberish"] = True
        issues["warnings"].append("repeated char run >20")
    return issues


def append_results_md(row: dict, md_path: str = "final/results.md") -> None:
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


# ── MILU helpers ─────────────────────────────────────────────────────────────

def format_question(ex: dict) -> str:
    q = f"Question: {ex['question']}\n"
    for key, letter in zip(OPTION_KEYS, CHOICES):
        q += f"{letter}. {ex[key]}\n"
    q += "Answer:"
    return q


def extract_answer(raw: str) -> str | None:
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    raw = raw.strip()
    if raw and raw[0].upper() in CHOICES:
        return raw[0].upper()
    for c in raw:
        if c.upper() in CHOICES:
            return c.upper()
    return None


def call_model(client: OpenAI, model: str, system: str,
               user: str, max_tokens: int) -> tuple[str, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or "", resp.choices[0].finish_reason


def run_milu_eval(client: OpenAI, model: str, examples: list, lang: str,
                  batch_size: int, max_tokens: int) -> dict:
    results = [None] * len(examples)
    correct = errors = overflow = gibberish = 0

    print(f"\n{'─'*60}")
    print(f"  MILU {lang} — model={model} — {len(examples)} samples")
    print(f"{'─'*60}")

    with tqdm(total=len(examples), unit="q", dynamic_ncols=True) as pbar:
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

                val = validate_response(raw, finish, think_mode=False)
                if val["overflow"]:
                    overflow += 1
                if val["gibberish"]:
                    gibberish += 1

                pred = extract_answer(raw)
                is_correct = (pred == gold) if (pred and gold) else False
                if is_correct:
                    correct += 1

                results[i] = {
                    "idx": i, "question": ex["question"][:200],
                    "gold": gold, "predicted": pred, "raw": raw[:400],
                    "correct": is_correct, "domain": ex.get("domain", ""),
                    "subject": ex.get("subject", ""), "validation": val,
                }

                pbar.update(1)
                done = sum(1 for r in results if r is not None)
                pbar.set_postfix(acc=f"{correct/done*100:.1f}%",
                                 err=errors, overflow=overflow)

                if done > 20 and overflow / done > 0.20:
                    print(f"\n  ⚠ ALERT: overflow {overflow/done*100:.1f}% > 20% "
                          f"— check max_model_len in serve.sh")

    total = len(examples)
    acc = correct / total * 100
    z = (correct / total - 0.25) / math.sqrt(0.25 * 0.75 / total)
    sig = "✓ sig(p<0.05)" if z > 1.65 else "✗ n.s."
    pred_dist = Counter(r["predicted"] for r in results)

    print(f"\n  Accuracy: {acc:.2f}%  ({correct}/{total})  z={z:+.2f} {sig}")
    print(f"  Errors={errors}  Overflow={overflow}  Gibberish={gibberish}")
    print(f"  Pred dist: {dict(sorted(pred_dist.items(), key=lambda x: str(x[0])))}")

    by_domain = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        d = r["domain"] or "unknown"
        by_domain[d]["t"] += 1
        by_domain[d]["c"] += r["correct"]
    print("  Per-domain:")
    for d, v in sorted(by_domain.items()):
        print(f"    {d:25s}: {v['c']}/{v['t']} = {v['c']/v['t']*100:.1f}%")

    return {
        "language": lang, "model": model,
        "accuracy": round(acc, 4), "correct": correct, "total": total,
        "errors": errors, "overflow": overflow, "gibberish": gibberish,
        "z_score": round(z, 3), "pred_distribution": dict(pred_dist),
        "per_domain": {
            d: {"correct": v["c"], "total": v["t"],
                "accuracy": round(v["c"] / v["t"] * 100, 2)}
            for d, v in by_domain.items()
        },
        "results": results,
    }


def print_samples(results: list, n: int = 10, label: str = "") -> None:
    valid = [r for r in results if r]
    samples = random.sample(valid, min(n, len(valid)))
    print(f"\n{'='*60}")
    print(f"  SAMPLE INSPECTION — {label} ({len(samples)} shown)")
    print(f"{'='*60}")
    for i, r in enumerate(samples):
        status = "✓" if r["correct"] else "✗"
        print(f"\n  [{i+1}] {status} domain={r['domain']} | gold={r['gold']} | pred={r['predicted']}")
        print(f"       Q: {r['question'][:150]}")
        print(f"       raw: {repr(r['raw'][:200])}")
        if r["validation"]["warnings"]:
            print(f"       ⚠ {r['validation']['warnings']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1: MILU eval (no-think, 0-shot)")
    ap.add_argument("--base-url", default="http://localhost:8002/v1")
    ap.add_argument("--model", default="deepseek-r1-8b",
                    help="Model name served by vLLM (base or LoRA adapter name)")
    ap.add_argument("--n-hindi", type=int, default=250)
    ap.add_argument("--n-english", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Allow model to complete think trace; extract_answer handles </think>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/phase1_before.json")
    ap.add_argument("--tag", default="baseline",
                    help="Label for results.md row (e.g. baseline, post-sft)")
    args = ap.parse_args()

    random.seed(args.seed)
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    print(f"Loading ai4bharat/MILU ...")
    ds_hi = load_dataset("ai4bharat/MILU", "Hindi", split="test")
    ds_en = load_dataset("ai4bharat/MILU", "English", split="test")

    hi_subset = list(ds_hi.select(range(min(args.n_hindi, len(ds_hi)))))
    en_subset = list(ds_en.select(range(min(args.n_english, len(ds_en)))))
    print(f"Hindi: {len(hi_subset)}  English: {len(en_subset)}")

    hi_res = run_milu_eval(client, args.model, hi_subset, "Hindi",
                           args.batch_size, args.max_tokens)
    print_samples(hi_res["results"], n=5, label="MILU Hindi")

    en_res = run_milu_eval(client, args.model, en_subset, "English",
                           args.batch_size, args.max_tokens)
    print_samples(en_res["results"], n=5, label="MILU English")

    output = {"model": args.model, "tag": args.tag, "hindi": hi_res, "english": en_res}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")

    append_results_md({
        "phase": "1-SFT", "stage": args.tag, "model_tag": args.model,
        "milu_hi": f"{hi_res['accuracy']:.1f}%",
        "milu_en": f"{en_res['accuracy']:.1f}%",
    })


if __name__ == "__main__":
    main()
