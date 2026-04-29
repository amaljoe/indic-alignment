"""
One-shot probe: run MILU 'zero-shot + thinking' on both 1.5B and 8B with a
generous max_tokens budget, then report the *actual* completion length per
sample so we can pick a sensible production budget.

Reports, per model:
  - p50 / p90 / p95 / p99 / max  completion-token length
  - fraction whose finish_reason == "length"  (i.e. STILL truncated even at the
    high budget)
  - fraction with </think> closed
  - fraction with a parseable letter

Output: prints summary + saves results/milu_thinklen_probe.json.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from openai import OpenAI

CHOICES = ["A", "B", "C", "D"]
OPTION_KEYS = ["option1", "option2", "option3", "option4"]
TARGET_TO_LETTER = {k: v for k, v in zip(OPTION_KEYS, CHOICES)}

SYSTEM_WITH_THINK = (
    "You are a helpful assistant. Answer the following multiple-choice question. "
    "Think step by step, then respond with only the letter of the correct answer "
    "(A, B, C, or D) on the final line."
)


def format_question(ex):
    s = f"Question: {ex['question']}\n"
    for key, letter in zip(OPTION_KEYS, CHOICES):
        s += f"{letter}. {ex[key]}\n"
    return s + "Answer:"


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


def call(client, model, prompt, max_tokens):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_WITH_THINK},
                  {"role": "user",   "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    msg = resp.choices[0].message.content or ""
    fr  = resp.choices[0].finish_reason
    usage = resp.usage
    return msg, fr, (usage.completion_tokens if usage else None)


def run(server, model, samples, max_tokens, workers):
    client = OpenAI(base_url=server, api_key="dummy")
    out = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(call, client, model, format_question(ex), max_tokens): i
                for i, ex in enumerate(samples)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                msg, fr, ntok = fut.result()
            except Exception as e:
                msg, fr, ntok = "", f"ERROR: {type(e).__name__}: {e}", None
            out[i] = {
                "completion_tokens": ntok,
                "finish_reason": fr,
                "closed_think": "</think>" in msg,
                "predicted": extract_answer(msg),
                "gold": TARGET_TO_LETTER[samples[i]["target"]],
            }
    return out


def percentile(xs, p):
    xs = sorted(xs)
    if not xs: return 0
    k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarise(rows):
    n = len(rows)
    toks = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
    return {
        "n": n,
        "trunc_pct": 100 * sum(1 for r in rows if r["finish_reason"] == "length") / n,
        "closed_think_pct": 100 * sum(r["closed_think"] for r in rows) / n,
        "parsed_pct": 100 * sum(1 for r in rows if r["predicted"] is not None) / n,
        "acc_pct":    100 * sum(1 for r in rows if r["predicted"] == r["gold"]) / n,
        "tok_p50": percentile(toks, 50),
        "tok_p90": percentile(toks, 90),
        "tok_p95": percentile(toks, 95),
        "tok_p99": percentile(toks, 99),
        "tok_avg": sum(toks)/len(toks) if toks else 0,
        "tok_max": max(toks) if toks else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--language", default="Hindi")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="generous one-shot budget")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default="results/milu_thinklen_probe.json")
    args = ap.parse_args()

    print(f"Loading MILU [{args.language}] ...")
    ds = load_dataset("ai4bharat/MILU", args.language, split="test")
    samples = list(ds.select(range(args.num_samples)))

    servers = [
        ("1.5B", "http://localhost:8002/v1", "deepseek-r1-1p5b"),
        ("8B",   "http://localhost:8003/v1", "deepseek-r1-8b"),
    ]

    summaries = {}
    print(f"\nProbing zero-shot+thinking with max_tokens={args.max_tokens}, "
          f"n={args.num_samples}, lang={args.language}")
    print(f"\n{'model':6s} {'avg':>6s} {'p50':>6s} {'p90':>6s} {'p95':>6s} "
          f"{'p99':>6s} {'max':>6s} {'still_trunc%':>13s} "
          f"{'closed%':>9s} {'parsed%':>8s} {'acc%':>6s}")
    full = {}
    for label, url, model in servers:
        rows = run(url, model, samples, args.max_tokens, args.workers)
        s = summarise(rows)
        summaries[label] = s
        full[label] = rows
        print(f"{label:6s} {s['tok_avg']:>6.0f} {s['tok_p50']:>6.0f} "
              f"{s['tok_p90']:>6.0f} {s['tok_p95']:>6.0f} {s['tok_p99']:>6.0f} "
              f"{s['tok_max']:>6.0f} {s['trunc_pct']:>12.1f}% "
              f"{s['closed_think_pct']:>8.1f}% {s['parsed_pct']:>7.1f}% "
              f"{s['acc_pct']:>5.1f}%")

    print("\nRecommendation: pick max_tokens ≥ p99 (rounded to power of two) so "
          "<1% are truncated; budget = max for 0% truncation.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"max_tokens_budget": args.max_tokens,
                   "language": args.language,
                   "num_samples": args.num_samples,
                   "summaries": summaries,
                   "rows": full}, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
