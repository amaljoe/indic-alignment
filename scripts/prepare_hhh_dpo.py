"""
Convert HHH alignment JSONL files to DPO training format.

Input:  {"subset": ..., "input": ..., "target_scores": {"resp_a": 1, "resp_b": 0}}
Output: {"prompt": "\n\nHuman: <input>\n\nAssistant:", "chosen": "<resp score=1>",
         "rejected": "<resp score=0>", "lang": "<lang>"}

Usage:
  python scripts/prepare_hhh_dpo.py \
      --out data/hhh_dpo_train.jsonl
"""
import argparse
import json
import os

FILES = [
    ("data/hhh_alignment/english.jsonl",            "en"),
    ("data/hhh_alignment/hindi_gemma3_27b.jsonl",   "hi"),
    ("data/hhh_alignment/malayalam_gemma3_27b.jsonl","ml"),
]


def convert(path, lang_tag):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            scores = ex["target_scores"]
            chosen = rejected = None
            for resp, score in scores.items():
                if score == 1:
                    chosen = resp.strip()
                else:
                    rejected = resp.strip()
            if chosen is None or rejected is None:
                continue
            prompt = f"\n\nHuman: {ex['input'].strip()}\n\nAssistant:"
            rows.append({
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
                "lang":     lang_tag,
                "subset":   ex.get("subset", ""),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hhh_dpo_train.jsonl")
    args = ap.parse_args()

    all_rows = []
    for path, tag in FILES:
        rows = convert(path, tag)
        print(f"  {tag}: {len(rows)} pairs from {path}")
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} total pairs → {args.out}")


if __name__ == "__main__":
    main()
