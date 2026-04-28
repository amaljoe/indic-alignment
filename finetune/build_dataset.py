"""
Merge, deduplicate and format all teacher JSONL files into a single train.jsonl
compatible with trl SFTTrainer (list-of-messages format).
"""

import json
import os
import random

DATA_DIR = "finetune/data"
OUTPUT   = "finetune/data/train.jsonl"
SEED     = 42

FILES = [
    ("normad_teacher.jsonl",       3.0),   # upweight: most important task
    ("milu_en_teacher.jsonl",      1.0),
    ("milu_hi_teacher.jsonl",      1.0),
    ("bhed_teacher.jsonl",         2.0),   # upweight: debiasing
    ("globalopinion_teacher.jsonl",1.0),
]

random.seed(SEED)


def load_jsonl(path):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def dedup(records):
    seen = set()
    out = []
    for r in records:
        key = r["messages"][0]["content"][:120]  # dedupe by start of user prompt
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    all_records = []
    for fname, weight in FILES:
        path = os.path.join(DATA_DIR, fname)
        recs = load_jsonl(path)
        # Integer oversampling by weight
        n_copies = max(1, round(weight))
        expanded = recs * n_copies
        print(f"  {fname}: {len(recs)} → {len(expanded)} (×{n_copies})")
        all_records.extend(expanded)

    before = len(all_records)
    all_records = dedup(all_records)
    print(f"\nBefore dedup: {before}  After: {len(all_records)}")

    random.shuffle(all_records)

    # Keep only messages field for SFTTrainer
    output_records = [{"messages": r["messages"]} for r in all_records]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(output_records)} training examples → {OUTPUT}")

    # Print source distribution
    from collections import Counter
    srcs = Counter(r.get("source", "unknown") for r in all_records)
    print("Source distribution:")
    for src, n in sorted(srcs.items(), key=lambda x: -x[1]):
        print(f"  {src:<35} {n}")


if __name__ == "__main__":
    main()
