#!/usr/bin/env python3
"""
filter_milu_india.py — Filter MILU English to India-specific questions only.

MILU comes from Indian competitive exams (UPSC, SSC, state PSC, JEE, NEET etc.)
so all questions have some Indian context, but Science/Engineering domains
often have universal content. This script keeps only questions with explicit
India references.

Also optionally samples from sarvamai/mmlu-indic as a supplement.

Usage:
  python filter_milu_india.py --output data/milu_india_en.jsonl
  python filter_milu_india.py --output data/milu_india_en.jsonl --include-mmlu-indic
"""

import argparse
import json
import os
import random as _random
import re
from collections import Counter

OPTION_KEYS = ["option1", "option2", "option3", "option4"]
CHOICES = ["A", "B", "C", "D"]
GOLD_MAP = dict(zip(OPTION_KEYS, CHOICES))

# India-specific keywords — explicit references to India, its regions, institutions,
# history, culture, people, language, law, geography.
INDIA_KEYWORDS = [
    # Country / nationality
    r"\bindian?\b", r"\bindia\b", r"\bbharat\b",
    # States / union territories
    r"\bgujarat\b", r"\bmahara.?shtra\b", r"\brajasthan\b", r"\bpunjab\b",
    r"\bkerala\b", r"\btamil.?nad\b", r"\bkarnatak\b", r"\btelangana\b",
    r"\bandhra\b", r"\bodisha\b", r"\borissa\b", r"\bjharkhand\b",
    r"\bchhattisgarh\b", r"\bbihar\b", r"\buttar.?prad\b", r"\bmadhy?a.?prad\b",
    r"\bassam\b", r"\bmanipur\b", r"\bnagaland\b", r"\bmeghalay\b",
    r"\btripura\b", r"\bmizoram\b", r"\barunachal\b", r"\bsikkim\b",
    r"\bgoa\b", r"\bhimachal\b", r"\buttarakhand\b", r"\bharyana\b",
    r"\bjammu\b", r"\bkashmir\b", r"\blakshadweep\b", r"\bandaman\b",
    r"\bputtu?cheri\b", r"\bpondicherry\b", r"\bchandigarh\b",
    # Cities
    r"\bdelhi\b", r"\bnew.?delhi\b", r"\bmumbai\b", r"\bbombay\b",
    r"\bkolkata\b", r"\bcalcutta\b", r"\bchennai\b", r"\bmadras\b",
    r"\bhyder.?abad\b", r"\bbenGaluru\b", r"\bbangalore\b", r"\bpune\b",
    r"\bahmedabad\b", r"\bjaipur\b", r"\blucknow\b", r"\bpatna\b",
    r"\bbhopal\b", r"\bindore\b", r"\bnagpur\b", r"\bsurat\b",
    r"\bvaranasi\b", r"\bkochi\b", r"\bcoimbatore\b", r"\bvisakhapatnam\b",
    r"\bamritsar\b", r"\bbhubaneswar\b", r"\bguwahati\b", r"\bagra\b",
    # Rivers / geography
    r"\bganga\b", r"\bganges\b", r"\byamuna\b", r"\bbrahmaputra\b",
    r"\bnarmada\b", r"\bgodavari\b", r"\bkrishna\b", r"\bcauvery\b",
    r"\bhimalay\b", r"\bwestern.?ghat\b", r"\beastern.?ghat\b",
    r"\bsundarban\b", r"\bthar\b", r"\bdeccan\b",
    # Institutions / Government
    r"\blok.?sabha\b", r"\brajya.?sabha\b", r"\bparliament.of.india\b",
    r"\bsupreme.court.of.india\b", r"\bhigh.court\b", r"\bupsc\b",
    r"\bisro\b", r"\bdrdo\b", r"\biit\b", r"\biim\b", r"\baiims\b",
    r"\bniti.?aayog\b", r"\bplanning.commission\b", r"\brbi\b",
    r"\breserve.bank\b", r"\bsei?bi?\b", r"\bsebi\b", r"\bnse\b", r"\bbse\b",
    r"\bgspc?\b",
    # Historical figures / people
    r"\bmahatma.?gandhi\b", r"\bjawaharlal\b", r"\bnehru\b",
    r"\bambedkar\b", r"\bsubhas.?chandra\b", r"\bbose\b", r"\btagore\b",
    r"\bpatel\b", r"\brabin?dra\b", r"\bswami.?vivekanand\b",
    r"\baudrey.?tang\b", r"\bbal.?gangadhar\b", r"\btilak\b",
    r"\bbhagat.?singh\b", r"\bchandra.?shekhar\b", r"\bindira\b",
    r"\brajiv.?gandhi\b", r"\bnamo\b", r"\bnarendramodi\b", r"\bmodi\b",
    r"\bmanmohan\b", r"\bapj\b", r"\bkalam\b", r"\bpranab\b",
    r"\bkovind\b", r"\bdroupadi\b",
    # Religion / culture
    r"\bhinduism\b", r"\bhindu\b", r"\bislam.in.india\b",
    r"\bsikhism\b", r"\bjainism\b", r"\bbuddhism.in.india\b",
    r"\bdiwali\b", r"\bholi\b", r"\beid\b", r"\bnavratri\b",
    r"\bpongal\b", r"\bonam\b", r"\bbaisakhi\b", r"\bganesh\b",
    r"\bdurga.?puja\b", r"\bramadan\b",
    # Languages
    r"\bhindi\b", r"\bsanskrit\b", r"\bbengali\b", r"\btamil\b",
    r"\btelugu\b", r"\bmarathi\b", r"\bgujarati\b", r"\bpunjabi\b",
    r"\bkannada\b", r"\bmalayalam\b", r"\bodia\b", r"\burdu\b",
    r"\bassamese\b", r"\bmaithili\b", r"\bkashmir[i]?\b",
    # Law / constitution
    r"\bconstitution.of.india\b", r"\bfundamental.rights\b",
    r"\bdirective.principles\b", r"\bpreamble.of\b", r"\bipcs?\b",
    r"\bcrpc\b", r"\bnational.emergency\b", r"\bpresident.of.india\b",
    r"\bgovernor.general\b", r"\bvice.?president.of.india\b",
    # Awards / competitions
    r"\bbharat.?ratna\b", r"\bpadma.?(vibhushan|bhushan|shri)\b",
    r"\bnational.award\b", r"\bfilmfare\b", r"\bgrammy.india\b",
    r"\bolympic.india\b", r"\bcricket.india\b", r"\bbcci\b",
    # Economy / finance
    r"\bgst\b", r"\bdemonetiz\b", r"\bmake.in.india\b",
    r"\bswachh.?bharat\b", r"\bajay\b", r"\bunion.budget\b",
    r"\bfive.year.plan\b",
]

# Compile patterns (case-insensitive)
_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in INDIA_KEYWORDS]


def is_india_specific(question: str, options: list[str]) -> bool:
    text = question + " " + " ".join(options)
    return any(p.search(text) for p in _PATTERNS)


def build_sft_record(ex, split_label="") -> dict:
    gold = GOLD_MAP.get(ex.get("target", ""))
    if not gold:
        return None
    q = (
        f"Question: {ex['question']}\n"
        + "".join(f"{l}. {ex[k]}\n" for k, l in zip(OPTION_KEYS, CHOICES))
        + "Answer:"
    )
    return {
        "messages": [
            {"role": "system", "content":
                "You are a helpful assistant. Answer the following multiple-choice question. "
                "Respond with ONLY the single letter of the correct answer (A, B, C, or D)."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": gold},
        ],
        "domain": ex.get("domain", ""),
        "split": split_label,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/milu_india_en.jsonl")
    ap.add_argument("--split", default="train",
                    help="MILU split to filter: train, validation, test")
    ap.add_argument("--n-max", type=int, default=5000,
                    help="Max examples to keep after filtering")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-mmlu-indic", action="store_true",
                    help="Also sample from sarvamai/mmlu-indic (English)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print statistics only, do not write output")
    args = ap.parse_args()

    from datasets import load_dataset as _load

    _random.seed(args.seed)

    print(f"Loading MILU English {args.split} split...")
    all_examples = []
    for split_name in args.split.split(","):
        try:
            split_data = list(_load("ai4bharat/MILU", "English", split=split_name.strip()))
            print(f"  {split_name}: {len(split_data)} examples")
            all_examples.extend(split_data)
        except Exception as e:
            print(f"  Skipping {split_name}: {e}")

    print(f"Total MILU English: {len(all_examples)}")

    # Filter
    india_examples = []
    for ex in all_examples:
        opts = [ex.get(k, "") for k in OPTION_KEYS]
        if is_india_specific(ex.get("question", ""), opts):
            india_examples.append(ex)

    print(f"India-specific: {len(india_examples)} / {len(all_examples)} "
          f"({len(india_examples)/max(len(all_examples),1)*100:.1f}%)")

    # Domain breakdown
    domain_counts = Counter(ex.get("domain", "unknown") for ex in india_examples)
    print("\nPer-domain:")
    for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
        total_d = sum(1 for ex in all_examples if ex.get("domain") == d)
        print(f"  {d:35s}: {n:4d} / {total_d:4d} ({n/max(total_d,1)*100:.0f}% India-specific)")

    if args.dry_run:
        print("\n[dry-run] Not writing output.")
        return

    # Sample
    _random.shuffle(india_examples)
    selected = india_examples[:args.n_max]
    print(f"\nSelected {len(selected)} examples (max={args.n_max})")

    records = []
    for ex in selected:
        r = build_sft_record(ex, "milu_india_en")
        if r:
            records.append(r)

    # Optionally add mmlu-indic
    if args.include_mmlu_indic:
        print("\nLoading sarvamai/mmlu-indic (English)...")
        try:
            mmlu_ds = list(_load("sarvamai/mmlu-indic", "English", split="test"))
            print(f"  mmlu-indic English test: {len(mmlu_ds)} examples")
            for ex in mmlu_ds:
                # mmlu-indic format may differ — adapt as needed
                q_text = ex.get("question", "")
                choices_raw = ex.get("choices", []) or ex.get("options", [])
                answer_key = ex.get("answer", ex.get("target", ""))
                if not (q_text and choices_raw and answer_key):
                    continue
                if isinstance(answer_key, int):
                    gold = CHOICES[answer_key] if answer_key < 4 else None
                elif isinstance(answer_key, str) and answer_key.upper() in CHOICES:
                    gold = answer_key.upper()
                else:
                    continue
                if not gold:
                    continue
                q = f"Question: {q_text}\n"
                for j, ch in enumerate(choices_raw[:4]):
                    q += f"{CHOICES[j]}. {ch}\n"
                q += "Answer:"
                records.append({
                    "messages": [
                        {"role": "system", "content":
                            "You are a helpful assistant. Answer the following multiple-choice question. "
                            "Respond with ONLY the single letter of the correct answer (A, B, C, or D)."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": gold},
                    ],
                    "domain": ex.get("subject", "general"),
                    "split": "mmlu_indic_en",
                })
            print(f"  Added {sum(1 for r in records if r.get('split')=='mmlu_indic_en')} mmlu-indic records")
        except Exception as e:
            print(f"  Could not load mmlu-indic: {e}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} records → {args.output}")

    # Show 3 sample questions
    samples = _random.sample(records, min(3, len(records)))
    print("\nSample questions:")
    for s in samples:
        msgs = s["messages"]
        print(f"  [{s['domain']}] {msgs[1]['content'][:120].strip()}")
        print(f"  Answer: {msgs[2]['content']}")


if __name__ == "__main__":
    main()
