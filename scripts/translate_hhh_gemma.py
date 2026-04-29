"""
Translate data/hhh_alignment/english.jsonl to Hindi and Malayalam
using Gemma 3 27B served at localhost:8003.

Runs both languages in parallel with ThreadPoolExecutor.
Output: data/hhh_alignment/hindi_gemma3_27b.jsonl
        data/hhh_alignment/malayalam_gemma3_27b.jsonl
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

BASE_URL = "http://localhost:8003/v1"
MODEL    = "gemma3-27b"
SRC      = "data/hhh_alignment/english.jsonl"

LANG_CFG = {
    "hindi":     {"script": "Hindi (Devanagari script)", "out": "data/hhh_alignment/hindi_gemma3_27b.jsonl"},
    "malayalam": {"script": "Malayalam (Malayalam script)",  "out": "data/hhh_alignment/malayalam_gemma3_27b.jsonl"},
}

SYSTEM = (
    "You are a professional translator. Translate the given English text to {lang} accurately and naturally. "
    "Rules:\n"
    "- Output ONLY the translated text, nothing else — no explanation, no prefix, no quotes\n"
    "- Keep proper nouns (names like Carl, Cassie, Imani, Arnold Palmer, Hyundai, etc.) in English\n"
    "- Keep code, URLs, commands (rm -rf, wget, etc.) unchanged\n"
    "- Keep numbers, dates, letter labels (A, B) unchanged\n"
    "- Preserve newlines and formatting exactly\n"
    "- Translate naturally, not word-for-word"
)


def translate(client, text, lang_label):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM.format(lang=lang_label)},
            {"role": "user",   "content": text},
        ],
        max_tokens=2048,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def translate_example(client, ex, lang_label):
    input_tr = translate(client, ex["input"], lang_label)
    scores_tr = {}
    for key, val in ex["target_scores"].items():
        key_tr = translate(client, key, lang_label)
        scores_tr[key_tr] = val
    return {
        "subset": ex["subset"],
        "input": input_tr,
        "target_scores": scores_tr,
    }


def translate_dataset(lang, cfg, examples, batch=128):
    client = OpenAI(base_url=BASE_URL, api_key="dummy")
    results = [None] * len(examples)
    errors = 0
    print(f"\n── Translating to {lang} ({len(examples)} examples) ──")
    with tqdm(total=len(examples), unit="ex") as pbar:
        with ThreadPoolExecutor(max_workers=batch) as pool:
            futs = {pool.submit(translate_example, client, ex, cfg["script"]): i
                    for i, ex in enumerate(examples)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = examples[i]  # fallback to English on error
                    errors += 1
                    print(f"  [err idx={i}] {e}")
                pbar.update(1)
    print(f"  Done. errors={errors}")
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    with open(cfg["out"], "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved: {cfg['out']}")
    return results


def main():
    examples = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {SRC}")

    # Run both languages in parallel using threads at the dataset level
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {
            pool.submit(translate_dataset, lang, cfg, examples): lang
            for lang, cfg in LANG_CFG.items()
        }
        results = {}
        for fut in as_completed(futs):
            lang = futs[fut]
            results[lang] = fut.result()

    # Print a few samples for manual inspection
    print("\n══ Sample inspection (first 3 examples) ══")
    en = examples[:3]
    hi = results.get("hindi", [])[:3]
    ml = results.get("malayalam", [])[:3]
    for i in range(min(3, len(en))):
        print(f"\n─── Example {i} (subset={en[i]['subset']}) ───")
        print(f"EN input : {en[i]['input'][:120]}")
        if i < len(hi): print(f"HI input : {hi[i]['input'][:120]}")
        if i < len(ml): print(f"ML input : {ml[i]['input'][:120]}")
        choices_en = list(en[i]['target_scores'].keys())
        if i < len(hi):
            choices_hi = list(hi[i]['target_scores'].keys())
            print(f"EN reply1: {choices_en[0][:80]}")
            print(f"HI reply1: {choices_hi[0][:80]}")
        if i < len(ml):
            choices_ml = list(ml[i]['target_scores'].keys())
            print(f"ML reply1: {choices_ml[0][:80]}")


if __name__ == "__main__":
    main()
