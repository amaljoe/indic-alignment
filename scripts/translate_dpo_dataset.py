"""
Translate a DPO-format JSONL dataset to a target language using vLLM.

Input JSONL: {"chosen": "<full hh-rlhf conversation>", "rejected": "<...>"}
Output JSONL: {"chosen": "<translated>", "rejected": "<translated>"}

Each example is translated in a SINGLE inference call (chosen + rejected together).
Uses curated few-shot examples per language for quality.

Usage:
  python scripts/translate_dpo_dataset.py \
      --input data/hh_rlhf/hh_rlhf_5k_en.jsonl \
      --lang hindi \
      --output data/hh_rlhf/hh_rlhf_5k_hindi.jsonl \
      --base-url http://localhost:8003/v1 \
      --model gemma3-27b \
      --batch 128
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ── Few-shot examples (curated) ───────────────────────────────────────────────
FEW_SHOT = {
    "hindi": [
        {
            "chosen": "\n\nHuman: What is the capital of France?\n\nAssistant: The capital of France is Paris. It is one of the most visited cities in the world, known for the Eiffel Tower and its rich culture.",
            "rejected": "\n\nHuman: What is the capital of France?\n\nAssistant: I am not entirely sure, but I think it might be Lyon.",
            "chosen_hi": "\n\nHuman: फ्रांस की राजधानी क्या है?\n\nAssistant: फ्रांस की राजधानी Paris है। यह दुनिया के सबसे अधिक देखे जाने वाले शहरों में से एक है, जो Eiffel Tower और अपनी समृद्ध संस्कृति के लिए जाना जाता है।",
            "rejected_hi": "\n\nHuman: फ्रांस की राजधानी क्या है?\n\nAssistant: मुझे पूरी तरह से यकीन नहीं है, लेकिन मुझे लगता है कि यह Lyon हो सकता है।",
        },
        {
            "chosen": "\n\nHuman: How do I sort a list in Python?\n\nAssistant: You can use the built-in `sorted()` function or the `.sort()` method:\n\n```python\nmy_list = [3, 1, 2]\nmy_list.sort()  # in-place\nprint(sorted([3, 1, 2]))  # returns new list\n```",
            "rejected": "\n\nHuman: How do I sort a list in Python?\n\nAssistant: I would need more context to answer that. What kind of list do you have?",
            "chosen_hi": "\n\nHuman: Python में list को sort कैसे करें?\n\nAssistant: आप built-in `sorted()` function या `.sort()` method का उपयोग कर सकते हैं:\n\n```python\nmy_list = [3, 1, 2]\nmy_list.sort()  # in-place\nprint(sorted([3, 1, 2]))  # returns new list\n```",
            "rejected_hi": "\n\nHuman: Python में list को sort कैसे करें?\n\nAssistant: इसका उत्तर देने के लिए मुझे अधिक संदर्भ की आवश्यकता होगी। आपके पास किस प्रकार की list है?",
        },
    ],
    "malayalam": [
        {
            "chosen": "\n\nHuman: What is the capital of France?\n\nAssistant: The capital of France is Paris. It is one of the most visited cities in the world, known for the Eiffel Tower and its rich culture.",
            "rejected": "\n\nHuman: What is the capital of France?\n\nAssistant: I am not entirely sure, but I think it might be Lyon.",
            "chosen_ml": "\n\nHuman: ഫ്രാൻസിന്റെ തലസ്ഥാനം ഏതാണ്?\n\nAssistant: ഫ്രാൻസിന്റെ തലസ്ഥാനം Paris ആണ്. ലോകത്തിലെ ഏറ്റവും കൂടുതൽ സന്ദർശിക്കപ്പെടുന്ന നഗരങ്ങളിൽ ഒന്നാണിത്, Eiffel Tower ഉം സമ്പന്നമായ സംസ്കാരവും കൊണ്ട് പ്രശസ്തമാണ്.",
            "rejected_ml": "\n\nHuman: ഫ്രാൻസിന്റെ തലസ്ഥാനം ഏതാണ്?\n\nAssistant: എനിക്ക് പൂർണ്ണ ഉറപ്പില്ല, പക്ഷേ Lyon ആയിരിക്കുമെന്ന് തോന്നുന്നു.",
        },
        {
            "chosen": "\n\nHuman: How do I sort a list in Python?\n\nAssistant: You can use the built-in `sorted()` function or the `.sort()` method:\n\n```python\nmy_list = [3, 1, 2]\nmy_list.sort()  # in-place\nprint(sorted([3, 1, 2]))  # returns new list\n```",
            "rejected": "\n\nHuman: How do I sort a list in Python?\n\nAssistant: I would need more context to answer that. What kind of list do you have?",
            "chosen_ml": "\n\nHuman: Python-ൽ ഒരു list sort ചെയ്യുന്നതെങ്ങനെ?\n\nAssistant: നിങ്ങൾക്ക് built-in `sorted()` function അല്ലെങ്കിൽ `.sort()` method ഉപയോഗിക്കാം:\n\n```python\nmy_list = [3, 1, 2]\nmy_list.sort()  # in-place\nprint(sorted([3, 1, 2]))  # returns new list\n```",
            "rejected_ml": "\n\nHuman: Python-ൽ ഒരു list sort ചെയ്യുന്നതെങ്ങനെ?\n\nAssistant: ഉത്തരം നൽകാൻ കൂടുതൽ സന്ദർഭം ആവശ്യമാണ്. നിങ്ങൾക്ക് എന്ത് തരം list ആണ് ഉള്ളത്?",
        },
    ],
}

LANG_KEY = {"hindi": "hi", "malayalam": "ml"}
LANG_LABEL = {"hindi": "Hindi (Devanagari script)", "malayalam": "Malayalam (Malayalam script)"}


def build_system(lang):
    label = LANG_LABEL[lang]
    return (
        f"You are a professional translator. Translate English conversations to {label}.\n"
        "Rules:\n"
        "- Translate ONLY the text; do not answer or add commentary\n"
        "- Keep proper nouns (names, places, product names) in English\n"
        "- Keep code blocks, URLs, commands EXACTLY unchanged\n"
        "- Keep '\\n\\nHuman:' and '\\n\\nAssistant:' markers EXACTLY as-is\n"
        "- Translate naturally and fluently, not word-for-word\n"
        "- Output ONLY the two translated texts in the exact format shown"
    )


def build_prompt(chosen, rejected, lang):
    key = LANG_KEY[lang]
    shots = FEW_SHOT[lang]

    messages = [{"role": "system", "content": build_system(lang)}]

    # Few-shot turns
    for ex in shots:
        user_ex = (
            f"Translate these two conversation texts to {LANG_LABEL[lang]}:\n\n"
            f"CHOSEN:\n{ex['chosen']}\n\n"
            f"REJECTED:\n{ex['rejected']}"
        )
        asst_ex = (
            f"CHOSEN_TRANSLATED:\n{ex[f'chosen_{key}']}\n\n"
            f"REJECTED_TRANSLATED:\n{ex[f'rejected_{key}']}"
        )
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": asst_ex})

    # Actual request — truncate very long conversations to fit context
    chosen_t  = chosen[:1500]
    rejected_t = rejected[:1500]
    messages.append({"role": "user", "content": (
        f"Translate these two conversation texts to {LANG_LABEL[lang]}:\n\n"
        f"CHOSEN:\n{chosen_t}\n\n"
        f"REJECTED:\n{rejected_t}"
    )})
    return messages


def parse_response(text, chosen_orig, rejected_orig):
    """Extract CHOSEN_TRANSLATED and REJECTED_TRANSLATED from model output."""
    chosen_tr = rejected_tr = None
    if "CHOSEN_TRANSLATED:" in text and "REJECTED_TRANSLATED:" in text:
        parts = text.split("REJECTED_TRANSLATED:")
        chosen_part = parts[0].replace("CHOSEN_TRANSLATED:", "").strip()
        rejected_part = parts[1].strip() if len(parts) > 1 else ""
        chosen_tr  = chosen_part
        rejected_tr = rejected_part
    else:
        # Fallback: try splitting on the marker
        lines = text.strip().split("\n")
        half = len(lines) // 2
        chosen_tr  = "\n".join(lines[:half]).strip()
        rejected_tr = "\n".join(lines[half:]).strip()
    return chosen_tr or chosen_orig, rejected_tr or rejected_orig


def translate_one(client, model, ex, lang, max_tokens):
    msgs = build_prompt(ex["chosen"], ex["rejected"], lang)
    resp = client.chat.completions.create(
        model=model, messages=msgs,
        max_tokens=max_tokens, temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    chosen_tr, rejected_tr = parse_response(raw, ex["chosen"], ex["rejected"])
    return {"chosen": chosen_tr, "rejected": rejected_tr}


def translate_dataset(input_path, output_path, lang, base_url, model, batch, max_tokens):
    client = OpenAI(base_url=base_url, api_key="dummy")
    examples = [json.loads(l) for l in open(input_path) if l.strip()]
    results = [None] * len(examples)
    errors = 0

    print(f"\n── Translating {len(examples)} examples → {lang} ──")
    with tqdm(total=len(examples), unit="ex") as pbar:
        with ThreadPoolExecutor(max_workers=batch) as pool:
            futs = {
                pool.submit(translate_one, client, model, ex, lang, max_tokens): i
                for i, ex in enumerate(examples)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = examples[i]
                    errors += 1
                pbar.update(1)
                pbar.set_postfix(err=errors)

    print(f"  Done. errors={errors}/{len(examples)}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved: {output_path}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    required=True)
    ap.add_argument("--lang",     required=True, choices=list(LANG_KEY.keys()))
    ap.add_argument("--output",   required=True)
    ap.add_argument("--base-url", default="http://localhost:8003/v1")
    ap.add_argument("--model",    default="gemma3-27b")
    ap.add_argument("--batch",    type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    translate_dataset(args.input, args.output, args.lang,
                      args.base_url, args.model, args.batch, args.max_tokens)


if __name__ == "__main__":
    main()
