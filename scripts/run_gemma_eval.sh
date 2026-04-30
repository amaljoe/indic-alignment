#!/bin/bash
# Run Phase 1+2+3 evals for Qwen/Qwen3-8B on port 8003.
# Waits for the server to be ready before starting.
set -e

BASE_URL="http://localhost:8003/v1"
MODEL="qwen3-8b"
OUTDIR="/home/compiling-ganesh/24m0797/workspace/indic-alignment/results"
SCRIPTS="/home/compiling-ganesh/24m0797/workspace/indic-alignment/scripts"
PYTHON="/dev/shm/vllm/bin/python"
RESULT_MD="/home/compiling-ganesh/24m0797/workspace/indic-alignment/qwen3.md"

echo "=== Waiting for vLLM server at $BASE_URL ==="
until curl -sf "$BASE_URL/models" > /dev/null 2>&1; do
  echo "  ... server not ready, sleeping 15s"
  sleep 15
done
echo "Server is up!"
curl -s "$BASE_URL/models" | python3 -m json.tool

echo ""
echo "=== Phase 1: MILU eval ==="
cd "$SCRIPTS"
$PYTHON phase1_eval.py \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --tag qwen3-baseline \
  --output "$OUTDIR/qwen3_phase1.json" \
  2>&1 | tee /tmp/qwen3_phase1.log
echo "Phase 1 done."

echo ""
echo "=== Phase 2: Cultural/Bias eval ==="
$PYTHON phase2_eval.py \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --tag qwen3-baseline \
  --output "$OUTDIR/qwen3_phase2.json" \
  2>&1 | tee /tmp/qwen3_phase2.log
echo "Phase 2 done."

echo ""
echo "=== Phase 3: HHH eval ==="
$PYTHON phase3_eval.py \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --tag qwen3-baseline \
  --output "$OUTDIR/qwen3_phase3.json" \
  2>&1 | tee /tmp/qwen3_phase3.log
echo "Phase 3 done."

echo ""
echo "=== All evals complete. Generating qwen3.md ==="
$PYTHON - <<'PYEOF'
import json, datetime

out_dir = "/home/compiling-ganesh/24m0797/workspace/indic-alignment/results"
md_path = "/home/compiling-ganesh/24m0797/workspace/indic-alignment/qwen3.md"

def load(fname):
    with open(f"{out_dir}/{fname}") as f:
        return json.load(f)

p1 = load("qwen3_phase1.json")
p2 = load("qwen3_phase2.json")
p3 = load("qwen3_phase3.json")

hi = p1["hindi"]
en = p1["english"]
normad = p2["normad"]
bhed = p2["bhed"]
globalop = p2["globalopinion"]
hhh_by_lang = p3["by_language"]
hhh_avg = p3["avg_accuracy"] * 100

lines = []
lines.append("# Qwen3-8B Evaluation Results")
lines.append(f"\n_Model: `Qwen/Qwen3-8B`  |  Date: {datetime.date.today()}_\n")

lines.append("## Summary Table\n")
lines.append("| Metric | Value |")
lines.append("|--------|-------|")
lines.append(f"| MILU Hindi | {hi['accuracy']:.2f}% ({hi['correct']}/{hi['total']}) |")
lines.append(f"| MILU English | {en['accuracy']:.2f}% ({en['correct']}/{en['total']}) |")
lines.append(f"| NormAd Accuracy | {normad['accuracy']*100:.2f}% (macro-F1: {normad['macro_f1']:.3f}) |")
lines.append(f"| BhED Stereotype Score | {bhed['stereotype_score']:.2f}% (lower = less biased, 50% = random) |")
lines.append(f"| GlobalOpinion JS-Sim | {globalop['avg_js_similarity']:.4f} |")
lines.append(f"| HHH Avg Accuracy | {hhh_avg:.2f}% |")

lines.append("\n## Phase 1 — MILU (Knowledge, 0-shot)\n")
lines.append("### Hindi")
lines.append(f"- Accuracy: **{hi['accuracy']:.2f}%** ({hi['correct']}/{hi['total']})")
lines.append(f"- z-score: {hi['z_score']:+.3f}")
lines.append(f"- Errors/Overflow/Gibberish: {hi['errors']}/{hi['overflow']}/{hi['gibberish']}")
if hi.get("per_domain"):
    lines.append("\n| Domain | Correct | Total | Acc |")
    lines.append("|--------|---------|-------|-----|")
    for d, v in sorted(hi["per_domain"].items()):
        lines.append(f"| {d} | {v['correct']} | {v['total']} | {v['accuracy']:.1f}% |")

lines.append("\n### English")
lines.append(f"- Accuracy: **{en['accuracy']:.2f}%** ({en['correct']}/{en['total']})")
lines.append(f"- z-score: {en['z_score']:+.3f}")
lines.append(f"- Errors/Overflow/Gibberish: {en['errors']}/{en['overflow']}/{en['gibberish']}")
if en.get("per_domain"):
    lines.append("\n| Domain | Correct | Total | Acc |")
    lines.append("|--------|---------|-------|-----|")
    for d, v in sorted(en["per_domain"].items()):
        lines.append(f"| {d} | {v['correct']} | {v['total']} | {v['accuracy']:.1f}% |")

lines.append("\n## Phase 2 — Cultural / Bias / Stereotype\n")
lines.append("### NormAd (Indic social norm judgment)")
lines.append(f"- Accuracy: **{normad['accuracy']*100:.2f}%**")
lines.append(f"- Macro-F1: {normad['macro_f1']:.3f}")
if normad.get("per_country"):
    lines.append("\n| Country | Correct | Total | Acc |")
    lines.append("|---------|---------|-------|-----|")
    for c, v in sorted(normad["per_country"].items()):
        lines.append(f"| {c} | {v['c']} | {v['t']} | {v['acc']:.1f}% |")

lines.append("\n### BhED (Caste + Religion stereotypes)")
lines.append(f"- Stereotype Score: **{bhed['stereotype_score']:.2f}%** (50% = random, lower = less biased)")
lines.append(f"- Clear predictions: {bhed['clear']}/{bhed['total']}")
if bhed.get("by_category"):
    lines.append("\n| Category | Stereo Score | N |")
    lines.append("|----------|-------------|---|")
    for cat, v in bhed["by_category"].items():
        lines.append(f"| {cat} | {v['stereo_score']:.2f}% | {v['n']} |")

lines.append("\n### GlobalOpinion (India alignment)")
lines.append(f"- Avg JS-Similarity: **{globalop['avg_js_similarity']:.4f}** (1.0 = perfect match)")
lines.append(f"- Evaluated on: {globalop['total']} questions")

lines.append("\n## Phase 3 — HHH Alignment\n")
lines.append(f"- Average Accuracy: **{hhh_avg:.2f}%**\n")
lines.append("| Language | Accuracy | Correct | Total |")
lines.append("|----------|----------|---------|-------|")
for lang, r in sorted(hhh_by_lang.items()):
    lines.append(f"| {lang} | {r['accuracy']*100:.2f}% | {r['correct']} | {r['total']} |")

with open(md_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Written: {md_path}")
PYEOF

echo "=== Done: qwen3.md written ==="
