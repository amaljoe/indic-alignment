# Qwen3-8B Evaluation Results

_Model: `Qwen/Qwen3-8B`  |  Date: 2026-05-01_

## Summary Table

| Metric | Value |
|--------|-------|
| MILU Hindi | 4840.00% (121/250) |
| MILU English | 5840.00% (146/250) |
| NormAd Accuracy | 71.01% (macro-F1: 0.606) |
| BhED Stereotype Score | 28.82% (lower = less biased, 50% = random) |
| GlobalOpinion JS-Sim | 0.7053 |
| HHH Avg Accuracy | 82.89% |

## Phase 1 — MILU (Knowledge, 0-shot)

### Hindi
- Accuracy: **4840.00%** (121/250)
- z-score: +8.544
- Errors/Overflow/Gibberish: 0/0/0

| Domain | Correct | Total | Acc |
|--------|---------|-------|-----|
| Arts & Humanities | 39 | 78 | 50.0% |
| Business Studies | 4 | 6 | 66.7% |
| Engineering & Tech | 21 | 41 | 51.2% |
| Environmental Sciences | 3 | 11 | 27.3% |
| Health & Medicine | 0 | 1 | 0.0% |
| Law & Governance | 1 | 1 | 100.0% |
| Science | 51 | 105 | 48.6% |
| Social Sciences | 2 | 7 | 28.6% |

### English
- Accuracy: **5840.00%** (146/250)
- z-score: +12.196
- Errors/Overflow/Gibberish: 0/0/5

| Domain | Correct | Total | Acc |
|--------|---------|-------|-----|
| Arts & Humanities | 13 | 20 | 65.0% |
| Business Studies | 6 | 14 | 42.9% |
| Engineering & Tech | 47 | 87 | 54.0% |
| Environmental Sciences | 7 | 7 | 100.0% |
| Health & Medicine | 3 | 3 | 100.0% |
| Law & Governance | 0 | 2 | 0.0% |
| Science | 67 | 113 | 59.3% |
| Social Sciences | 3 | 4 | 75.0% |

## Phase 2 — Cultural / Bias / Stereotype

### NormAd (Indic social norm judgment)
- Accuracy: **71.01%**
- Macro-F1: 0.606

| Country | Correct | Total | Acc |
|---------|---------|-------|-----|
| bangladesh | 26 | 33 | 78.8% |
| india | 21 | 29 | 72.4% |
| nepal | 25 | 37 | 67.6% |
| pakistan | 23 | 35 | 65.7% |
| sri_lanka | 25 | 35 | 71.4% |

### BhED (Caste + Religion stereotypes)
- Stereotype Score: **28.82%** (50% = random, lower = less biased)
- Clear predictions: 229/229

| Category | Stereo Score | N |
|----------|-------------|---|
| caste | 20.75% | 106 |
| religion | 35.77% | 123 |

### GlobalOpinion (India alignment)
- Avg JS-Similarity: **0.7053** (1.0 = perfect match)
- Evaluated on: 100 questions

## Phase 3 — HHH Alignment

- Average Accuracy: **82.89%**

| Language | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| bengali | 88.15% | 186 | 211 |
| english | 89.14% | 197 | 221 |
| hindi | 81.19% | 177 | 218 |
| malayalam | 75.91% | 167 | 220 |
| marathi | 83.64% | 179 | 214 |
| tamil | 82.08% | 174 | 212 |
| telugu | 80.10% | 165 | 206 |
