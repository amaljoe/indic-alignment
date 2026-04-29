#!/usr/bin/env python3
"""Generate 6 bar graphs (2 per phase: before-only and before+after) → assets/"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

BLUE  = "#4C72B0"
GREEN = "#55A868"
BAR_W = 0.35

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")

# ── Phase 1: MILU Hindi per-domain ──────────────────────────────────────────
domains = [
    "Arts &\nHumanities", "Business\nStudies", "Engineering\n& Tech",
    "Environmental\nSciences", "Health &\nMedicine", "Law &\nGovernance",
    "Science", "Social\nSciences",
]
p1_before = [35.7, 41.7, 56.5, 37.9, 87.5, 57.1, 47.1, 48.3]
p1_after  = [50.0, 62.5, 71.7, 55.2, 87.5, 38.1, 62.7, 44.8]
x = np.arange(len(domains))

# before-only
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x, p1_before, color=BLUE, label="Before")
ax.set_xticks(x); ax.set_xticklabels(domains, fontsize=9)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Phase 1 — MILU Hindi: Baseline per Domain")
ax.axhline(sum(p1_before)/len(p1_before), color=BLUE, linestyle="--", alpha=0.6, label=f"Avg {sum(p1_before)/len(p1_before):.1f}%")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "phase1_before.png")

# before + after
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - BAR_W/2, p1_before, BAR_W, color=BLUE, label="Before")
ax.bar(x + BAR_W/2, p1_after,  BAR_W, color=GREEN, label="After")
ax.set_xticks(x); ax.set_xticklabels(domains, fontsize=9)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Phase 1 — MILU Hindi: Before vs After per Domain")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "phase1_before_after.png")

# ── Phase 2: Cultural metrics ────────────────────────────────────────────────
metrics   = ["NormAd\nAccuracy (%)", "BhED Stereo\nScore (%) ↓", "GlobalOp\nJS-sim (×100)"]
p2_before = [69.2, 44.1, 66.84]
p2_after  = [69.2, 27.9, 71.46]
x2 = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x2, p2_before, color=BLUE, label="Before")
ax.set_xticks(x2); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("Score"); ax.set_ylim(0, 100)
ax.set_title("Phase 2 — Cultural Alignment: Baseline")
ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="BhED random=50%")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "phase2_before.png")

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x2 - BAR_W/2, p2_before, BAR_W, color=BLUE, label="Before")
ax.bar(x2 + BAR_W/2, p2_after,  BAR_W, color=GREEN, label="After")
ax.set_xticks(x2); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("Score"); ax.set_ylim(0, 100)
ax.set_title("Phase 2 — Cultural Alignment: Before vs After")
ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="BhED random=50%")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "phase2_before_after.png")

# ── Phase 3: HHH 7 languages ─────────────────────────────────────────────────
langs     = ["Bengali", "English", "Hindi", "Malayalam", "Marathi", "Tamil", "Telugu"]
p3_before = [61.1, 91.0, 51.8, 56.2, 56.1, 62.7, 57.8]
p3_after  = [80.1, 86.9, 78.0, 78.2, 78.0, 75.9, 76.2]
x3 = np.arange(len(langs))

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x3, p3_before, color=BLUE, label="Before")
ax.set_xticks(x3); ax.set_xticklabels(langs, fontsize=10)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Phase 3 — HHH Safety Alignment: Baseline (7 Languages)")
avg_b = sum(p3_before)/len(p3_before)
ax.axhline(avg_b, color=BLUE, linestyle="--", alpha=0.6, label=f"Avg {avg_b:.1f}%")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "phase3_before.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x3 - BAR_W/2, p3_before, BAR_W, color=BLUE, label="Before")
ax.bar(x3 + BAR_W/2, p3_after,  BAR_W, color=GREEN, label="After")
ax.set_xticks(x3); ax.set_xticklabels(langs, fontsize=10)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Phase 3 — HHH Safety Alignment: Before vs After (7 Languages)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "phase3_before_after.png")

print("All 6 graphs saved to assets/")
