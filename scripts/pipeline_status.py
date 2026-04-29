#!/usr/bin/env python3
"""Pipeline status dashboard — shows what's done, running, and pending."""
import json, os, subprocess, sys, glob
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(ROOT, "logs", "pipeline_state.json")
STUDENT_URL = "http://localhost:8002/v1"
TEACHER_URL = "http://localhost:8003/v1"

STEPS = [
    "eval_baseline_phase1",
    "eval_baseline_phase2",
    "eval_baseline_phase3",
    "train_phase1",
    "datagen_phase2",
    "train_phase2",
    "datagen_phase3",
    "train_phase3",
]


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def check_server(url):
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/models", timeout=3) as r:
            d = json.loads(r.read())
            return [m["id"] for m in d["data"]]
    except Exception:
        return None


def gpu_summary():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        )
        lines = []
        for row in out.strip().splitlines():
            idx, name, util, used, total, temp = [x.strip() for x in row.split(",")]
            pct = int(used) * 100 // max(int(total), 1)
            lines.append(f"  GPU{idx}  {util:>3}% util  {used}/{total} MiB ({pct}%)  {temp}°C  {name}")
        return "\n".join(lines)
    except Exception as e:
        return f"  (nvidia-smi unavailable: {e})"


def tail_log(path, n=6):
    if not os.path.exists(path):
        return "  (no log)"
    try:
        lines = open(path).readlines()
        return "".join("  " + l for l in lines[-n:]).rstrip()
    except Exception:
        return "  (unreadable)"


def results_summary():
    path = os.path.join(ROOT, "final", "results.md")
    if not os.path.exists(path):
        return "  (not yet written)"
    with open(path) as f:
        return f.read().strip()


def main():
    state = load_state()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*64}")
    print(f"  Indic-Alignment Pipeline Status  [{now}]")
    print(f"{'='*64}")

    # Servers
    student = check_server(STUDENT_URL)
    teacher = check_server(TEACHER_URL)
    print(f"\n  Student vLLM ({STUDENT_URL}): ", end="")
    print(f"UP — {student}" if student else "OFFLINE")
    print(f"  Teacher vLLM ({TEACHER_URL}): ", end="")
    print(f"UP — {teacher}" if teacher else "OFFLINE")

    # Steps
    print(f"\n  Pipeline steps:")
    for step in STEPS:
        info = state.get(step, {})
        status = info.get("status", "pending")
        ts = info.get("ts", "")
        icon = {"done": "✓", "running": "▶", "failed": "✗", "pending": "·"}.get(status, "?")
        extra = f"  ({ts})" if ts else ""
        msg = info.get("msg", "")
        print(f"    [{icon}] {step:<28} {status}{extra}  {msg}")

    # GPU
    print(f"\n  GPU status:")
    print(gpu_summary())

    # Recent logs
    log_files = {
        "phase1_eval":  os.path.join(ROOT, "logs", "phase1_eval.log"),
        "phase1_train": os.path.join(ROOT, "logs", "phase1_train.log"),
        "phase2_eval":  os.path.join(ROOT, "logs", "phase2_eval.log"),
        "phase2_train": os.path.join(ROOT, "logs", "phase2_train.log"),
        "phase3_eval":  os.path.join(ROOT, "logs", "phase3_eval.log"),
        "phase3_train": os.path.join(ROOT, "logs", "phase3_train.log"),
    }
    for label, path in log_files.items():
        if os.path.exists(path):
            print(f"\n  -- {label} (last 4 lines) --")
            print(tail_log(path, 4))

    # Results
    print(f"\n  final/results.md:")
    print(results_summary())

    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
