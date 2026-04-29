#!/usr/bin/env python3
"""
State manager for the agent-based pipeline.

Usage:
  python pipeline_state.py get <step>
  python pipeline_state.py set <step> <status> [msg]
  python pipeline_state.py list
"""
import json, os, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(ROOT, "logs", "pipeline_state.json")

VALID_STEPS = [
    "eval_baseline_phase1",
    "eval_baseline_phase2",
    "eval_baseline_phase3",
    "train_phase1",
    "datagen_phase2",
    "train_phase2",
    "datagen_phase3",
    "train_phase3",
]
VALID_STATUSES = {"pending", "running", "done", "failed"}


def load():
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_status(step):
    return load().get(step, {}).get("status", "pending")


def set_status(step, status, msg=""):
    assert step in VALID_STEPS, f"Unknown step: {step}"
    assert status in VALID_STATUSES, f"Unknown status: {status}"
    state = load()
    state[step] = {"status": status, "ts": datetime.now().strftime("%H:%M:%S"), "msg": msg}
    save(state)
    print(f"[state] {step} → {status}  {msg}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "get":
        print(get_status(sys.argv[2]))
    elif cmd == "set":
        msg = sys.argv[4] if len(sys.argv) > 4 else ""
        set_status(sys.argv[2], sys.argv[3], msg)
    elif cmd == "list":
        state = load()
        for s in VALID_STEPS:
            info = state.get(s, {})
            print(f"{s:30} {info.get('status','pending'):8} {info.get('msg','')}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
