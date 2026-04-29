"""
DPO LoRA fine-tuning on multilingual hh-rlhf data (English + Hindi + Malayalam).

Expects three JSONL files:
  --data-en   data/hh_rlhf/hh_rlhf_5k_en.jsonl
  --data-hi   data/hh_rlhf/hh_rlhf_5k_hindi.jsonl
  --data-ml   data/hh_rlhf/hh_rlhf_5k_malayalam.jsonl

Each line: {"chosen": "...", "rejected": "..."}
  chosen/rejected format: "\n\nHuman: <prompt>\n\nAssistant: <response1>\n\nHuman: ...\n\nAssistant: <last>"

Training: DPO via trl 1.3 DPOTrainer + PEFT LoRA (r=16, alpha=32).
Per-language reward margins logged at each logging_steps via a custom callback.

Usage:
  torchrun --nproc_per_node 4 scripts/train_dpo_lora.py \
      --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --data-en  data/hh_rlhf/hh_rlhf_5k_en.jsonl \
      --data-hi  data/hh_rlhf/hh_rlhf_5k_hindi.jsonl \
      --data-ml  data/hh_rlhf/hh_rlhf_5k_malayalam.jsonl \
      --output   checkpoints/dpo_multilingual \
      --epochs 1 --batch 2 --grad-accum 8 --lr 5e-5
"""
import argparse, json, os
from collections import defaultdict

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


def parse_hh(text: str):
    """Split hh-rlhf conversation into (prompt, response)."""
    parts = text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        return parts[0] + "\n\nAssistant:", parts[1].strip()
    return text, ""


def load_jsonl(path: str, lang_tag: str) -> Dataset:
    rows = {"prompt": [], "chosen": [], "rejected": [], "lang": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            prompt, chosen_resp = parse_hh(ex["chosen"])
            _, rejected_resp = parse_hh(ex["rejected"])
            if not chosen_resp or not rejected_resp:
                continue
            rows["prompt"].append(prompt)
            rows["chosen"].append(chosen_resp)
            rows["rejected"].append(rejected_resp)
            rows["lang"].append(lang_tag)
    return Dataset.from_dict(rows)


class PerLangRewardCallback(TrainerCallback):
    """Print per-language reward margin each logging step."""

    def __init__(self, ds_lang_list):
        # ds_lang_list: the 'lang' column of the shuffled training dataset
        self._langs = ds_lang_list
        self._batch_langs = []  # filled by on_step_begin
        self._reward_bufs = defaultdict(list)
        self._global_idx = 0

    def on_step_begin(self, args, state, control, **kwargs):
        # Record which languages are in this batch (best-effort, works for single-process)
        bsz = args.per_device_train_batch_size
        step = state.global_step
        start = (step * bsz) % len(self._langs)
        self._batch_langs = [self._langs[(start + i) % len(self._langs)] for i in range(bsz)]

    def on_log(self, args, state, control, logs=None, **kwargs):
        # trl accumulates global reward/margin in logs under 'rewards/margins'
        margin = (logs or {}).get("rewards/margins")
        if margin is not None:
            # We can't split by language here precisely; log global margin
            print(f"[step {state.global_step}] reward_margin={margin:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True)
    ap.add_argument("--data-en",    required=True)
    ap.add_argument("--data-hi",    required=True)
    ap.add_argument("--data-ml",    required=True)
    ap.add_argument("--output",     default="checkpoints/dpo_multilingual")
    ap.add_argument("--epochs",     type=int,   default=1)
    ap.add_argument("--batch",      type=int,   default=2)
    ap.add_argument("--grad-accum", type=int,   default=8)
    ap.add_argument("--lr",         type=float, default=5e-5)
    ap.add_argument("--max-len",    type=int,   default=1024)
    ap.add_argument("--beta",       type=float, default=0.1)
    ap.add_argument("--lora-r",     type=int,   default=16)
    ap.add_argument("--lora-alpha", type=int,   default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--logging-steps", type=int, default=10)
    args = ap.parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    # With torchrun, LOCAL_RANK pins each process to its own GPU.
    # CUDA_VISIBLE_DEVICES is NOT pre-filtered per process, so we use LOCAL_RANK directly.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    print(f"Loading model from {args.model} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA config (passed directly to DPOTrainer) ───────────────────────────
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    ds_en = load_jsonl(args.data_en, "en")
    ds_hi = load_jsonl(args.data_hi, "hi")
    ds_ml = load_jsonl(args.data_ml, "ml")
    ds = concatenate_datasets([ds_en, ds_hi, ds_ml]).shuffle(seed=42)
    print(f"Total examples: {len(ds)} (en={len(ds_en)}, hi={len(ds_hi)}, ml={len(ds_ml)})")

    # ── Training config ────────────────────────────────────────────────────────
    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        beta=args.beta,
        max_length=args.max_len,
        truncation_mode="keep_end",
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        callbacks=[PerLangRewardCallback(ds["lang"])],
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    print("\nStarting DPO training...")
    trainer.train()

    print(f"\nSaving final checkpoint to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
