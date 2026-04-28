"""
SFT training — context distillation for Indic alignment.

Full-parameter finetuning of DeepSeek-R1-Distill-Qwen-1.5B on distilled
Indic cultural data. No LoRA needed at 1.5B — fits in ~14 GB BF16.

Launch:
    accelerate launch --num_processes 2 --mixed_precision bf16 finetune/train.py
Or single GPU:
    python finetune/train.py
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data",         default="finetune/data/train.jsonl")
    parser.add_argument("--output",       default="finetune/checkpoints")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--grad-accum",   type=int,   default=4)
    parser.add_argument("--max-seq-len",  type=int,   default=1024)
    parser.add_argument("--save-steps",   type=int,   default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ── Load data ─────────────────────────────────────────────────────────────
    raw = load_jsonl(args.data)
    print(f"Loaded {len(raw)} training examples from {args.data}")

    # Split 95/5 train/val
    n_val = max(1, int(len(raw) * 0.05))
    val_data   = Dataset.from_list(raw[:n_val])
    train_data = Dataset.from_list(raw[n_val:])
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    # ── Training config ───────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=2,
        # SFT-specific
        max_length=args.max_seq_len,
        packing=True,           # pack short sequences for efficiency
        dataset_text_field="text",  # will be overridden by messages format
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training: {len(train_data)} examples × {args.epochs} epochs")
    print(f"Effective batch size: {args.batch_size * args.grad_accum} × num_gpus")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to: {final_dir}")
    print(f"To serve: bash scripts/start_vllm.sh {final_dir} 8002 2")


if __name__ == "__main__":
    main()
