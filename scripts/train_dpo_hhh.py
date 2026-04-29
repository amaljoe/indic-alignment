"""
DPO LoRA fine-tuning on HHH preference pairs (EN + HI + ML).

Input JSONL: {"prompt": "...", "chosen": "...", "rejected": "...", "lang": "..."}
(produced by scripts/prepare_hhh_dpo.py)

Usage:
  python3.10 -m torch.distributed.run --nproc_per_node 4 \
      scripts/train_dpo_hhh.py \
      --data  data/hhh_dpo_train.jsonl \
      --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --output checkpoints/dpo_hhh \
      --epochs 3 --batch 2 --grad-accum 4 --lr 5e-5
"""
import argparse, json, os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


class RewardMarginCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        margin = (logs or {}).get("rewards/margins")
        if margin is not None:
            print(f"[step {state.global_step}] reward_margin={margin:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--data",        default="data/hhh_dpo_train.jsonl")
    ap.add_argument("--output",      default="checkpoints/dpo_hhh")
    ap.add_argument("--epochs",      type=int,   default=3)
    ap.add_argument("--batch",       type=int,   default=2)
    ap.add_argument("--grad-accum",  type=int,   default=4)
    ap.add_argument("--lr",          type=float, default=5e-5)
    ap.add_argument("--max-len",     type=int,   default=512)
    ap.add_argument("--beta",        type=float, default=0.1)
    ap.add_argument("--lora-r",      type=int,   default=16)
    ap.add_argument("--lora-alpha",  type=int,   default=32)
    ap.add_argument("--lora-dropout",type=float, default=0.05)
    ap.add_argument("--logging-steps",type=int,  default=5)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    print(f"Loading model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    rows = {"prompt": [], "chosen": [], "rejected": []}
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            rows["prompt"].append(ex["prompt"])
            rows["chosen"].append(ex["chosen"])
            rows["rejected"].append(ex["rejected"])
    ds = Dataset.from_dict(rows).shuffle(seed=42)
    print(f"Training on {len(ds)} examples, {args.epochs} epochs")

    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        beta=args.beta,
        max_length=args.max_len,
        truncation_mode="keep_end",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        callbacks=[RewardMarginCallback()],
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
