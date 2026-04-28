"""
KL-distillation SFT for Indic alignment.

Loss = alpha * CE(student, teacher_response) + (1-alpha) * T² * KL(teacher_top256 || student)

The sparse KL term uses teacher's stored top-256 log-probs as the target distribution.
Student log-probs at those K positions are gathered from the full vocab distribution.

Launch (2 GPUs):
    accelerate launch --config_file finetune/accelerate_config.yaml \\
        finetune/train_kl.py

Single GPU:
    python finetune/train_kl.py
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ── Dataset ────────────────────────────────────────────────────────────────────

class KLDistillationDataset(TorchDataset):
    """
    Tokenises each record on-the-fly.

    Each record must have:
        messages        list of {role, content}   (system + user + assistant)
        teacher_logprobs list of list of {token_id, log_prob}   (per resp token × top-K)

    Returns a dict with:
        input_ids   [L]
        labels      [L]   (-100 for prompt positions, token_id for response positions)
        resp_start  int   (index of first response token in input_ids)
        teacher_ids   [R, K]  (long)
        teacher_lps   [R, K]  (float)
    """

    def __init__(self, records, tokenizer, max_length=1024):
        self.records   = records
        self.tok       = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        messages        = rec["messages"]
        teacher_logprobs = rec["teacher_logprobs"]

        # Full chat-formatted text
        full_text   = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Prompt only (without assistant turn)
        prompt_text = self.tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        full_ids   = self.tok.encode(full_text,   add_special_tokens=False)
        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
        resp_start = len(prompt_ids)

        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]

        labels = [-100] * len(full_ids)
        for i in range(resp_start, len(full_ids)):
            labels[i] = full_ids[i]

        # Build teacher tensors [R, K]
        R = len(teacher_logprobs)
        K = len(teacher_logprobs[0]) if R > 0 else 0

        t_ids = torch.zeros(R, K, dtype=torch.long)
        t_lps = torch.full((R, K), fill_value=-1e9, dtype=torch.float32)
        for r, top_k in enumerate(teacher_logprobs):
            for k, entry in enumerate(top_k):
                t_ids[r, k] = entry["token_id"]
                t_lps[r, k] = entry["log_prob"]

        return {
            "input_ids":    torch.tensor(full_ids,  dtype=torch.long),
            "labels":       torch.tensor(labels,    dtype=torch.long),
            "resp_start":   resp_start,
            "teacher_ids":  t_ids,
            "teacher_lps":  t_lps,
        }


# ── Collator ───────────────────────────────────────────────────────────────────

@dataclass
class KLDataCollator:
    tokenizer:  Any
    pad_token_id: int = 0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad input_ids / labels to max sequence length in batch
        max_seq = max(f["input_ids"].shape[0] for f in features)

        input_ids_batch = []
        labels_batch    = []
        attn_mask_batch = []
        resp_starts     = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_seq - seq_len
            input_ids_batch.append(
                F.pad(f["input_ids"], (0, pad_len), value=self.tokenizer.pad_token_id)
            )
            labels_batch.append(
                F.pad(f["labels"], (0, pad_len), value=-100)
            )
            attn_mask_batch.append(
                torch.cat([torch.ones(seq_len, dtype=torch.long),
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            resp_starts.append(f["resp_start"])

        # Pad teacher tensors to (max_R, K)
        max_R = max(f["teacher_ids"].shape[0] for f in features)
        K     = features[0]["teacher_ids"].shape[1] if features[0]["teacher_ids"].shape[0] > 0 else 256

        teacher_ids_batch = []
        teacher_lps_batch = []
        resp_mask_batch   = []   # [B, max_R] — 1 for real tokens

        for f in features:
            R   = f["teacher_ids"].shape[0]
            pad = max_R - R
            teacher_ids_batch.append(
                F.pad(f["teacher_ids"], (0, 0, 0, pad), value=0)
            )
            teacher_lps_batch.append(
                F.pad(f["teacher_lps"], (0, 0, 0, pad), value=-1e9)
            )
            resp_mask_batch.append(
                torch.cat([torch.ones(R, dtype=torch.bool),
                           torch.zeros(pad, dtype=torch.bool)])
            )

        return {
            "input_ids":       torch.stack(input_ids_batch),
            "labels":          torch.stack(labels_batch),
            "attention_mask":  torch.stack(attn_mask_batch),
            "resp_starts":     torch.tensor(resp_starts, dtype=torch.long),
            "teacher_ids":     torch.stack(teacher_ids_batch),    # [B, max_R, K]
            "teacher_lps":     torch.stack(teacher_lps_batch),    # [B, max_R, K]
            "resp_mask":       torch.stack(resp_mask_batch),       # [B, max_R]
        }


# ── Trainer ────────────────────────────────────────────────────────────────────

class KLDistillationTrainer(Trainer):
    """
    Custom trainer adding a sparse KL distillation term to the CE loss.

    Loss = alpha * CE + (1 - alpha) * T² * mean_KL

    where mean_KL is averaged over all non-padded response token positions.
    """

    def __init__(self, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha       = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop our custom fields before forwarding to model
        teacher_ids  = inputs.pop("teacher_ids",  None)   # [B, max_R, K]
        teacher_lps  = inputs.pop("teacher_lps",  None)   # [B, max_R, K]
        resp_starts  = inputs.pop("resp_starts",  None)   # [B]
        resp_mask    = inputs.pop("resp_mask",    None)   # [B, max_R]

        outputs = model(**inputs)
        logits  = outputs.logits   # [B, L, V]
        labels  = inputs["labels"] # [B, L]

        V = logits.shape[-1]

        # ── CE loss (standard next-token on response tokens) ──────────────────
        shift_logits = logits[:, :-1, :].contiguous()    # [B, L-1, V]
        shift_labels = labels[:, 1:].contiguous()        # [B, L-1]
        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        if teacher_ids is None or self.alpha >= 1.0:
            loss = ce_loss
            return (loss, outputs) if return_outputs else loss

        # ── Sparse KL loss ────────────────────────────────────────────────────
        B, max_R, K = teacher_ids.shape
        T = self.temperature

        # Student log-probs over full vocab at response positions [B, max_R, V]
        # resp_starts[b] = first response position in the label sequence
        # shift_logits[b, resp_starts[b] - 1 + r, :] is the dist predicting response token r
        # (shift already removes one position from the left)
        student_resp_logits = torch.zeros(B, max_R, V, dtype=logits.dtype, device=logits.device)
        for b in range(B):
            rs = int(resp_starts[b].item()) - 1   # offset into shift_logits
            n  = min(max_R, shift_logits.shape[1] - rs)
            if rs >= 0 and n > 0:
                student_resp_logits[b, :n, :] = shift_logits[b, rs:rs + n, :]

        # Apply temperature and compute log-softmax
        student_log_probs = F.log_softmax(student_resp_logits / T, dim=-1)  # [B, max_R, V]

        # Gather student log-probs at teacher's top-K positions
        # teacher_ids: [B, max_R, K]
        student_at_k = student_log_probs.gather(-1, teacher_ids)  # [B, max_R, K]

        # Re-temperature teacher: stored as log_softmax(logits), T=1
        # At temperature T: log_p_T[i] = log_p[i]/T - log(sum_j exp(log_p[j]/T))
        # We approximate the normaliser over top-K only (good enough when K=256)
        teacher_lps_T = teacher_lps / T
        teacher_lps_T = teacher_lps_T - torch.logsumexp(teacher_lps_T, dim=-1, keepdim=True)  # [B, max_R, K]

        # KL(teacher || student) = sum_k teacher_k * (log_teacher_k - log_student_k)
        teacher_probs_T = teacher_lps_T.exp()                     # [B, max_R, K]
        kl_per_pos = (teacher_probs_T * (teacher_lps_T - student_at_k)).sum(-1)  # [B, max_R]

        # Mask out padded positions
        resp_mask_float = resp_mask.float()
        n_valid = resp_mask_float.sum().clamp(min=1.0)
        kl_loss = (kl_per_pos * resp_mask_float).sum() / n_valid

        # Clamp to avoid instability from very divergent distributions
        kl_loss = kl_loss.clamp(max=20.0)

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * (T ** 2) * kl_loss
        return (loss, outputs) if return_outputs else loss


# ── Data loading ───────────────────────────────────────────────────────────────

def load_soft_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_dataset(data_dir, tokenizer, max_length, sources_weights):
    """Load and oversample soft-label JSONL files."""
    all_records = []
    for fname, weight in sources_weights:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        recs = load_soft_jsonl(path)
        n = max(1, round(weight))
        all_records.extend(recs * n)
        print(f"  {fname}: {len(recs)} × {n} = {len(recs) * n}")

    import random
    random.seed(42)
    random.shuffle(all_records)

    n_val = max(1, int(len(all_records) * 0.05))
    val_recs   = all_records[:n_val]
    train_recs = all_records[n_val:]
    print(f"  Total train: {len(train_recs)}  val: {len(val_recs)}")

    train_ds = KLDistillationDataset(train_recs, tokenizer, max_length)
    val_ds   = KLDistillationDataset(val_recs,   tokenizer, max_length)
    return train_ds, val_ds


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data-dir",     default="finetune/data")
    parser.add_argument("--output",       default="finetune/checkpoints_kl")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--batch-size",   type=int,   default=2)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--max-seq-len",  type=int,   default=1024)
    parser.add_argument("--save-steps",   type=int,   default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha",        type=float, default=0.5,
                        help="CE weight; KL weight = 1 - alpha")
    parser.add_argument("--temperature",  type=float, default=2.0,
                        help="Distillation temperature")
    args = parser.parse_args()

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # ── Data ───────────────────────────────────────────────────────────────────
    SOURCES_WEIGHTS = [
        ("normad_soft.jsonl",       3.0),
        ("milu_en_soft.jsonl",      1.0),
        ("milu_hi_soft.jsonl",      1.0),
        ("bhed_soft.jsonl",         2.0),
        ("globalopinion_soft.jsonl",1.0),
    ]
    train_ds, val_ds = build_dataset(
        args.data_dir, tokenizer, args.max_seq_len, SOURCES_WEIGHTS
    )

    collator = KLDataCollator(tokenizer=tokenizer)

    # ── Training args ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
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
        remove_unused_columns=False,   # essential: keep our custom fields
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = KLDistillationTrainer(
        alpha=args.alpha,
        temperature=args.temperature,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"\nalpha={args.alpha}  T={args.temperature}")
    print(f"Effective batch: {args.batch_size * args.grad_accum} × n_gpu")
    trainer.train()

    final_dir = os.path.join(args.output, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to: {final_dir}")
    print(f"To serve: bash scripts/start_vllm.sh {final_dir} 8002 2")


if __name__ == "__main__":
    main()
