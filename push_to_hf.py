#!/usr/bin/env python3
"""
push_to_hf.py — Merge LoRA adapter into base model and push to HuggingFace Hub.

Usage:
  python push_to_hf.py --adapter checkpoints/overfit_lora --repo amaljoe88/deepseek-r1-8b-indic
"""
import argparse, os, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/overfit_lora")
    ap.add_argument("--repo", default="amaljoe88/deepseek-r1-8b-indic-aligned")
    ap.add_argument("--base-model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--push-adapter-only", action="store_true",
                    help="Push just the PEFT adapter (smaller, references base model)")
    args = ap.parse_args()

    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:3128")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:3128")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if args.push_adapter_only:
        print(f"Pushing adapter only to {args.repo}...")
        from peft import PeftModel, PeftConfig
        # Update adapter_config to reference the base model
        cfg_path = os.path.join(args.adapter, "adapter_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["base_model_name_or_path"] = args.base_model
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.adapter)
        model.push_to_hub(args.repo, private=False)
        tokenizer.push_to_hub(args.repo, private=False)
    else:
        print(f"Loading and merging adapter: {args.adapter}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.adapter)
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        print(f"Pushing merged model to {args.repo}...")
        model.push_to_hub(args.repo, private=False, safe_serialization=True)
        tokenizer.push_to_hub(args.repo, private=False)

    print(f"\nDone! Model at: https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
