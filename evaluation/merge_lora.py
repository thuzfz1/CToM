import argparse
import glob
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM


def copy_tokenizer_files(base_model: str, output_dir: str):
    """Copy tokenizer files from base model to output dir.

    Avoids re-serializing via save_pretrained which can corrupt
    extra_special_tokens (list vs dict bug in some transformers versions).
    """
    # Resolve the base model path (download if needed)
    if os.path.isdir(base_model):
        base_dir = base_model
    else:
        base_dir = snapshot_download(base_model)

    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "tokenizer.model",
        # Qwen3 keeps its chat template in a standalone file (newer HF layout);
        # without it vLLM rejects every /v1/chat/completions request with 400
        # ("default chat template is no longer allowed"). added_tokens.json
        # likewise must travel with the tokenizer.
        "chat_template.jinja", "added_tokens.json",
    ]
    copied = 0
    for fname in tokenizer_files:
        src = os.path.join(base_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            copied += 1
    print(f"  Copied {copied} tokenizer files from {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output_dir", required=True, help="Where to save merged model")
    args = parser.parse_args()

    print(f"[1/4] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    print(f"[2/4] Loading LoRA adapter: {args.lora_adapter}")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter)

    print(f"[3/4] Merging LoRA weights...")
    merged = model.merge_and_unload()

    print(f"[4/4] Saving to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    copy_tokenizer_files(args.base_model, args.output_dir)

    print("Done.")

main()