# src/export_merge_lora.py
import argparse
import os
import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = peft_model.merge_and_unload()

    merged.save_pretrained(args.out_dir, safe_serialization=True)
    proc = AutoProcessor.from_pretrained(args.base_model)
    proc.save_pretrained(args.out_dir)
    print("✅ Merged model saved to:", args.out_dir)

if __name__ == "__main__":
    main()
