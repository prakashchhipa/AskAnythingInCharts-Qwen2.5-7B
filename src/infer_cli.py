# src/infer_cli.py
import argparse
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter", default="outputs/qwen2_5_vl_7b_sft_chart_text")
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None
    ).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    processor = AutoProcessor.from_pretrained(args.base_model)

    image = Image.open(args.image).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.question},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out_text)

if __name__ == "__main__":
    main()
