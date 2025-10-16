# evaluations/eval_chartqa.py
import argparse
import json
from pathlib import Path
from datasets import load_dataset
import gc
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

def build_model(base_model, adapter=None, device="cuda"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)

    if adapter:
        # Use PeftModel.from_pretrained() which handles key transformations correctly
        print(f"Loading LoRA adapter from: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        print("Merging adapter weights into base model...")
        model = model.merge_and_unload()
        print("Adapter loaded and merged successfully!")

    # Use adapter's processor if provided
    processor_src = adapter if adapter else base_model
    processor = AutoProcessor.from_pretrained(processor_src)
    return model, processor

def infer(model, processor, image, question, device, max_new_tokens=64):
    # Use the SAME format as training for consistency
    SYSTEM_MSG = (
        "You are a helpful Vision-Language assistant. "
        "Be concise and accurate. If the image contains small text, read it carefully. "
        "When answering chart questions, give the shortest correct answer."
    )
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image"}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Update messages for vision processing (add actual image)
    messages_with_image = [
        {"role": "system", "content": messages[0]["content"]},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image", "image": image}
        ]}
    ]
    image_inputs, _ = process_vision_info(messages_with_image)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return out.strip()


def _to_number(s: str):
    x = s.strip().lower().replace(",", "")
    # handle k/m suffixes
    mul = 1.0
    if x.endswith("k"):
        mul = 1e3
        x = x[:-1]
    elif x.endswith("m"):
        mul = 1e6
        x = x[:-1]
    try:
        return float(x) * mul
    except Exception:
        return None


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    # remove simple punctuation and spaces
    for ch in [",", "%", "$", ":", ";", ".", " "]:
        s = s.replace(ch, "")
    return s

def evaluate(model, processor, ds, device, limit=500, max_new_tokens=64):
    n, correct = 0, 0
    for sample in ds.select(range(min(limit, len(ds)))):
        img = sample["image"]
        q = sample["query"]
        gt = sample["label"][0] if isinstance(sample["label"], list) else sample["label"]
        pred = infer(model, processor, img, q, device, max_new_tokens=max_new_tokens)
        # EM with light normalization
        p_norm = normalize_answer(str(pred))
        g_norm = normalize_answer(str(gt))
        ok = p_norm == g_norm
        if not ok:
            # try numeric match with small tolerance
            pn = _to_number(str(pred))
            gn = _to_number(str(gt))
            if pn is not None and gn is not None:
                if abs(pn - gn) <= max(0.5, 0.01 * abs(gn)):
                    ok = True
        if ok:
            correct += 1
        n += 1
        if n % 50 == 0:
            print(f"{n} done; running acc={correct/n:.3f}")
    return correct, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen2_5_vl_7b_sft_chart_text")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--compare_both", action="store_true", help="Run base-only and with adapter in one go")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("HuggingFaceM4/ChartQA", trust_remote_code=True)["val"]

    if args.compare_both:
        # Base-only
        print("=== Evaluating BASE (no adapter) ===")
        model, processor = build_model(args.base_model, adapter=None, device=device)
        correct, n = evaluate(model, processor, ds, device, limit=args.limit, max_new_tokens=args.max_new_tokens)
        print(f"BASE ChartQA acc on {n} samples: {correct/n:.3f}")
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # With adapter (if provided)
        if args.adapter:
            print("\n=== Evaluating LoRA (with adapter) ===")
            model, processor = build_model(args.base_model, adapter=args.adapter, device=device)
            correct2, n2 = evaluate(model, processor, ds, device, limit=args.limit, max_new_tokens=args.max_new_tokens)
            print(f"LoRA ChartQA acc on {n2} samples: {correct2/n2:.3f}")
        else:
            print("No adapter path provided; skipped LoRA evaluation.")
    else:
        # Single mode (existing behavior)
        model, processor = build_model(args.base_model, args.adapter, device)
        correct, n = evaluate(model, processor, ds, device, limit=args.limit, max_new_tokens=args.max_new_tokens)
        label = "LoRA" if args.adapter else "BASE"
        print(f"{label} ChartQA acc on {n} samples: {correct/n:.3f}")

if __name__ == "__main__":
    main()
