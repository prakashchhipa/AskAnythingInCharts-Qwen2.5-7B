#!/usr/bin/env python3
"""
Find ChartQA examples where the trained model performs better than the base model.
This script generates a demo-ready report of improved predictions.
"""
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
from datetime import datetime


def build_model(base_model, adapter=None, device="cuda"):
    """Build model with optional LoRA adapter"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)

    if adapter:
        print(f"Loading LoRA adapter from: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        print("Merging adapter weights into base model...")
        model = model.merge_and_unload()
        print("Adapter loaded and merged successfully!")

    processor_src = adapter if adapter else base_model
    processor = AutoProcessor.from_pretrained(processor_src)
    return model, processor


def infer(model, processor, image, question, device, max_new_tokens=64):
    """Run inference on a single image-question pair"""
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
    """Convert string to number, handling k/m suffixes"""
    x = s.strip().lower().replace(",", "")
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
    """Normalize answer for comparison"""
    s = s.strip().lower()
    for ch in [",", "%", "$", ":", ";", ".", " "]:
        s = s.replace(ch, "")
    return s


def is_correct(pred: str, gt: str) -> bool:
    """Check if prediction matches ground truth"""
    p_norm = normalize_answer(str(pred))
    g_norm = normalize_answer(str(gt))
    
    if p_norm == g_norm:
        return True
    
    # Try numeric match with small tolerance
    pn = _to_number(str(pred))
    gn = _to_number(str(gt))
    if pn is not None and gn is not None:
        if abs(pn - gn) <= max(0.5, 0.01 * abs(gn)):
            return True
    
    return False


def compare_models(base_model_path, adapter_path, device, limit=500, max_new_tokens=64):
    """Compare base and trained models, return detailed results"""
    ds = load_dataset("HuggingFaceM4/ChartQA", trust_remote_code=True)["val"]
    results = []
    
    # Evaluate BASE model
    print("=== Evaluating BASE model ===")
    base_model, base_processor = build_model(base_model_path, adapter=None, device=device)
    
    base_predictions = []
    for idx, sample in enumerate(ds.select(range(min(limit, len(ds))))):
        pred = infer(base_model, base_processor, sample["image"], sample["query"], device, max_new_tokens)
        base_predictions.append(pred)
        if (idx + 1) % 50 == 0:
            print(f"Base model: {idx + 1}/{limit} done")
    
    del base_model, base_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Evaluate TRAINED model
    print("\n=== Evaluating TRAINED model ===")
    trained_model, trained_processor = build_model(base_model_path, adapter=adapter_path, device=device)
    
    trained_predictions = []
    for idx, sample in enumerate(ds.select(range(min(limit, len(ds))))):
        pred = infer(trained_model, trained_processor, sample["image"], sample["query"], device, max_new_tokens)
        trained_predictions.append(pred)
        if (idx + 1) % 50 == 0:
            print(f"Trained model: {idx + 1}/{limit} done")
    
    del trained_model, trained_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Compare results
    print("\n=== Comparing results ===")
    base_correct = 0
    trained_correct = 0
    improved_count = 0
    
    for idx, sample in enumerate(ds.select(range(min(limit, len(ds))))):
        gt = sample["label"][0] if isinstance(sample["label"], list) else sample["label"]
        base_pred = base_predictions[idx]
        trained_pred = trained_predictions[idx]
        
        base_is_correct = is_correct(base_pred, gt)
        trained_is_correct = is_correct(trained_pred, gt)
        
        if base_is_correct:
            base_correct += 1
        if trained_is_correct:
            trained_correct += 1
        
        # Categories:
        # 1. Both wrong -> both_wrong
        # 2. Base wrong, Trained correct -> improved (BEST FOR DEMO!)
        # 3. Both correct -> both_correct
        # 4. Base correct, Trained wrong -> regressed
        
        category = None
        if not base_is_correct and not trained_is_correct:
            category = "both_wrong"
        elif not base_is_correct and trained_is_correct:
            category = "improved"
            improved_count += 1
        elif base_is_correct and trained_is_correct:
            category = "both_correct"
        else:  # base correct, trained wrong
            category = "regressed"
        
        results.append({
            "index": idx,
            "question": sample["query"],
            "ground_truth": gt,
            "base_prediction": base_pred,
            "trained_prediction": trained_pred,
            "base_correct": base_is_correct,
            "trained_correct": trained_is_correct,
            "category": category,
            "image": sample["image"]  # PIL Image object
        })
    
    print(f"\n=== Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Base model accuracy: {base_correct}/{len(results)} = {base_correct/len(results)*100:.1f}%")
    print(f"Trained model accuracy: {trained_correct}/{len(results)} = {trained_correct/len(results)*100:.1f}%")
    print(f"Improved (base wrong → trained correct): {improved_count}")
    
    return results


def save_results(results, output_dir):
    """Save results to disk with images and reports"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different categories
    improved_dir = output_path / "improved"
    both_correct_dir = output_path / "both_correct"
    regressed_dir = output_path / "regressed"
    both_wrong_dir = output_path / "both_wrong"
    
    for d in [improved_dir, both_correct_dir, regressed_dir, both_wrong_dir]:
        d.mkdir(exist_ok=True)
    
    # Organize results by category
    categorized = {
        "improved": [],
        "both_correct": [],
        "regressed": [],
        "both_wrong": []
    }
    
    for result in results:
        category = result["category"]
        categorized[category].append(result)
    
    # Save images and create JSON reports for each category
    for category, items in categorized.items():
        category_dir = output_path / category
        category_data = []
        
        for i, item in enumerate(items):
            # Save image
            img_filename = f"example_{i:04d}.png"
            img_path = category_dir / img_filename
            item["image"].save(img_path)
            
            # Add to JSON (without PIL Image object)
            category_data.append({
                "index": item["index"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "base_prediction": item["base_prediction"],
                "trained_prediction": item["trained_prediction"],
                "base_correct": item["base_correct"],
                "trained_correct": item["trained_correct"],
                "image_file": img_filename
            })
        
        # Save JSON report
        json_path = category_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(category_data, f, indent=2)
        
        print(f"Saved {len(items)} examples to {category_dir}/")
    
    # Create HTML report for improved examples (best for demo)
    create_html_report(categorized["improved"], improved_dir)
    
    # Create overall summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "improved": len(categorized["improved"]),
        "both_correct": len(categorized["both_correct"]),
        "regressed": len(categorized["regressed"]),
        "both_wrong": len(categorized["both_wrong"]),
        "base_accuracy": sum(1 for r in results if r["base_correct"]) / len(results),
        "trained_accuracy": sum(1 for r in results if r["trained_correct"]) / len(results)
    }
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== All results saved to {output_path}/ ===")
    print(f"Check {improved_dir}/ for examples where trained model improved!")


def create_html_report(improved_items, output_dir):
    """Create an HTML report for improved examples (great for demos)"""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>ChartQA: Improved Examples (Trained Model)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; }
        .example { 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .example img { 
            max-width: 600px; 
            border: 1px solid #ddd; 
            border-radius: 4px;
            margin: 10px 0;
        }
        .question { font-weight: bold; color: #2c3e50; margin: 10px 0; }
        .ground-truth { color: #27ae60; margin: 5px 0; }
        .base-pred { color: #e74c3c; margin: 5px 0; }
        .trained-pred { color: #27ae60; margin: 5px 0; font-weight: bold; }
        .label { font-weight: bold; }
        .summary { 
            background: #3498db; 
            color: white; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>🎯 ChartQA: Examples Where Trained Model Outperforms Base</h1>
    <div class="summary">
        <p><strong>Total Improved Examples:</strong> """ + str(len(improved_items)) + """</p>
        <p>These are examples where the base model got it wrong, but the trained model got it right!</p>
    </div>
"""
    
    for i, item in enumerate(improved_items):
        html_content += f"""
    <div class="example">
        <h3>Example {i+1} (Dataset Index: {item['index']})</h3>
        <img src="example_{i:04d}.png" alt="Chart">
        <div class="question"><span class="label">Question:</span> {item['question']}</div>
        <div class="ground-truth"><span class="label">Ground Truth:</span> {item['ground_truth']}</div>
        <div class="base-pred"><span class="label">❌ Base Model:</span> {item['base_prediction']}</div>
        <div class="trained-pred"><span class="label">✅ Trained Model:</span> {item['trained_prediction']}</div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    html_path = output_dir / "improved_examples.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"HTML report created: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find ChartQA examples where trained model performs better than base model"
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base model path"
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to trained LoRA adapter"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        default="demo_examples",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Compare models and get detailed results
    results = compare_models(
        args.base_model,
        args.adapter,
        device,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens
    )
    
    # Save results to disk
    save_results(results, args.output_dir)
    
    print("\n✅ Done! Check the output directory for demo-ready examples.")


if __name__ == "__main__":
    main()

