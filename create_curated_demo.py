#!/usr/bin/env python3
"""
Create a curated demo with only user-selected examples.
"""

import json
import shutil
from pathlib import Path

# User-selected example indices (from the original results.json)
SELECTED_INDICES = [191, 267, 317, 360, 391, 393, 471, 495]

def create_curated_demo(input_dir, output_dir, selected_indices):
    """Create demo with only selected examples"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    with open(input_path / "results.json") as f:
        all_results = json.load(f)
    
    # Filter by selected indices
    selected_examples = []
    for example in all_results:
        if example["index"] in selected_indices:
            selected_examples.append(example)
    
    # Sort by the order in SELECTED_INDICES
    selected_examples.sort(key=lambda x: selected_indices.index(x["index"]))
    
    print(f"Found {len(selected_examples)} examples out of {len(selected_indices)} requested")
    
    # Copy images and create new results
    curated_results = []
    for i, example in enumerate(selected_examples):
        old_img = input_path / example["image_file"]
        new_img_name = f"example_{i:04d}.png"
        new_img = output_path / new_img_name
        
        if old_img.exists():
            shutil.copy2(old_img, new_img)
            print(f"✓ Copied {example['image_file']} → {new_img_name} (index {example['index']})")
        else:
            print(f"✗ Warning: {old_img} not found!")
        
        # Update entry
        new_entry = example.copy()
        new_entry["image_file"] = new_img_name
        curated_results.append(new_entry)
    
    # Save curated results
    with open(output_path / "results.json", "w") as f:
        json.dump(curated_results, f, indent=2)
    
    print(f"\n✅ Saved {len(curated_results)} curated examples to: {output_path}/")
    
    # Create HTML report
    create_html_report(curated_results, output_path)
    
    return curated_results


def create_html_report(examples, output_dir):
    """Create HTML report for curated examples"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Curated ChartQA Demo Examples</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .summary { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 20px; 
            border-radius: 12px; 
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .example { 
            background: white; 
            padding: 25px; 
            margin: 25px 0; 
            border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .example:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .example img { 
            max-width: 700px; 
            border: 2px solid #e0e0e0; 
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .question { 
            font-weight: bold; 
            color: #2c3e50; 
            margin: 15px 0;
            font-size: 18px;
        }
        .ground-truth { 
            color: #27ae60; 
            margin: 10px 0;
            padding: 10px;
            background: #e8f5e9;
            border-left: 4px solid #27ae60;
            border-radius: 4px;
        }
        .base-pred { 
            color: #e74c3c; 
            margin: 10px 0;
            padding: 10px;
            background: #ffebee;
            border-left: 4px solid #e74c3c;
            border-radius: 4px;
        }
        .trained-pred { 
            color: #2196f3; 
            margin: 10px 0;
            padding: 10px;
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            border-radius: 4px;
            font-weight: bold;
        }
        .label { font-weight: bold; }
        .index { 
            color: #666; 
            font-size: 14px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>📊 Curated ChartQA Demo Examples</h1>
    <div class="summary">
        <h2 style="margin-top: 0;">🎯 Hand-Picked Improvements</h2>
        <p style="font-size: 18px; margin: 10px 0;">
            <strong>""" + str(len(examples)) + """ carefully selected examples</strong> 
            showing clear improvements from fine-tuning
        </p>
        <p style="margin: 5px 0;">✓ Base model was wrong</p>
        <p style="margin: 5px 0;">✓ Fine-tuned model got it right</p>
        <p style="margin: 5px 0;">✓ Clear, demonstrable differences</p>
    </div>
"""
    
    for i, item in enumerate(examples):
        html += f"""
    <div class="example">
        <h3>Example {i+1} <span class="index">(Dataset Index: {item['index']})</span></h3>
        <img src="{item['image_file']}" alt="Chart">
        <div class="question"><span class="label">❓ Question:</span> {item['question']}</div>
        <div class="ground-truth"><span class="label">✓ Ground Truth:</span> {item['ground_truth']}</div>
        <div class="base-pred"><span class="label">❌ Base Model:</span> {item['base_prediction']}</div>
        <div class="trained-pred"><span class="label">✅ Fine-tuned Model:</span> {item['trained_prediction']}</div>
    </div>
"""
    
    html += """
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
        <p style="font-size: 14px;">Generated for demo purposes</p>
    </div>
</body>
</html>
"""
    
    html_path = output_dir / "curated_demo.html"
    with open(html_path, "w") as f:
        f.write(html)
    
    print(f"✓ Created HTML report: {html_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create curated demo with selected examples")
    parser.add_argument(
        "--input_dir",
        default="demo_chatqa/improved",
        help="Input directory with all examples"
    )
    parser.add_argument(
        "--output_dir",
        default="demo_curated",
        help="Output directory for curated examples"
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=SELECTED_INDICES,
        help="List of indices to include"
    )
    
    args = parser.parse_args()
    
    print(f"Creating curated demo with indices: {args.indices}")
    print()
    
    results = create_curated_demo(args.input_dir, args.output_dir, args.indices)
    
    print("\n" + "="*60)
    print("📋 Summary of Selected Examples:")
    print("="*60)
    for i, ex in enumerate(results):
        print(f"\n{i+1}. [Index {ex['index']}] {ex['question'][:60]}...")
        print(f"   Base: {ex['base_prediction'][:50]}...")
        print(f"   Trained: {ex['trained_prediction'][:50]}...")
    
    print("\n" + "="*60)
    print(f"✅ Curated demo ready in: {args.output_dir}/")
    print(f"   - {len(results)} examples")
    print(f"   - results.json")
    print(f"   - example_*.png")
    print(f"   - curated_demo.html (Open this!)")
    print("="*60)
    
    print("\n📖 Next step: Update app.py to use demo_curated/")
    print("   Change: EXAMPLES_DIR = Path('demo_curated')")


if __name__ == "__main__":
    main()

