#!/usr/bin/env python3
"""
Filter out false positive "improvements" where the base model actually had the right answer
but was too verbose. Keep only genuine improvements where base was truly wrong.
"""

import json
import shutil
from pathlib import Path
import re


def normalize_for_comparison(s: str) -> str:
    """Normalize string for loose comparison"""
    s = s.strip().lower()
    # Remove punctuation and extra spaces
    s = re.sub(r'[,\.;:!\?"\']', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def extract_number(s: str):
    """Extract first number from string"""
    # Try to find numbers (including decimals, percentages, etc.)
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        try:
            return float(match.group())
        except:
            pass
    return None


def is_genuine_improvement(example):
    """
    Check if this is a genuine improvement or just a verbose vs concise difference.
    
    Returns True only if:
    1. Base model answer doesn't contain the ground truth (even loosely)
    2. Base model's extracted number doesn't match ground truth number
    3. Base model made a genuine semantic error
    """
    base_pred = example["base_prediction"]
    trained_pred = example["trained_prediction"]
    ground_truth = example["ground_truth"]
    
    # Normalize all answers
    base_norm = normalize_for_comparison(base_pred)
    trained_norm = normalize_for_comparison(trained_pred)
    gt_norm = normalize_for_comparison(ground_truth)
    
    # Check 1: Does base prediction contain the ground truth?
    # If yes, it's probably just verbose, not wrong
    if gt_norm in base_norm:
        return False
    
    # Check 2: For numeric answers, check if base has the right number somewhere
    gt_num = extract_number(ground_truth)
    if gt_num is not None:
        base_num = extract_number(base_pred)
        if base_num is not None:
            # Allow small tolerance
            if abs(base_num - gt_num) <= max(0.5, 0.01 * abs(gt_num)):
                return False
    
    # Check 3: Common false positive patterns
    # Base says "X is Y" when ground truth is just "Y"
    # Pattern: "... is ANSWER" or "... represents ANSWER" or "... shows ANSWER"
    verbose_patterns = [
        rf'\bis\s+{re.escape(gt_norm)}',
        rf'\brepresents?\s+{re.escape(gt_norm)}',
        rf'\bshows?\s+{re.escape(gt_norm)}',
        rf'\bcalled?\s+{re.escape(gt_norm)}',
        rf'\bnamed?\s+{re.escape(gt_norm)}',
        rf'{re.escape(gt_norm)}\s+\(',  # "ANSWER (with explanation)"
    ]
    
    for pattern in verbose_patterns:
        if re.search(pattern, base_norm):
            return False
    
    # Check 4: For ratio/comparison questions, check if base has the calculation
    # e.g., Ground truth: "2.125", Base: "21:8" (which is 2.625)
    if gt_num is not None and ':' in base_pred:
        # Try to parse ratio
        parts = base_pred.split(':')
        if len(parts) == 2:
            try:
                ratio = float(parts[0].strip()) / float(parts[1].strip())
                if abs(ratio - gt_num) <= max(0.5, 0.1 * abs(gt_num)):
                    return False
            except:
                pass
    
    # Check 5: Yes/No questions with explanations
    if gt_norm in ['yes', 'no']:
        # Check if base starts with yes/no
        if base_norm.startswith(gt_norm):
            return False
    
    # If none of the false positive patterns matched, this is likely genuine
    return True


def filter_examples(input_dir, output_dir):
    """Filter examples and copy only genuine improvements"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(input_path / "results.json") as f:
        all_examples = json.load(f)
    
    print(f"Total improved examples: {len(all_examples)}")
    
    # Filter examples
    genuine_examples = []
    filtered_out = []
    
    for example in all_examples:
        if is_genuine_improvement(example):
            genuine_examples.append(example)
        else:
            filtered_out.append(example)
    
    print(f"Genuine improvements: {len(genuine_examples)}")
    print(f"Filtered out (verbose but correct): {len(filtered_out)}")
    
    # Copy images and update JSON for genuine examples
    genuine_json = []
    for i, example in enumerate(genuine_examples):
        # Copy image
        old_img = input_path / example["image_file"]
        new_img_name = f"example_{i:04d}.png"
        new_img = output_path / new_img_name
        
        if old_img.exists():
            shutil.copy2(old_img, new_img)
        
        # Update JSON entry
        new_entry = example.copy()
        new_entry["image_file"] = new_img_name
        genuine_json.append(new_entry)
    
    # Save genuine examples JSON
    with open(output_path / "results.json", "w") as f:
        json.dump(genuine_json, f, indent=2)
    
    # Save filtered out examples for review
    with open(output_path / "filtered_out.json", "w") as f:
        json.dump(filtered_out, f, indent=2)
    
    # Create HTML report for genuine examples
    create_html_report(genuine_json, output_path)
    
    # Create HTML report for filtered out (for review)
    create_review_html(filtered_out, input_path, output_path)
    
    print(f"\n✅ Filtered results saved to: {output_path}/")
    print(f"   - {len(genuine_examples)} genuine improvements")
    print(f"   - results.json (genuine examples)")
    print(f"   - filtered_out.json (verbose but correct)")
    print(f"   - genuine_improvements.html")
    print(f"   - review_filtered.html (check these manually)")


def create_html_report(examples, output_dir):
    """Create HTML report for genuine improvements"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Genuine ChartQA Improvements</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .summary { background: #4caf50; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .example { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .example img { max-width: 600px; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }
        .question { font-weight: bold; color: #2c3e50; margin: 10px 0; }
        .ground-truth { color: #27ae60; margin: 5px 0; }
        .base-pred { color: #e74c3c; margin: 5px 0; }
        .trained-pred { color: #27ae60; margin: 5px 0; font-weight: bold; }
        .label { font-weight: bold; }
    </style>
</head>
<body>
    <h1>🎯 Genuine ChartQA Improvements</h1>
    <div class="summary">
        <p><strong>These are GENUINE improvements where the base model was truly wrong</strong></p>
        <p>Total: """ + str(len(examples)) + """ examples</p>
        <p>Filtered out verbose-but-correct examples to show only real improvements!</p>
    </div>
"""
    
    for i, item in enumerate(examples):
        html += f"""
    <div class="example">
        <h3>Example {i+1} (Original Index: {item['index']})</h3>
        <img src="{item['image_file']}" alt="Chart">
        <div class="question"><span class="label">Question:</span> {item['question']}</div>
        <div class="ground-truth"><span class="label">✓ Ground Truth:</span> {item['ground_truth']}</div>
        <div class="base-pred"><span class="label">❌ Base Model:</span> {item['base_prediction']}</div>
        <div class="trained-pred"><span class="label">✅ Trained Model:</span> {item['trained_prediction']}</div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_dir / "genuine_improvements.html", "w") as f:
        f.write(html)


def create_review_html(filtered_examples, input_dir, output_dir):
    """Create HTML to review filtered out examples"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Filtered Examples (Review)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .summary { background: #ff9800; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .example { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .example img { max-width: 600px; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }
        .question { font-weight: bold; color: #2c3e50; margin: 10px 0; }
        .ground-truth { color: #27ae60; margin: 5px 0; }
        .base-pred { color: #ff9800; margin: 5px 0; background: #fff3e0; padding: 5px; }
        .trained-pred { color: #27ae60; margin: 5px 0; font-weight: bold; }
        .label { font-weight: bold; }
    </style>
</head>
<body>
    <h1>⚠️ Filtered Examples (Base was verbose but correct)</h1>
    <div class="summary">
        <p><strong>These examples were filtered out</strong></p>
        <p>The base model had the correct answer but was too verbose</p>
        <p>Total: """ + str(len(filtered_examples)) + """ examples</p>
    </div>
"""
    
    for i, item in enumerate(filtered_examples):
        img_path = Path(input_dir) / item['image_file']
        rel_path = img_path.relative_to(output_dir.parent) if img_path.exists() else item['image_file']
        
        html += f"""
    <div class="example">
        <h3>Filtered {i+1} (Original Index: {item['index']})</h3>
        <img src="../{rel_path}" alt="Chart">
        <div class="question"><span class="label">Question:</span> {item['question']}</div>
        <div class="ground-truth"><span class="label">✓ Ground Truth:</span> {item['ground_truth']}</div>
        <div class="base-pred"><span class="label">⚠️ Base Model (verbose but has answer):</span> {item['base_prediction']}</div>
        <div class="trained-pred"><span class="label">✅ Trained Model (concise):</span> {item['trained_prediction']}</div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_dir / "review_filtered.html", "w") as f:
        f.write(html)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter genuine improvements from verbose-but-correct examples"
    )
    parser.add_argument(
        "--input_dir",
        default="demo_chatqa/improved",
        help="Input directory with all improved examples"
    )
    parser.add_argument(
        "--output_dir",
        default="demo_genuine",
        help="Output directory for filtered genuine improvements"
    )
    
    args = parser.parse_args()
    
    filter_examples(args.input_dir, args.output_dir)
    
    print("\n📊 Next Steps:")
    print("1. Open genuine_improvements.html to see filtered results")
    print("2. Open review_filtered.html to verify filtered examples")
    print("3. Use demo_genuine/ folder for your demo instead of demo_chatqa/improved/")


if __name__ == "__main__":
    main()

