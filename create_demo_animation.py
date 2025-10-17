#!/usr/bin/env python3
"""
Create animated demo for GitHub README
Shows side-by-side comparison of base vs fine-tuned model
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import time
import os

# Load example data
EXAMPLES_DIR = Path("hf_space_deploy/demo_curated")
if EXAMPLES_DIR.exists():
    with open(EXAMPLES_DIR / "results.json") as f:
        EXAMPLE_DATA = json.load(f)
else:
    EXAMPLE_DATA = []

def create_demo_gif():
    """Create an animated GIF showing model comparison"""
    
    # Select 3-4 best examples
    demo_examples = EXAMPLE_DATA[:4] if len(EXAMPLE_DATA) >= 4 else EXAMPLE_DATA
    
    if not demo_examples:
        print("No examples found. Please run the demo first to generate examples.")
        return
    
    frames = []
    
    for i, example in enumerate(demo_examples):
        # Load the chart image
        img_path = EXAMPLES_DIR / example["image_file"]
        if not img_path.exists():
            continue
            
        chart_img = Image.open(img_path)
        
        # Create a frame with the chart and comparison
        frame = create_comparison_frame(
            chart_img, 
            example["question"],
            example["base_prediction"],
            example["trained_prediction"],
            example["ground_truth"],
            i + 1,
            len(demo_examples)
        )
        
        frames.append(frame)
    
    # Save as GIF
    if frames:
        output_path = "demo_animation.gif"
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=3000,  # 3 seconds per frame
            loop=0
        )
        print(f"✅ Demo animation saved as: {output_path}")
        return output_path
    else:
        print("❌ No frames created")
        return None

def create_comparison_frame(chart_img, question, base_pred, trained_pred, ground_truth, frame_num, total_frames):
    """Create a single frame showing the comparison"""
    
    # Resize chart to fit nicely
    chart_img = chart_img.resize((400, 300), Image.Resampling.LANCZOS)
    
    # Create main frame (1200x800)
    frame = Image.new('RGB', (1200, 800), 'white')
    draw = ImageDraw.Draw(frame)
    
    # Try to load a font, fallback to default if not available
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Title
    draw.text((50, 20), f"AskAnything in Charts - Powered by Qwen 2.5", fill='black', font=title_font)
    draw.text((50, 50), f"Example {frame_num}/{total_frames}", fill='gray', font=small_font)
    
    # Question
    draw.text((50, 80), f"Question: {question}", fill='black', font=text_font)
    
    # Chart (left side)
    frame.paste(chart_img, (50, 120))
    
    # Model comparison (right side)
    x_offset = 500
    
    # Base model result
    draw.text((x_offset, 120), "Qwen 2.5 7B (Base):", fill='red', font=text_font)
    draw.text((x_offset, 150), f"Answer: {base_pred}", fill='black', font=text_font)
    
    # Fine-tuned model result
    draw.text((x_offset, 200), "Qwen 2.5 7B + LORA:", fill='green', font=text_font)
    draw.text((x_offset, 230), f"Answer: {trained_pred}", fill='black', font=text_font)
    
    # Ground truth
    draw.text((x_offset, 280), f"Ground Truth: {ground_truth}", fill='blue', font=text_font)
    
    # Performance indicators
    base_correct = "✓" if base_pred.strip().lower() == ground_truth.strip().lower() else "✗"
    trained_correct = "✓" if trained_pred.strip().lower() == ground_truth.strip().lower() else "✗"
    
    draw.text((x_offset, 320), f"Base Model: {base_correct}", fill='red' if base_correct == "✗" else 'green', font=text_font)
    draw.text((x_offset, 350), f"Fine-tuned: {trained_correct}", fill='red' if trained_correct == "✗" else 'green', font=text_font)
    
    # Improvement indicator
    if base_correct == "✗" and trained_correct == "✓":
        draw.text((x_offset, 380), "🎉 IMPROVEMENT!", fill='green', font=text_font)
    elif base_correct == "✓" and trained_correct == "✗":
        draw.text((x_offset, 380), "⚠️ REGRESSION", fill='red', font=text_font)
    elif base_correct == "✓" and trained_correct == "✓":
        draw.text((x_offset, 380), "✅ BOTH CORRECT", fill='blue', font=text_font)
    else:
        draw.text((x_offset, 380), "❌ BOTH WRONG", fill='gray', font=text_font)
    
    # Footer
    draw.text((50, 750), "Try it live: https://huggingface.co/spaces/prakashchhipa/chart-qa-demo-qwen2.5", 
              fill='gray', font=small_font)
    
    return frame

def create_static_demo_image():
    """Create a static demo image for the README"""
    
    if not EXAMPLE_DATA:
        print("No examples found. Please run the demo first to generate examples.")
        return
    
    # Use the first example
    example = EXAMPLE_DATA[0]
    img_path = EXAMPLES_DIR / example["image_file"]
    
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
    
    chart_img = Image.open(img_path)
    chart_img = chart_img.resize((600, 400), Image.Resampling.LANCZOS)
    
    # Create demo image (1000x600)
    demo_img = Image.new('RGB', (1000, 600), 'white')
    draw = ImageDraw.Draw(demo_img)
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Title
    draw.text((20, 20), "AskAnything in Charts - Powered by Qwen 2.5", fill='black', font=title_font)
    
    # Chart (left side)
    demo_img.paste(chart_img, (20, 70))
    
    # Question
    draw.text((20, 490), f"Question: {example['question']}", fill='black', font=text_font)
    
    # Model comparison (right side)
    x_offset = 650
    
    # Base model result
    draw.text((x_offset, 70), "Qwen 2.5 7B (Base):", fill='red', font=text_font)
    draw.text((x_offset, 100), f"Answer: {example['base_prediction']}", fill='black', font=text_font)
    
    # Fine-tuned model result
    draw.text((x_offset, 150), "Qwen 2.5 7B + LORA:", fill='green', font=text_font)
    draw.text((x_offset, 180), f"Answer: {example['trained_prediction']}", fill='black', font=text_font)
    
    # Ground truth
    draw.text((x_offset, 230), f"Ground Truth: {example['ground_truth']}", fill='blue', font=text_font)
    
    # Performance indicators
    base_correct = "✓" if example['base_prediction'].strip().lower() == example['ground_truth'].strip().lower() else "✗"
    trained_correct = "✓" if example['trained_prediction'].strip().lower() == example['ground_truth'].strip().lower() else "✗"
    
    draw.text((x_offset, 280), f"Base Model: {base_correct}", fill='red' if base_correct == "✗" else 'green', font=text_font)
    draw.text((x_offset, 310), f"Fine-tuned: {trained_correct}", fill='red' if trained_correct == "✗" else 'green', font=text_font)
    
    # Improvement indicator
    if base_correct == "✗" and trained_correct == "✓":
        draw.text((x_offset, 350), "🎉 IMPROVEMENT!", fill='green', font=text_font)
    
    # Footer
    draw.text((20, 550), "Try it live: https://huggingface.co/spaces/prakashchhipa/chart-qa-demo-qwen2.5", 
              fill='gray', font=small_font)
    
    # Save
    output_path = "demo_comparison.png"
    demo_img.save(output_path)
    print(f"✅ Static demo image saved as: {output_path}")
    return output_path

if __name__ == "__main__":
    print("🎬 Creating demo animations...")
    
    # Create static demo image
    static_img = create_static_demo_image()
    
    # Create animated GIF
    gif_path = create_demo_gif()
    
    print("\n✅ Demo files created:")
    if static_img:
        print(f"  📸 Static image: {static_img}")
    if gif_path:
        print(f"  🎬 Animated GIF: {gif_path}")
    
    print("\n📝 Add these to your README.md:")
    if static_img:
        print(f"![Demo]({static_img})")
    if gif_path:
        print(f"![Animated Demo]({gif_path})")
