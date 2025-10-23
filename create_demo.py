#!/usr/bin/env python3
"""
Create perfect demo GIF with all improvements
"""

import os
import math
import json
from PIL import Image, ImageDraw, ImageFont

# Configuration
WIDTH, HEIGHT = 1400, 900
BACKGROUND_COLOR = (0, 0, 0)  # Black background
TEXT_COLOR = (255, 255, 255)  # White text
ACCENT_COLOR = (0, 255, 255)  # Cyan accent
SUCCESS_COLOR = (0, 255, 0)   # Green for correct answers
WARNING_COLOR = (255, 165, 0) # Orange for incorrect answers
HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow for highlights

# Timing (previous better speed)
FRAME_DURATION = 2000  # 2 seconds per frame
TRANSITION_DURATION = 400  # 0.4 seconds for transitions (previous speed)
LOADING_TRANSITION = 300  # 0.3 seconds for loading (a bit slower than current)

# Chart configuration
LARGE_CHART_WIDTH = 800
LARGE_CHART_HEIGHT = 500
SMALL_CHART_WIDTH = 400
SMALL_CHART_HEIGHT = 250

def create_black_background():
    """Create a black background image"""
    return Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND_COLOR)

def get_font(size, bold=False):
    """Get font with fallback"""
    try:
        if bold:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        else:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()

def wrap_text(text, font, max_width):
    """Wrap text to fit within max_width"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def create_title_frame():
    """Create the title frame with project info and name at bottom"""
    img = create_black_background()
    draw = ImageDraw.Draw(img)
    
    # Project title (main focus)
    title_font = get_font(48, bold=True)
    title_text = "AskAnythingInCharts"
    title_bbox = title_font.getbbox(title_text)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (WIDTH - title_width) // 2
    title_y = HEIGHT // 2 - 120
    
    # Draw title with outline
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            if dx != 0 or dy != 0:
                draw.text((title_x + dx, title_y + dy), title_text, fill=(0, 0, 0), font=title_font)
    draw.text((title_x, title_y), title_text, fill=HIGHLIGHT_COLOR, font=title_font)
    
    # Subtitle
    subtitle_font = get_font(24)
    subtitle_text = "Fine-tuned Qwen2.5-VL-7B for Chart Understanding"
    subtitle_bbox = subtitle_font.getbbox(subtitle_text)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = (WIDTH - subtitle_width) // 2
    subtitle_y = title_y + 70
    
    draw.text((subtitle_x, subtitle_y), subtitle_text, fill=TEXT_COLOR, font=subtitle_font)
    
    # Performance metrics
    metrics_font = get_font(20)
    metrics_text = "🎯 66.0% Accuracy on ChartQA (+8.5% improvement)"
    metrics_bbox = metrics_font.getbbox(metrics_text)
    metrics_width = metrics_bbox[2] - metrics_bbox[0]
    metrics_x = (WIDTH - metrics_width) // 2
    metrics_y = subtitle_y + 50
    
    draw.text((metrics_x, metrics_y), metrics_text, fill=SUCCESS_COLOR, font=metrics_font)
    
    # Your name (bigger and higher up)
    name_font = get_font(22)
    name_text = "Prakash Chandra Chhipa, October 2025"
    name_bbox = name_font.getbbox(name_text)
    name_width = name_bbox[2] - name_bbox[0]
    name_x = (WIDTH - name_width) // 2
    name_y = HEIGHT - 120
    
    draw.text((name_x, name_y), name_text, fill=ACCENT_COLOR, font=name_font)
    
    return img

def create_intro_sequence():
    """Create intro sequence with your name and project"""
    frames = []
    
    # Single frame with your name and project
    frame = create_title_frame()
    frames.append(frame)
    frames.append(frame)  # Hold
    frames.append(frame)  # Hold
    
    return frames

def create_loading_animation():
    """Create loading animation (a bit slower than very fast)"""
    frames = []
    
    for i in range(3):  # 3 frames for better animation
        img = create_black_background()
        draw = ImageDraw.Draw(img)
        
        # Loading text
        font = get_font(28)
        text = "Loading Chart Examples"
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (WIDTH - text_width) // 2
        text_y = HEIGHT // 2 - 50
        
        draw.text((text_x, text_y), text, fill=TEXT_COLOR, font=font)
        
        # Animated dots
        dot_y = text_y + 50
        for j in range(3):
            dot_x = text_x + text_width + 20 + j * 30
            if j < i:
                draw.ellipse([dot_x, dot_y, dot_x + 8, dot_y + 8], fill=ACCENT_COLOR)
            else:
                draw.ellipse([dot_x, dot_y, dot_x + 8, dot_y + 8], fill=(100, 100, 100))
        
        frames.append(img)
    
    return frames

def is_correct(answer, ground_truth):
    """Check if answer is correct with numerical tolerance"""
    if isinstance(answer, (int, float)) and isinstance(ground_truth, (int, float)):
        return math.isclose(float(answer), float(ground_truth), rel_tol=0.01, abs_tol=0.1)
    
    try:
        num_answer = float(str(answer).replace(' hours', '').strip())
        num_ground_truth = float(str(ground_truth).replace(' hours', '').strip())
        return math.isclose(num_answer, num_ground_truth, rel_tol=0.01, abs_tol=0.1)
    except ValueError:
        pass

    return str(answer).strip().lower() == str(ground_truth).strip().lower()

def create_large_chart_frame(example, current, total):
    """Create frame showing LARGE chart only"""
    img = create_black_background()
    draw = ImageDraw.Draw(img)
    
    # Load and resize chart image to LARGE size
    chart_path = f"demo/demo_curated/{example['image_file']}" if example['image_file'].startswith('example_') else example['image_file']
    if os.path.exists(chart_path):
        chart_img = Image.open(chart_path)
        chart_img = chart_img.resize((LARGE_CHART_WIDTH, LARGE_CHART_HEIGHT), Image.Resampling.LANCZOS)
        
        # Center the chart
        chart_x = (WIDTH - LARGE_CHART_WIDTH) // 2
        chart_y = (HEIGHT - LARGE_CHART_HEIGHT) // 2 - 30
        
        img.paste(chart_img, (chart_x, chart_y))
    
    # Example counter
    counter_font = get_font(20, bold=True)
    counter_text = f"📊 Example {current} of {total}"
    counter_bbox = counter_font.getbbox(counter_text)
    counter_width = counter_bbox[2] - counter_bbox[0]
    counter_x = (WIDTH - counter_width) // 2
    counter_y = 50
    
    # Counter background
    padding = 15
    draw.rectangle([counter_x - padding, counter_y - 8, counter_x + counter_width + padding, counter_y + 30], 
                  fill=(30, 30, 30), outline=ACCENT_COLOR, width=2)
    draw.text((counter_x, counter_y), counter_text, fill=ACCENT_COLOR, font=counter_font)
    
    return img

def create_full_comparison_frame(example, current, total):
    """Create frame with LARGE chart on top and responses below"""
    img = create_black_background()
    draw = ImageDraw.Draw(img)
    
    # Load and resize chart image to LARGE size for top
    chart_path = f"demo/demo_curated/{example['image_file']}" if example['image_file'].startswith('example_') else example['image_file']
    if os.path.exists(chart_path):
        chart_img = Image.open(chart_path)
        chart_img = chart_img.resize((LARGE_CHART_WIDTH, LARGE_CHART_HEIGHT), Image.Resampling.LANCZOS)
        
        # Position chart at the top
        chart_x = (WIDTH - LARGE_CHART_WIDTH) // 2
        chart_y = 80
        
        # Chart border
        draw.rectangle([chart_x - 5, chart_y - 5, chart_x + LARGE_CHART_WIDTH + 5, chart_y + LARGE_CHART_HEIGHT + 5], 
                      outline=ACCENT_COLOR, width=2)
        img.paste(chart_img, (chart_x, chart_y))
    
    # Example counter
    counter_font = get_font(18, bold=True)
    counter_text = f"📊 Example {current} of {total}"
    counter_bbox = counter_font.getbbox(counter_text)
    counter_width = counter_bbox[2] - counter_bbox[0]
    counter_x = (WIDTH - counter_width) // 2
    counter_y = 30
    
    draw.text((counter_x, counter_y), counter_text, fill=ACCENT_COLOR, font=counter_font)
    
    # Question (below the chart)
    question_font = get_font(22, bold=True)
    question_text = f"❓ {example['question']}"
    question_lines = wrap_text(question_text, question_font, WIDTH - 100)
    
    question_y = chart_y + LARGE_CHART_HEIGHT + 30
    
    for line in question_lines:
        question_x = (WIDTH - question_font.getbbox(line)[2] + question_font.getbbox(line)[0]) // 2
        draw.text((question_x, question_y), line, fill=TEXT_COLOR, font=question_font)
        question_y += 30
    
    # Model responses (below question)
    response_y = question_y + 20
    label_font = get_font(18, bold=True)
    answer_font = get_font(16)
    
    # Base model response
    base_correct = is_correct(example['base_answer'], example['ground_truth'])
    base_color = SUCCESS_COLOR if base_correct else WARNING_COLOR
    
    # Response box
    box_width = WIDTH - 100
    box_height = 50
    draw.rectangle([50, response_y - 10, 50 + box_width, response_y + box_height], 
                  fill=(20, 20, 20), outline=base_color, width=2)
    
    draw.text((60, response_y), "🤖 Qwen2.5-7B Base:", fill=base_color, font=label_font)
    base_answer_text = str(example['base_answer'])
    base_answer_lines = wrap_text(base_answer_text, answer_font, box_width - 20)
    for line in base_answer_lines:
        response_y += 20
        draw.text((70, response_y), line, fill=TEXT_COLOR, font=answer_font)
    
    # Our model response
    response_y += 30
    trained_correct = is_correct(example['trained_answer'], example['ground_truth'])
    trained_color = SUCCESS_COLOR if trained_correct else WARNING_COLOR
    
    # Response box
    draw.rectangle([50, response_y - 10, 50 + box_width, response_y + 50], 
                  fill=(20, 20, 20), outline=trained_color, width=2)
    
    draw.text((60, response_y), "🚀 AskAnythingInCharts-Qwen2.5-7B:", fill=trained_color, font=label_font)
    trained_answer_text = str(example['trained_answer'])
    trained_answer_lines = wrap_text(trained_answer_text, answer_font, box_width - 20)
    for line in trained_answer_lines:
        response_y += 20
        draw.text((70, response_y), line, fill=TEXT_COLOR, font=answer_font)
    
    # Ground truth
    response_y += 30
    draw.text((60, response_y), "✅ Ground Truth:", fill=HIGHLIGHT_COLOR, font=label_font)
    ground_truth_text = str(example['ground_truth'])
    ground_truth_lines = wrap_text(ground_truth_text, answer_font, box_width - 20)
    for line in ground_truth_lines:
        response_y += 20
        draw.text((70, response_y), line, fill=HIGHLIGHT_COLOR, font=answer_font)
    
    return img

def create_example_sequence(example, current, total):
    """Create a sequence of frames for each example"""
    frames = []
    
    # Frame 1: LARGE chart only - hold
    frame1 = create_large_chart_frame(example, current, total)
    frames.append(frame1)
    frames.append(frame1)  # Hold
    
    # Frame 2: Full comparison with chart on top - hold (20% faster)
    frame2 = create_full_comparison_frame(example, current, total)
    frames.append(frame2)
    frames.append(frame2)  # Hold
    frames.append(frame2)  # Hold
    
    return frames

def create_fast_transition():
    """Create transition between examples (previous speed)"""
    frames = []
    
    for i in range(2):  # 2 frames for previous speed
        img = create_black_background()
        draw = ImageDraw.Draw(img)
        
        # Fade effect
        alpha = int(255 * (i + 1) / 2)
        
        # Loading text
        font = get_font(24)
        text = "Next example..."
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (WIDTH - text_width) // 2
        text_y = HEIGHT // 2
        
        draw.text((text_x, text_y), text, fill=(alpha, alpha, alpha), font=font)
        
        frames.append(img)
    
    return frames

def main():
    """Create the demo GIF"""
    print("Creating demo GIF...")
    
    # Load real demo data
    with open('demo_data.json', 'r') as f:
        examples = json.load(f)
    
    # Create stacked_students example as first
    stacked_students_example = {
        "id": 8,
        "image_file": "stacked_students.png",
        "question": "Which department is having higher male student than Physics and but lower than Math?",
        "ground_truth": "Biology",
        "base_answer": "CS",
        "trained_answer": "Biology"
    }
    
    # Examples (stacked_students first, then others)
    demo_examples = [stacked_students_example] + examples[:5]
    
    all_frames = []
    
    # Intro sequence with your name
    print("Creating intro sequence with your name...")
    all_frames.extend(create_intro_sequence())
    
    # Fast loading animation
    print("Creating loading animation...")
    all_frames.extend(create_loading_animation())
    
    # Examples
    print(f"Creating example sequences for {len(demo_examples)} examples...")
    for i, example in enumerate(demo_examples, 1):
        print(f"  Processing example {i}/{len(demo_examples)}: {example['image_file']}")
        all_frames.extend(create_example_sequence(example, i, len(demo_examples)))
        
        # Add fast transition between examples (except for the last one)
        if i < len(demo_examples):
            all_frames.extend(create_fast_transition())
    
    # Final frame
    print("Creating final frame...")
    final_img = create_black_background()
    draw = ImageDraw.Draw(final_img)
    
    final_font = get_font(42, bold=True)
    final_text = "🎉 Try the Demo!"
    final_bbox = final_font.getbbox(final_text)
    final_width = final_bbox[2] - final_bbox[0]
    final_x = (WIDTH - final_width) // 2
    final_y = HEIGHT // 2 - 80
    
    # Draw title with outline
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            if dx != 0 or dy != 0:
                draw.text((final_x + dx, final_y + dy), final_text, fill=(0, 0, 0), font=final_font)
    draw.text((final_x, final_y), final_text, fill=ACCENT_COLOR, font=final_font)
    
    # Hugging Face logo (simple text representation)
    hf_font = get_font(24, bold=True)
    hf_text = "🤗"
    hf_bbox = hf_font.getbbox(hf_text)
    hf_width = hf_bbox[2] - hf_bbox[0]
    hf_x = (WIDTH - hf_width) // 2
    hf_y = final_y + 70
    
    draw.text((hf_x, hf_y), hf_text, fill=HIGHLIGHT_COLOR, font=hf_font)
    
    # Demo link (bigger)
    link_font = get_font(24, bold=True)
    link_text = "https://huggingface.co/spaces/prakashchhipa/chart-qa-demo-qwen2.5"
    link_bbox = link_font.getbbox(link_text)
    link_width = link_bbox[2] - link_bbox[0]
    link_x = (WIDTH - link_width) // 2
    link_y = hf_y + 40
    
    # Link background
    padding = 15
    draw.rectangle([link_x - padding, link_y - 8, link_x + link_width + padding, link_y + 35], 
                  fill=(30, 30, 30), outline=HIGHLIGHT_COLOR, width=2)
    draw.text((link_x, link_y), link_text, fill=HIGHLIGHT_COLOR, font=link_font)
    
    all_frames.append(final_img)
    all_frames.append(final_img)  # Hold final frame
    
    # Save GIF with different durations for different frame types
    print(f"Saving GIF with {len(all_frames)} frames...")
    
    # Create frames with appropriate durations
    durations = []
    frame_count = 0
    
    # Intro sequence (normal speed)
    durations.extend([FRAME_DURATION] * 3)
    frame_count += 3
    
    # Loading animation (a bit slower)
    durations.extend([LOADING_TRANSITION] * 3)
    frame_count += 3
    
    # Examples (previous better speed)
    for i, example in enumerate(demo_examples):
        # Large chart frames (normal speed)
        durations.extend([FRAME_DURATION] * 2)
        frame_count += 2
        
        # Full comparison frames (normal speed)
        durations.extend([FRAME_DURATION] * 3)
        frame_count += 3
        
        # Transition between examples (previous speed)
        if i < len(demo_examples) - 1:
            durations.extend([TRANSITION_DURATION] * 2)
            frame_count += 2
    
    # Final frame (normal speed)
    durations.extend([FRAME_DURATION] * 2)
    frame_count += 2
    
    # Ensure we have the right number of durations
    while len(durations) < len(all_frames):
        durations.append(FRAME_DURATION)
    
    all_frames[0].save(
        'demo.gif',
        save_all=True,
        append_images=all_frames[1:],
        duration=durations,
        loop=0
    )
    
    print("✅ Demo GIF created: demo.gif")

if __name__ == "__main__":
    main()
