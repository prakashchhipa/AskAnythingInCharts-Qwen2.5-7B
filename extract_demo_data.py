#!/usr/bin/env python3
"""
Extract demo data from the HTML file
"""

import re
import json

def extract_demo_data():
    """Extract demo data from HTML file"""
    
    with open('demo/demo_curated/curated_demo.html', 'r') as f:
        content = f.read()
    
    # Find all examples
    examples = []
    
    # Pattern to match each example
    pattern = r'<div class="example">.*?<h3>Example (\d+).*?<img src="(example_\d+\.png)".*?<div class="question">.*?❓ Question:</span> (.*?)</div>.*?<div class="ground-truth">.*?✓ Ground Truth:</span> (.*?)</div>.*?<div class="base-pred">.*?❌ Base Model:</span> (.*?)</div>.*?<div class="trained-pred">.*?✅ Fine-tuned Model:</span> (.*?)</div>'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    for i, match in enumerate(matches):
        example_num, image_file, question, ground_truth, base_answer, trained_answer = match
        
        # Clean up the text
        question = question.strip()
        ground_truth = ground_truth.strip()
        base_answer = base_answer.strip()
        trained_answer = trained_answer.strip()
        
        examples.append({
            "id": i,
            "image_file": image_file,
            "question": question,
            "ground_truth": ground_truth,
            "base_answer": base_answer,
            "trained_answer": trained_answer
        })
    
    return examples

def main():
    examples = extract_demo_data()
    
    print(f"Extracted {len(examples)} examples:")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"  Image: {example['image_file']}")
        print(f"  Question: {example['question']}")
        print(f"  Ground Truth: {example['ground_truth']}")
        print(f"  Base Model: {example['base_answer']}")
        print(f"  Trained Model: {example['trained_answer']}")
    
    # Save to JSON
    with open('demo_data.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"\nSaved {len(examples)} examples to demo_data.json")

if __name__ == "__main__":
    main()
