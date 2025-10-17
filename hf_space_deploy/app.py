"""
Gradio Demo: Chart Understanding with Fine-tuned Qwen2.5-VL-7B
Simplified version for HuggingFace Spaces
"""

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import json
from pathlib import Path

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "prakashchhipa/Qwen2.5-VL-7B-ChartQA-LoRA"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load example data
EXAMPLES_DIR = Path("demo_curated")
if EXAMPLES_DIR.exists():
    with open(EXAMPLES_DIR / "results.json") as f:
        EXAMPLE_DATA = json.load(f)
else:
    EXAMPLE_DATA = []


def build_model(base_model, adapter=None):
    """Build model with optional LoRA adapter (loads on-demand to save memory)"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto"
    )
    
    if adapter:
        print(f"Loading LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    
    processor_src = adapter if adapter else base_model
    processor = AutoProcessor.from_pretrained(processor_src)
    
    return model, processor


# Global variables for lazy loading
base_model_cache = None
base_processor_cache = None
finetuned_model_cache = None
finetuned_processor_cache = None

def get_base_model():
    """Lazy load base model"""
    global base_model_cache, base_processor_cache
    if base_model_cache is None:
        print("Loading base model...")
        base_model_cache, base_processor_cache = build_model(BASE_MODEL, adapter=None)
        print("✅ Base model loaded")
    return base_model_cache, base_processor_cache

def get_finetuned_model():
    """Lazy load fine-tuned model"""
    global finetuned_model_cache, finetuned_processor_cache
    if finetuned_model_cache is None:
        print("Loading fine-tuned model...")
        finetuned_model_cache, finetuned_processor_cache = build_model(BASE_MODEL, adapter=ADAPTER_PATH)
        print("✅ Fine-tuned model loaded")
    return finetuned_model_cache, finetuned_processor_cache

def clear_model_cache():
    """Clear cached models to free memory"""
    global base_model_cache, base_processor_cache, finetuned_model_cache, finetuned_processor_cache
    base_model_cache = None
    base_processor_cache = None
    finetuned_model_cache = None
    finetuned_processor_cache = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

print("✅ Demo ready! Models will load on first use.")


def infer(model, processor, image, question, max_new_tokens=64):
    """Run inference on a chart image with a question"""
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
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    messages_with_image = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image", "image": image}
        ]}
    ]
    
    image_inputs, _ = process_vision_info(messages_with_image)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return out.strip()


def predict(image, question):
    """Compare base and fine-tuned models (loads one at a time to save memory)"""
    if image is None:
        return "⚠️ Please upload an image", "⚠️ Please upload an image"
    
    if not question or question.strip() == "":
        return "⚠️ Please enter a question", "⚠️ Please enter a question"
    
    import gc
    
    try:
        # Step 1: Load and run base model
        clear_model_cache()  # Clear everything first
        base_model, base_processor = get_base_model()
        base_answer = infer(base_model, base_processor, image, question)
        
        # Step 2: Clear base, load and run fine-tuned
        clear_model_cache()
        finetuned_model, finetuned_processor = get_finetuned_model()
        finetuned_answer = infer(finetuned_model, finetuned_processor, image, question)
        
        return base_answer, finetuned_answer
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg


def load_example(example_idx):
    """Load a pre-selected example"""
    if 0 <= example_idx < len(EXAMPLE_DATA):
        example = EXAMPLE_DATA[example_idx]
        img_path = EXAMPLES_DIR / example["image_file"]
        
        if img_path.exists():
            return (
                str(img_path),
                example["question"],
                example["base_prediction"],
                example["trained_prediction"],
                f"**Ground Truth:** {example['ground_truth']}"
            )
    
    return None, "", "", "", ""


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Chart QA: Base vs Fine-tuned") as demo:
    gr.Markdown("""
    # AskAnything in Charts - Powered by Qwen 2.5
    
    An interactive demo showcasing a **fine-tuned vision-language model** for chart understanding.
    Finetuned adapted for pinpoint answer for question on ChartQA benchmark.
    Compare the base model with the fine-tuned version side-by-side!
    
    ### 🎯 Results
    - **Qwen 2.5 7B:** 57.5%
    - **Qwen 2.5 7B + LORA:** 60.0%
    - **Improvement:** +2.5%
    
    ### How to use:
    1. Upload a chart/graph image or select an example
    2. Ask a question about the chart
    3. Compare answers from both models side-by-side
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            image_input = gr.Image(type="pil", label="Upload Chart/Graph")
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g., What is the highest value in the chart?",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Compare Models", variant="primary", size="lg")
                clear_btn = gr.ClearButton(
                    [image_input, question_input], value="🗑️ Clear"
                )
        
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Model Responses")
            
            with gr.Row():
                base_output = gr.Textbox(
                    label="Qwen 2.5 7B",
                    lines=4,
                    interactive=False
                )
                finetuned_output = gr.Textbox(
                    label="Qwen 2.5 7B + LORA SFT",
                    lines=4,
                    interactive=False
                )
            
            ground_truth_output = gr.Markdown("", visible=False)
    
    # Examples section
    if EXAMPLE_DATA:
        gr.Markdown("### 🎨 Pre-loaded Examples")
        gr.Markdown("*Examples showing clear improvements: base model **wrong** → fine-tuned model **right**!*")
        
        with gr.Row():
            example_slider = gr.Slider(
                minimum=0,
                maximum=len(EXAMPLE_DATA) - 1,
                step=1,
                value=0,
                label=f"Select Example (1-{len(EXAMPLE_DATA)})",
                interactive=True
            )
            load_example_btn = gr.Button("📥 Load Example", size="sm")
        
        example_slider.change(
            fn=load_example,
            inputs=[example_slider],
            outputs=[image_input, question_input, base_output, finetuned_output, ground_truth_output]
        )
        
        load_example_btn.click(
            fn=load_example,
            inputs=[example_slider],
            outputs=[image_input, question_input, base_output, finetuned_output, ground_truth_output]
        )
    
    # Sample examples for quick start
    if EXAMPLE_DATA and len(EXAMPLE_DATA) >= 3:
        gr.Examples(
            examples=[
                ["demo_curated/example_0000.png", "Which region saw the highest proportion of accreditation over the given years?"],
                ["demo_curated/example_0001.png", "What's the median value of the green bars?"],
                ["demo_curated/example_0002.png", "Is the Very value in All voters more than Somewhat in All voters?"],
                ["scatter_temp_energy.png", "Which point does not follow correlation?"],
            ],
            inputs=[image_input, question_input],
            label="Quick Start Examples"
        )
    
    # Connect the submit button
    submit_btn.click(
        fn=predict,
        inputs=[image_input, question_input],
        outputs=[base_output, finetuned_output]
    )
    
    gr.Markdown("""
    ---
    ### 📝 Notes
    - **First query may be slow** as models load on-demand (memory optimization)
    - The model is optimized for **short, concise answers**
    - Works best with **bar charts, line graphs, and pie charts**
    - Training data: ChartQA dataset (chart understanding benchmark)
    - Base model: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
    - **Memory efficient:** Models are loaded sequentially to reduce GPU memory usage
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

