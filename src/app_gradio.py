# src/app_gradio.py
import gradio as gr
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from ocr_tool import ocr_easyocr

def build_model(base_model, adapter=None, device="cuda"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(base_model)
    return model, processor

def generate_response(image: Image.Image, question: str, use_ocr: bool, base_model: str, adapter: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(generate_response, "_model"):
        generate_response._model, generate_response._processor = build_model(base_model, adapter, device)
    model = generate_response._model
    processor = generate_response._processor

    # First pass
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=256)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out1 = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    if use_ocr:
        # Simple uncertainty heuristic -> call OCR
        if (len(out1) < 4) or ("cannot read" in out1.lower()) or ("unsure" in out1.lower()):
            ocr_text = ocr_easyocr(image)
            if len(ocr_text) > 8:
                question2 = (f"{question}\n\nOCR text (use only if helpful):\n"
                             f"-----\n{ocr_text}\n-----")
                messages2 = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question2},
                ]}]
                text2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
                img2, _ = process_vision_info(messages2)
                inputs2 = processor(text=[text2], images=img2, return_tensors="pt").to(device)
                with torch.no_grad():
                    out_ids2 = model.generate(**inputs2, max_new_tokens=256)
                trimmed2 = [o[len(i):] for i, o in zip(inputs2.input_ids, out_ids2)]
                out2 = processor.batch_decode(trimmed2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                return out2

    return out1

def ui():
    with gr.Blocks(title="OpenVisionAssistant") as demo:
        gr.Markdown("# OpenVisionAssistant\nUpload an image (chart, receipt, poster, screenshot) and ask a question.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image")
                question = gr.Textbox(label="Question", placeholder="e.g., What's the 2021 sales? or What is the invoice total?")
                use_ocr = gr.Checkbox(label="Enable OCR tool-use if needed", value=True)
                base_model = gr.Textbox(value="Qwen/Qwen2.5-VL-7B-Instruct", label="Base model id")
                adapter = gr.Textbox(value="outputs/qwen2_5_vl_7b_sft_chart_text", label="LoRA adapter path (or empty)")
                btn = gr.Button("Ask")
            with gr.Column():
                answer = gr.Textbox(label="Answer", lines=8)

        btn.click(fn=generate_response, inputs=[image, question, use_ocr, base_model, adapter], outputs=[answer])
    return demo

if __name__ == "__main__":
    app = ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
