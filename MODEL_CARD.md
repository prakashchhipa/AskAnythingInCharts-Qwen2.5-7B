---
tags:
- vision-language-model
- chart-understanding
- qwen2.5-vl
- lora
- fine-tuned
- chartqa
license: mit
author: Prakash Chandra Chhipa
---

# Qwen 2.5 VL 7B - ChartQA Fine-tuned (LoRA)

This is a fine-tuned version of the [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) model, specifically adapted for enhanced chart understanding and question answering on the [ChartQA dataset](https://huggingface.co/datasets/ChartQA).

**Author:** [Prakash Chandra Chhipa](https://github.com/prakashchhipa)  
**GitHub:** [AskAnythingInCharts-Qwen2.5-7B](https://github.com/prakashchhipa/AskAnythingInCharts-Qwen2.5-7B)  
**Portfolio:** [prakashchhipa.github.io](https://prakashchhipa.github.io)

## Model Description

The base model, Qwen 2.5 VL 7B, is a powerful vision-language model. This version has been further fine-tuned using LoRA (Low-Rank Adaptation) on the ChartQA benchmark to improve its ability to:
* Accurately extract information from various chart types (bar charts, line graphs, pie charts, scatter plots, etc.).
* Provide concise and pinpoint answers to questions about chart data.
* Reduce verbosity in responses, focusing on the direct answer.

## Performance

The fine-tuning process resulted in a notable improvement in accuracy on the ChartQA benchmark:

| Model | ChartQA Accuracy | Improvement |
|--------------------------|------------------|-------------|
| Qwen 2.5 7B (Base)       | **57.5%**        | -           |
| **Qwen 2.5 7B + LORA SFT** | **60.0%**        | **+2.5%**   |

## How to Use

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image

# Load base model
base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
)

# Load LoRA adapter
adapter_path = "prakashchhipa/Qwen2.5-VL-7B-ChartQA-LoRA"
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

processor = AutoProcessor.from_pretrained(base_model_id)

# Example usage
image = Image.open("chart.png")
question = "What is the highest value on the chart?"

# Process and generate
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image", "image": image}
    ]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# ... (process vision info and generate)
```

## Training Details

* **Base Model:** Qwen/Qwen2.5-VL-7B-Instruct
* **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
* **Dataset:** ChartQA
* **Training Software:** HuggingFace Transformers, PEFT
* **Author:** Prakash Chandra Chhipa
* **Repository:** [AskAnythingInCharts-Qwen2.5-7B](https://github.com/prakashchhipa/AskAnythingInCharts-Qwen2.5-7B)

## Citation

```bibtex
@misc{askanything-charts-qwen2.5,
  title={AskAnything in Charts - Powered by Qwen 2.5},
  author={Prakash Chandra Chhipa},
  year={2025},
  url={https://github.com/prakashchhipa/AskAnythingInCharts-Qwen2.5-7B}
}
```
