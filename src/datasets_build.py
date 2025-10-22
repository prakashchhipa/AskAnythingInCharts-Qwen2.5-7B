# src/datasets_build.py
import random
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image

SYSTEM_MSG = (
    "You are a helpful Vision-Language assistant. "
    "Be concise and accurate. If the image contains small text, read it carefully. "
    "When answering chart questions, give the shortest correct answer."
)

def format_chartqa_item(sample) -> Dict[str, Any]:
    # sample keys: image (PIL), query (str), label (List[str])  (ChartQA spec)
    # We'll take the first label as the supervised answer.
    ans = sample["label"][0] if isinstance(sample["label"], list) else sample["label"]
    return {
        "images": [sample["image"]],  # PIL.Image
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["query"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": str(ans)}]},
        ],
    }

def format_textcaps_item(sample) -> Dict[str, Any]:
    # lmms-lab/TextCaps provides: 'image' PIL + question/answer? It is formatted for LMM eval;
    # We construct a generic instruction: "Read the text and caption or answer the question."
    # Fallback to captions if available.
    image = sample["image"] if "image" in sample else sample.get("image_path", None)
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    question = sample.get("question", None)
    answers = sample.get("answers", None) or sample.get("answer", None)
    # Compose prompt
    user_text = question if isinstance(question, str) else "Read the text in this image and provide a short caption."
    # Normalize answer
    gt = answers[0] if isinstance(answers, list) and len(answers) > 0 else (answers or "")
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": str(gt)}]},
        ],
    }

def format_textvqa_item(sample) -> Dict[str, Any]:
    # facebook/textvqa: fields include image (PIL), question (str), answers (list)
    image = sample["image"]
    question = sample.get("question", "What does the image say?")
    answers = sample.get("answers", [])
    gt = answers[0] if len(answers) > 0 else ""
    return {
        "images": [image],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": str(gt)}]},
        ],
    }

def build_datasets(max_train: Optional[int] = 60000,
                   max_eval: Optional[int] = 2000,
                   seed: int = 42) -> Dict[str, Dataset]:
    random.seed(seed)

    # 1) ChartQA
    chartqa = load_dataset("HuggingFaceM4/ChartQA")  # train/val/test splits
    chartqa_train = chartqa["train"].map(format_chartqa_item, remove_columns=chartqa["train"].column_names)
    chartqa_val   = chartqa["val"].map(format_chartqa_item, remove_columns=chartqa["val"].column_names)

    # 2) TextCaps (train split for SFT; use val/test later for evaluation)
    # If this fails due to mirrors, switch to lmms-lab/textvqa only.
    try:
        textcaps = load_dataset("lmms-lab/TextCaps")
        textcaps_train = textcaps["train"].map(format_textcaps_item, remove_columns=textcaps["train"].column_names)
        textcaps_val = textcaps.get("val", None)
        textcaps_val = textcaps_val.map(format_textcaps_item, remove_columns=textcaps_val.column_names) if textcaps_val else None
    except Exception:
        textcaps_train, textcaps_val = None, None

    # 3) TextVQA (train split for SFT)
    try:
        textvqa = load_dataset("facebook/textvqa")  # 'train', 'val'
        textvqa_train = textvqa["train"].map(format_textvqa_item, remove_columns=textvqa["train"].column_names)
        textvqa_val   = textvqa["val"].map(format_textvqa_item, remove_columns=textvqa["val"].column_names)
    except Exception:
        textvqa_train, textvqa_val = None, None

    # Merge training sets
    train_sets = [chartqa_train]
    if textcaps_train is not None: train_sets.append(textcaps_train)
    if textvqa_train is not None:  train_sets.append(textvqa_train)
    train_merged = concatenate_datasets(train_sets)

    # Merge eval sets
    eval_sets = [chartqa_val]
    if textcaps_val is not None: eval_sets.append(textcaps_val)
    if textvqa_val is not None:  eval_sets.append(textvqa_val)
    eval_merged = concatenate_datasets(eval_sets)

    # Shuffle and cap sizes (keep training time bounded)
    train_merged = train_merged.shuffle(seed=seed)
    eval_merged = eval_merged.shuffle(seed=seed)

    if max_train is not None and len(train_merged) > max_train:
        train_merged = train_merged.select(range(max_train))
    if max_eval is not None and len(eval_merged) > max_eval:
        eval_merged = eval_merged.select(range(max_eval))

    return {"train": train_merged, "eval": eval_merged}