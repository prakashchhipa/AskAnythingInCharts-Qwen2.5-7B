# Quick Start Guide

## 🚀 Demo (Fastest)

```bash
cd demo
pip install -r requirements.txt
python app.py
```

## 🔬 Training

```bash
cd src
python train_vlm_sft.py --config ../configs/your_config.yaml
```

## 📊 Evaluation

```bash
cd evaluations
python eval_chartqa.py --base_model Qwen/Qwen2.5-VL-7B-Instruct --adapter your_adapter_path
```

## 📈 Analysis

```bash
python find_improved_examples.py --adapter your_adapter_path
python create_curated_demo.py --indices 191 267 317 360 391 393 471 495
```

## 📦 Installation

```bash
pip install -e .
```

For more details, see the main README.md
