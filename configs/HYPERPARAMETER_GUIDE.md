# Hyperparameter Guide for Multi-Epoch Fine-Tuning

## Overview

This guide explains the hyperparameter choices for different training durations and how to adjust them for your needs.

## Config Files Summary

| Config File | Epochs | Learning Rate | Use Case |
|------------|--------|---------------|----------|
| `sft_config_1epoch.json` | 1 | 2e-4 | Quick training, initial experiments |
| `sft_config.json` | 2 | 1e-4 | Standard training, good balance |
| **`sft_config_5epoch.json`** | **5** | **5e-5** | **Extended training, best quality** |

## Key Hyperparameters Explained

### 1. Learning Rate (`learning_rate`)

**What it does:** Controls how much the model weights change during each update.

**Recommendations:**
- **1 epoch:** `2e-4` (0.0002) - Aggressive learning for fast adaptation
- **2 epochs:** `1e-4` (0.0001) - Balanced learning rate
- **5 epochs:** `5e-5` (0.00005) - Lower to prevent overfitting over many epochs
- **10+ epochs:** `3e-5` (0.00003) - Very conservative for long training

**Rule of thumb:** Lower the learning rate as you increase epochs to prevent overfitting.

### 2. Learning Rate Scheduler (`lr_scheduler_type`)

**What it does:** Gradually adjusts learning rate during training.

**Options:**
- **`"linear"`** - Decreases linearly from max to 0
  - Good for: Short training (1-2 epochs)
  - Safe, predictable
  
- **`"cosine"`** - Decreases following a cosine curve (recommended for 5 epochs)
  - Good for: Medium to long training (3-10 epochs)
  - Smoother decay, often better results
  - **DEFAULT for 5 epochs**
  
- **`"constant"`** - Keeps LR fixed (not recommended for multi-epoch)
  - Good for: Very short training only

**Recommendation:** Use `"cosine"` for 5 epochs.

### 3. Warmup Ratio (`warmup_ratio`)

**What it does:** Gradually increases LR from 0 to `learning_rate` at the start.

**Recommendations:**
- **1-2 epochs:** `0.03` (3% of training) - ~25 steps warmup
- **5 epochs:** `0.05` (5% of training) - More gradual start
- **10+ epochs:** `0.1` (10% of training)

**Why it matters:** Prevents large weight updates early in training that can destabilize learning.

### 4. Weight Decay (`weight_decay`)

**What it does:** Adds L2 regularization to prevent overfitting.

**Recommendations:**
- **1 epoch:** `0.0` - Not needed for short training
- **2 epochs:** `0.0` or `0.005` - Optional
- **5 epochs:** `0.01` - Recommended to prevent overfitting
- **10+ epochs:** `0.05` or higher

**Rule:** Increase weight decay for longer training to maintain generalization.

### 5. Batch Size Settings

```json
"per_device_train_batch_size": 1,
"gradient_accumulation_steps": 8
```

**Effective batch size = per_device_batch_size × gradient_accumulation_steps × num_gpus**

For 8 GPUs: `1 × 8 × 8 = 64` effective batch size

**Guidelines:**
- Keep `per_device_train_batch_size = 1` for vision-language models (high memory usage)
- Adjust `gradient_accumulation_steps` based on your hardware:
  - 4-8 GPUs: Use `8` (recommended)
  - 2-4 GPUs: Use `16`
  - 1 GPU: Use `32-64` (slower but same results)

### 6. Gradient Clipping (`max_grad_norm`)

**What it does:** Clips gradients to prevent exploding gradients.

**Recommendation:** Keep at `0.3` for all configs (tested and stable).

### 7. Evaluation Strategy (`eval_strategy`, `eval_steps`)

```json
"eval_strategy": "steps",
"eval_steps": 500
```

**Options:**
- **"steps"** - Evaluate every N steps (recommended)
- **"epoch"** - Evaluate at end of each epoch (only for short epochs)
- **"no"** - No evaluation during training

**eval_steps recommendations:**
- Short datasets (< 1000 samples): `100-200`
- Medium datasets (1000-10000): `500` (recommended)
- Large datasets (> 10000): `1000-2000`

### 8. Checkpointing (`save_strategy`, `save_steps`)

```json
"save_strategy": "steps",
"save_steps": 500,
"save_total_limit": 3
```

**Key parameters:**
- **`save_steps`**: How often to save (same as `eval_steps` is common)
- **`save_total_limit`**: Keep only N best checkpoints (saves disk space)
- **`load_best_model_at_end`**: Load best checkpoint after training (recommended for 5+ epochs)

**Recommendations:**
- 1-2 epochs: Save every 500-1000 steps, no limit needed
- 5 epochs: Save every 500 steps, keep best 3
- 10+ epochs: Save every 1000 steps, keep best 5

### 9. Optimizer (`optim`)

**What it does:** Algorithm for updating model weights.

**Options:**
- **`"adamw_torch_fused"`** - Fastest, GPU-optimized (recommended)
- **`"adamw_torch"`** - Standard, widely compatible
- **`"adafactor"`** - Memory-efficient for very large models

**Recommendation:** Use `"adamw_torch_fused"` for best performance.

## Complete Configuration Examples

### For 5 Epochs (Recommended)

```json
{
  "num_train_epochs": 5,
  "learning_rate": 5e-5,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.05,
  "weight_decay": 0.01,
  "save_total_limit": 3,
  "load_best_model_at_end": true
}
```

**Best for:** High-quality training with good generalization

### For 10 Epochs (Advanced)

```json
{
  "num_train_epochs": 10,
  "learning_rate": 3e-5,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.1,
  "weight_decay": 0.05,
  "save_steps": 1000,
  "save_total_limit": 5,
  "load_best_model_at_end": true,
  "early_stopping_patience": 3
}
```

**Best for:** Maximum performance, large datasets

### For Fast Experiments (1 epoch)

```json
{
  "num_train_epochs": 1,
  "learning_rate": 2e-4,
  "warmup_ratio": 0.03,
  "weight_decay": 0.0,
  "save_steps": 500
}
```

**Best for:** Testing, debugging, proof-of-concept

## When to Use Each Config

### Use 1 Epoch When:
- ✅ Quick experiments or testing
- ✅ Limited compute budget
- ✅ Very large dataset (> 100k samples)
- ✅ Proof of concept

### Use 2 Epochs When:
- ✅ Standard training run
- ✅ Medium dataset (10k-50k samples)
- ✅ Balanced quality/time trade-off
- ✅ First serious training attempt

### Use 5 Epochs When:
- ✅ High-quality results needed
- ✅ Small to medium dataset (< 50k samples)
- ✅ Willing to invest compute time
- ✅ Production model training
- ✅ **Recommended for best results**

### Use 10+ Epochs When:
- ✅ Maximum performance required
- ✅ Very small dataset (< 5k samples)
- ✅ Research/competition
- ⚠️ Risk of overfitting - monitor validation loss!

## How to Monitor Training

### Good Training Signs ✅
- Loss decreases steadily
- Validation loss decreases (not increasing)
- Evaluation metrics improve
- No NaN/Inf in gradients

### Warning Signs ⚠️
- Validation loss increases while training loss decreases (overfitting)
- Loss plateaus early (learning rate too low)
- Loss oscillates wildly (learning rate too high)
- Gradients explode (need lower learning rate or smaller `max_grad_norm`)

### What to Check Each Epoch

```bash
# Check training logs
tail -f outputs/qwen2_5_vl_7b_sft_chart_text/logs/training_*.log

# Monitor loss trends
# Good: train_loss and eval_loss both decreasing
# Bad: train_loss decreasing but eval_loss increasing (overfitting!)
```

## Adjusting Hyperparameters Mid-Training

If you see issues during training:

**Loss too high / not decreasing:**
- ❌ Don't: Immediately increase learning rate (can make it worse)
- ✅ Do: Check if warmup is complete, wait a few hundred steps
- ✅ Do: If still high after 1000 steps, restart with 2x learning rate

**Overfitting (train loss << eval loss):**
- ✅ Add/increase weight decay (0.01 → 0.05)
- ✅ Reduce learning rate
- ✅ Stop training earlier (use fewer epochs)
- ✅ Add more training data

**Underfitting (both losses high):**
- ✅ Increase learning rate (5e-5 → 1e-4)
- ✅ Train for more epochs
- ✅ Increase LoRA rank (r=16 → r=32)
- ✅ Add more trainable modules (target_modules)

## Quick Reference Table

| Epochs | LR | Scheduler | Warmup | Weight Decay | Best For |
|--------|-----|-----------|--------|--------------|----------|
| 1 | 2e-4 | linear | 0.03 | 0.0 | Quick experiments |
| 2 | 1e-4 | linear | 0.03 | 0.0 | Standard training |
| **5** | **5e-5** | **cosine** | **0.05** | **0.01** | **Production (recommended)** |
| 10 | 3e-5 | cosine | 0.1 | 0.05 | Maximum quality |
| 20+ | 1e-5 | cosine | 0.1 | 0.1 | Research only |

## Running Training with Different Configs

```bash
# 1 epoch (fast)
deepspeed --num_gpus=8 src/train_vlm_sft.py \
  --sft_config configs/sft_config_1epoch.json \
  --output_dir outputs/qwen2_5_vl_7b_1ep

# 2 epochs (standard)
deepspeed --num_gpus=8 src/train_vlm_sft.py \
  --sft_config configs/sft_config.json \
  --output_dir outputs/qwen2_5_vl_7b_2ep

# 5 epochs (recommended for best results)
deepspeed --num_gpus=8 src/train_vlm_sft.py \
  --sft_config configs/sft_config_5epoch.json \
  --output_dir outputs/qwen2_5_vl_7b_5ep
```

## Advanced: Custom Learning Rate Schedules

For very long training, you might want custom schedules:

```json
{
  "lr_scheduler_type": "cosine_with_restarts",
  "lr_scheduler_kwargs": {
    "num_cycles": 2
  }
}
```

Or polynomial decay:
```json
{
  "lr_scheduler_type": "polynomial",
  "lr_scheduler_kwargs": {
    "power": 1.0
  }
}
```

## Summary

**For your 5-epoch training, use `configs/sft_config_5epoch.json` with:**
- ✅ Learning rate: `5e-5` (lower than 1-2 epochs to prevent overfitting)
- ✅ Scheduler: `cosine` (smooth decay)
- ✅ Warmup: `5%` (gradual start)
- ✅ Weight decay: `0.01` (regularization)
- ✅ Save best model: enabled
- ✅ Keep top 3 checkpoints: saves disk space

This configuration is well-tested and provides excellent results for extended training!

