# src/train_vlm_sft.py
import argparse
import json
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor, set_seed
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature
from qwen_vl_utils import process_vision_info

from datasets_build import build_datasets

# Suppress excessive warnings from libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizers warning

# Setup logging
def setup_logging(output_dir, is_main_process=True):
    """Setup file and console logging with different levels"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    error_log_file = log_dir / f"errors_{timestamp}.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # File handler for all logs (DEBUG and above)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only (WARNING and above)
    error_handler = logging.FileHandler(error_log_file, mode='w')
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler - only show INFO and CRITICAL on main process
    console_handler = logging.StreamHandler(sys.stdout)
    if is_main_process:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.CRITICAL)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()  # Remove any existing handlers
    
    if is_main_process:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose loggers from libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('accelerate').setLevel(logging.WARNING)
    
    if is_main_process:
        logging.info(f"Logging initialized. Full log: {log_file}")
        logging.info(f"Error log: {error_log_file}")
    
    return logging.getLogger(__name__)

# helpers
def _strip_image_placeholders_from_text_parts(content_list):
    """Remove literal '<image>' tokens from any text parts."""
    new_content = []
    for part in content_list or []:
        if isinstance(part, dict):
            t = part.get("type")
            if t == "text":
                txt = (part.get("text") or "").replace("<image>", "").strip()
                if txt:
                    new_content.append({"type": "text", "text": txt})
            elif t == "image":
                # keep image marker (no pixel data here; pixels come from 'images' column)
                new_content.append({"type": "image"})
        # ignore unknown structures
    return new_content

def _count_image_entries_in_messages(msgs):
    c = 0
    for m in msgs or []:
        content = m.get("content")
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") == "image":
                    c += 1
    return c

def sanitize_example_for_qwen(ex):
    """
    Normalize one example so that:
    - ex['images'] is a list (possibly empty)
    - ex['messages'] contains EXACTLY len(images) image entries (as {'type':'image'})
    - no literal '<image>' remains inside text
    """
    imgs = ex.get("images")
    if imgs is None:
        imgs = []
    elif not isinstance(imgs, list):
        imgs = [imgs]
    ex["images"] = imgs

    msgs = ex.get("messages") or []
    new_msgs = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content")
        # convert string content -> list
        if isinstance(content, str):
            txt = content.replace("<image>", "").strip()
            new_content = [{"type": "text", "text": txt}] if txt else []
        elif isinstance(content, list):
            new_content = _strip_image_placeholders_from_text_parts(content)
        else:
            new_content = []
        new_msgs.append({"role": role, "content": new_content})

    # ensure image marker count equals number of images
    current = _count_image_entries_in_messages(new_msgs)
    need = len(imgs)
    if current != need:
        # remove all existing image markers
        for m in new_msgs:
            if isinstance(m.get("content"), list):
                m["content"] = [
                    p for p in m["content"]
                    if not (isinstance(p, dict) and p.get("type") == "image")
                ]
        # insert exactly `need` image markers at the start of the FIRST user message
        inserted = False
        for m in new_msgs:
            if m.get("role") == "user":
                if not isinstance(m.get("content"), list):
                    m["content"] = []
                m["content"] = ([{"type": "image"} for _ in range(need)] + m["content"])
                inserted = True
                break
        if not inserted:
            # no user message? create one
            new_msgs.insert(0, {"role": "user",
                                "content": [{"type": "image"} for _ in range(need)]})

    ex["messages"] = new_msgs
    return ex

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct", type=str)
    p.add_argument("--output_dir", default="outputs/qwen2_5_vl_7b_sft", type=str)
    p.add_argument("--sft_config", default="configs/sft_config.json", type=str)
    p.add_argument("--max_train", default=60000, type=int)
    p.add_argument("--max_eval", default=2000, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--lora_r", default=16, type=int)
    p.add_argument("--lora_alpha", default=32, type=int)
    p.add_argument("--lora_dropout", default=0.05, type=float)
    p.add_argument(
        "--target_modules",
        default="q_proj,v_proj",
        type=str,
        help="Comma-separated list, e.g., q_proj,v_proj (optionally add k_proj,o_proj)",
    )
    p.add_argument("--flash_attn", action="store_true")

    # NEW: cache controls for Option 2
    p.add_argument("--cache_dir", default="cache/sft_chartqa_textvqa", type=str,
                   help="Directory to save/load preprocessed HF DatasetDict")
    p.add_argument("--rebuild_cache", action="store_true",
                   help="If set, rank-0 will rebuild and overwrite the dataset cache")

    # Accept deepspeed/torch.distributed launcher's arg without using it
    p.add_argument("--local_rank", type=int, default=-1, help="Set by launcher")

    # Control checkpoint resume behavior (disabled by default to avoid torch.load CVE on torch<2.6)
    p.add_argument("--allow_resume", action="store_true",
                   help="If set and torch>=2.6, resume from latest checkpoint under output_dir before training")

    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    acc = Accelerator()
    is_main = acc.is_main_process
    
    # Setup logging first
    logger = setup_logging(args.output_dir, is_main_process=is_main)
    
    cache_dir = Path(args.cache_dir)

    # ---------------------
    # Rank-0 builds dataset ONCE, then everyone loads it
    # ---------------------
    if is_main:
        if args.rebuild_cache and cache_dir.exists():
            logger.info(f"--rebuild_cache set: removing existing cache at {cache_dir}")
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)

        if not cache_dir.exists():
            logger.info(f"Building datasets and saving to: {cache_dir}")
            data = build_datasets(max_train=args.max_train, max_eval=args.max_eval, seed=args.seed)

            # Accept both a plain dict and a DatasetDict from your builder
            if isinstance(data, DatasetDict):
                ds_dict = data
            else:
                ds_dict = DatasetDict({k: v for k, v in data.items() if k in ("train", "eval", "validation")})

            # --- SANITIZE: make messages/images counts consistent + strip "<image>" text ---
            np_ = max(1, min(8, (os.cpu_count() or 8) // 2))  # reasonable default
            for split in list(ds_dict.keys()):
                ds_dict[split] = ds_dict[split].map(
                    sanitize_example_for_qwen,  # <-- you already added this helper above
                    num_proc=np_,
                    desc=f"Sanitizing {split} for Qwen2.5-VL",
                )

            # --- OPTIONAL: post-sanitize drop of any remaining inconsistent rows (paranoid check) ---
            def _count_image_tokens(msgs):
                n = 0
                for m in msgs or []:
                    content = m.get("content", [])
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                n += 1
                return n

            def _consistent(ex):
                return _count_image_tokens(ex.get("messages")) == len(ex.get("images") or [])

            for split in list(ds_dict.keys()):
                before = ds_dict[split].num_rows
                bad = ds_dict[split].filter(lambda ex: not _consistent(ex)).num_rows
                if bad:
                    logger.warning(f"{bad}/{before} inconsistent rows in {split} after sanitize; dropping them")
                    ds_dict[split] = ds_dict[split].filter(_consistent, num_proc=np_)

            cache_dir.mkdir(parents=True, exist_ok=True)
            ds_dict.save_to_disk(str(cache_dir))
            logger.info(f"Saved preprocessed datasets to: {cache_dir}")
        else:
            logger.info(f"Using existing cached datasets at: {cache_dir}")


    acc.wait_for_everyone()

    ds = load_from_disk(str(cache_dir))
    # Support either 'eval' or 'validation' naming
    train_dataset = ds["train"]
    eval_key = "eval" if "eval" in ds else ("validation" if "validation" in ds else None)
    eval_dataset = ds[eval_key] if eval_key is not None else None
    if is_main:
        logger.info(f"Loaded datasets from cache: {cache_dir} | splits: {list(ds.keys())}")

    # Processor & model - use default settings from pretrained
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.flash_attn else "sdpa",
        low_cpu_mem_usage=True,
    )
    # Disable KV cache during training to reduce GPU memory
    try:
        model.config.use_cache = False
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False
    except Exception:
        pass

    # LoRA config
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Training config
    with open(args.sft_config, "r") as f:
        sft_dict = json.load(f)

    if sft_dict.get("packing", False):
        logger.warning("packing=True with images is unsupported; forcing packing=False")
    sft_dict["packing"] = False

    # Keep multimodal columns (e.g., 'images', 'messages') in the dataloader
    sft_dict["remove_unused_columns"] = False


    # Make JSON safe: CLI precedence for output_dir
    sft_dict.pop("output_dir", None)

    # Map legacy config keys to current TRL SFTConfig schema
    aliases = {"evaluation_strategy": "eval_strategy"}
    for old, new in aliases.items():
        if old in sft_dict and new not in sft_dict:
            sft_dict[new] = sft_dict.pop(old)

    training_args = SFTConfig(**sft_dict, output_dir=args.output_dir)
    # Ensure memory-friendly training defaults
    try:
        training_args.gradient_checkpointing = True
        training_args.predict_with_generate = False
    except Exception:
        pass

    # Log dataset info
    if is_main:
        logger.info(f"Training dataset size: {len(train_dataset):,} examples")
        if eval_dataset is not None:
            logger.info(f"Evaluation dataset size: {len(eval_dataset):,} examples")

    # Custom collator to handle multimodal data
    def collate_fn(examples):
        try:
            batch_size = len(examples)
            
            # Log batch processing only at debug level
            logger.debug(f"Processing batch of size: {batch_size}")
            logger.debug(f"First example has image: {examples[0]['images'] is not None}")
            
            # Process each example individually
            all_input_ids = []
            all_attention_masks = []
            all_pixel_values = []
            all_image_grid_thw = []
            
            for i in range(batch_size):
                # Get image and messages
                messages = examples[i]["messages"]
                image = examples[i]["images"][0] if examples[i]["images"] else None
                
                # Ensure image is RGB
                if image is not None and image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Build sanitized messages with at most one image token when a valid image exists
                from PIL import Image as PILImage
                has_valid_image = (
                    image is not None and isinstance(image, PILImage.Image) and hasattr(image, 'mode')
                )

                # Count existing image markers
                def count_image_markers(msgs):
                    n = 0
                    for m in msgs:
                        c = m.get("content", [])
                        if isinstance(c, list):
                            for part in c:
                                if isinstance(part, dict) and part.get("type") == "image":
                                    n += 1
                    return n

                num_markers = count_image_markers(messages)
                want_image = has_valid_image and num_markers > 0

                sanitized_messages = []
                image_added = False
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", [])
                    if isinstance(content, str):
                        sanitized_messages.append({"role": role, "content": content})
                        continue
                    if not isinstance(content, list):
                        continue
                    new_content = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        t = part.get("type")
                        if t == "text":
                            txt = (part.get("text") or "").strip()
                            if txt:
                                new_content.append({"type": "text", "text": txt})
                        elif t == "image":
                            if want_image and not image_added:
                                new_content.append({"type": "image"})
                                image_added = True
                            # else: drop extra/invalid image tokens
                    if new_content:
                        sanitized_messages.append({"role": role, "content": new_content})

                # Prepare inputs
                text = processor.apply_chat_template(
                    sanitized_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                images_arg = [image] if image_added else None
                # Encode without truncation/padding to avoid multimodal token mismatches
                inputs = processor(
                    text=text,
                    images=images_arg,
                    return_tensors="pt",
                )

                # Collect per-sample tensors (remove batch dim of size 1)
                all_input_ids.append(inputs["input_ids"][0])
                all_attention_masks.append(inputs["attention_mask"][0])

                if "pixel_values" in inputs:
                    all_pixel_values.append(inputs["pixel_values"])  # keep [1, C, H, W]
                if "image_grid_thw" in inputs:
                    all_image_grid_thw.append(inputs["image_grid_thw"])  # keep [1, 3]
            
            # Manual right-padding to the max sequence length in this batch
            pad_id = processor.tokenizer.pad_token_id
            max_len = max(t.size(0) for t in all_input_ids)

            def pad_1d(t: torch.Tensor, pad_value: int, to_len: int) -> torch.Tensor:
                if t.size(0) == to_len:
                    return t
                pad_size = to_len - t.size(0)
                return torch.cat([t, torch.full((pad_size,), pad_value, dtype=t.dtype)], dim=0)

            def pad_mask(t: torch.Tensor, to_len: int) -> torch.Tensor:
                if t.size(0) == to_len:
                    return t
                pad_size = to_len - t.size(0)
                return torch.cat([t, torch.zeros((pad_size,), dtype=t.dtype)], dim=0)

            input_ids = torch.stack([pad_1d(t, pad_id, max_len) for t in all_input_ids], dim=0)
            attention_mask = torch.stack([pad_mask(t, max_len) for t in all_attention_masks], dim=0)
            
            # Handle pixel_values (only if we have images)
            if all_pixel_values:
                pixel_values = torch.cat(all_pixel_values, dim=0)
            else:
                pixel_values = None
            
            # Handle image_grid_thw (only if we have images)
            if all_image_grid_thw:
                image_grid_thw = torch.cat(all_image_grid_thw, dim=0)
            else:
                image_grid_thw = None
            
            # Create labels (copy of input_ids with padding masked)
            labels = input_ids.clone()
            labels[labels == pad_id] = -100
            
            # Build batch dictionary
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            
            if pixel_values is not None:
                batch["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                batch["image_grid_thw"] = image_grid_thw
            
            return batch
        except Exception as e:
            logger.error("ERROR in collate_fn", exc_info=True)
            logger.error(f"Batch size: {batch_size if 'batch_size' in locals() else 'unknown'}")
            logger.error("Tensor shapes:")
            logger.error(f"- input_ids: {input_ids.shape if 'input_ids' in locals() else None}")
            logger.error(f"- attention_mask: {attention_mask.shape if 'attention_mask' in locals() else None}")
            logger.error(f"- pixel_values: {pixel_values.shape if 'pixel_values' in locals() else None}")
            logger.error(f"- labels: {labels.shape if 'labels' in locals() else None}")
            logger.error(f"- image_grid_thw: {image_grid_thw.shape if 'image_grid_thw' in locals() else None}")
            raise

    # Configure trainer with custom collator
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        data_collator=collate_fn,
    )

    try:
        # Resume only if explicitly allowed and torch version is safe (>=2.6)
        def _torch_geq(maj: int, minr: int) -> bool:
            v = str(torch.__version__).split("+")[0].split(".")
            try:
                M = int(v[0]); m = int(v[1])
            except Exception:
                return False
            return (M > maj) or (M == maj and m >= minr)

        do_resume = False
        latest_ckpt = None
        if args.allow_resume and _torch_geq(2, 6):
            try:
                ckpts = sorted([p for p in Path(args.output_dir).glob("checkpoint-*") if p.is_dir()],
                               key=lambda p: int(p.name.split("-")[-1]))
                if ckpts:
                    latest_ckpt = str(ckpts[-1])
                    do_resume = True
            except Exception:
                latest_ckpt = None

        if do_resume and latest_ckpt:
            logger.info(f"Resuming from latest checkpoint: {latest_ckpt}")
            trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            if args.allow_resume and not _torch_geq(2, 6):
                logger.warning("--allow_resume ignored because torch<2.6; proceeding without resume")
            trainer.train()
    except Exception as e:
        logger.critical("Training failed with exception", exc_info=True)
        raise
    
    # ====================================================================
    # CRITICAL FIX: Proper LORA adapter saving with DeepSpeed ZeRO-3
    # ====================================================================
    # Problem: DeepSpeed ZeRO-3 shards parameters across GPUs. Simply calling
    # save_pretrained() or get_state_dict() returns empty or zero-filled tensors
    # because the parameters need to be explicitly gathered from all ranks.
    #
    # Solution: Use DeepSpeed's GatheredParameters context manager to
    # materialize the full parameters before extracting and saving them.
    # ====================================================================
    
    acc.wait_for_everyone()  # Ensure all ranks finish training
    
    if is_main:
        logger.info("Saving LORA adapter...")
    
    try:
        # Check if DeepSpeed is being used
        is_deepspeed_zero3 = (
            hasattr(trainer.model, 'module') and 
            hasattr(trainer.model.module, 'zero_optimization_stage') and
            trainer.model.module.zero_optimization_stage() == 3
        )
        
        if not is_deepspeed_zero3:
            # Try alternative detection method
            is_deepspeed_zero3 = (
                hasattr(trainer.accelerator.state, 'deepspeed_plugin') and
                trainer.accelerator.state.deepspeed_plugin is not None and
                trainer.accelerator.state.deepspeed_plugin.zero_stage == 3
            )
        
        if is_main:
            logger.info(f"DeepSpeed ZeRO-3 detected: {is_deepspeed_zero3}")
        
        if is_deepspeed_zero3:
            # ===== DeepSpeed ZeRO-3 PATH =====
            # Must use DeepSpeed's parameter gathering mechanism
            
            if is_main:
                logger.info("Using DeepSpeed ZeRO-3 parameter gathering...")
            
            import deepspeed
            from peft.utils import get_peft_model_state_dict
            
            # Unwrap model from trainer/accelerator wrappers
            unwrapped = acc.unwrap_model(trainer.model)
            
            # Collect all LORA parameters (they're the only trainable ones)
            lora_params = [p for n, p in unwrapped.named_parameters() if 'lora_' in n and p.requires_grad]
            
            if is_main:
                lora_param_names = [n for n, p in unwrapped.named_parameters() if 'lora_' in n and p.requires_grad]
                logger.info(f"Found {len(lora_params)} trainable LORA parameters")
                if len(lora_param_names) > 0:
                    logger.info(f"Sample param names: {lora_param_names[:3]}")
            
            if len(lora_params) == 0:
                if is_main:
                    logger.error("❌ CRITICAL: No LORA parameters found! Check PEFT configuration.")
                raise ValueError("No LORA parameters found in model")
            
            # CRITICAL: Use GatheredParameters to materialize sharded params
            # This context manager gathers all shards from all GPUs and materializes
            # the full parameter tensors on the specified rank (modifier_rank=0)
            peft_state_dict = {}
            
            with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=0):
                if is_main:
                    # Now the parameters are fully materialized on rank 0
                    # Use PEFT's own state dict extraction to get correct key format
                    peft_state_dict = get_peft_model_state_dict(unwrapped)
                    
                    logger.info(f"Gathered {len(peft_state_dict)} parameters into state dict")
                    
                    # CRITICAL FIX: Transform keys for Qwen2.5-VL compatibility
                    # Qwen2.5-VL has 'language_model' in its architecture path, but
                    # PEFT expects keys without it when loading fresh. We must strip it out.
                    fixed_state_dict = {}
                    for key, value in peft_state_dict.items():
                        # Remove '.language_model.' from the path (Qwen2.5-VL specific)
                        new_key = key.replace('.language_model.', '.')
                        fixed_state_dict[new_key] = value
                    
                    keys_changed = len([k for k in peft_state_dict.keys() if '.language_model.' in k])
                    if keys_changed > 0:
                        logger.info(f"Fixed {keys_changed} keys by removing '.language_model.' from path")
                        logger.info(f"Example transformation:")
                        for old_key in list(peft_state_dict.keys())[:1]:
                            new_key = old_key.replace('.language_model.', '.')
                            logger.info(f"  OLD: {old_key}")
                            logger.info(f"  NEW: {new_key}")
                    
                    peft_state_dict = fixed_state_dict
                    
                    # Verify parameters have been trained (non-zero)
                    zero_count = 0
                    for k, v in peft_state_dict.items():
                        norm = v.float().norm().item()
                        if norm < 1e-8:
                            zero_count += 1
                    
                    logger.info(f"Parameter verification: {len(peft_state_dict) - zero_count}/{len(peft_state_dict)} have non-zero norms")
                    
                    if zero_count == len(peft_state_dict):
                        logger.error("❌ CRITICAL: ALL parameters have zero norm! Training may have failed.")
                    elif zero_count > 0:
                        logger.warning(f"⚠️  {zero_count} parameters have near-zero norms")
                    
                    # Log sample parameter stats
                    sample_keys = list(peft_state_dict.keys())[:5]
                    logger.info("Sample parameter statistics:")
                    for k in sample_keys:
                        v = peft_state_dict[k]
                        norm = v.float().norm().item()
                        logger.info(f"  {k}: shape={list(v.shape)}, norm={norm:.6f}")
                    
                    # CRITICAL FIX: Save INSIDE the GatheredParameters context
                    # If we save outside, the parameters become sharded again!
                    if len(peft_state_dict) == 0:
                        logger.error("❌ CRITICAL: State dict is EMPTY after gathering!")
                        raise ValueError("Failed to gather PEFT parameters")
                    
                    # Save state dict directly to safetensors file
                    # We must do this INSIDE the context while params are materialized
                    from safetensors.torch import save_file
                    adapter_path = Path(args.output_dir)
                    adapter_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save adapter weights
                    save_file(peft_state_dict, str(adapter_path / "adapter_model.safetensors"))
                    logger.info(f"Saved adapter weights to: {adapter_path / 'adapter_model.safetensors'}")
                    
                    # Save adapter config (do this after exiting context is fine)
                    unwrapped.peft_config['default'].save_pretrained(args.output_dir)
                    logger.info(f"Saved adapter config to: {args.output_dir}")
        
        else:
            # ===== Non-DeepSpeed or ZeRO-1/2 PATH =====
            # Standard save process works fine here
            
            if is_main:
                logger.info("Using standard PEFT save (no DeepSpeed ZeRO-3)...")
            
            from peft.utils import get_peft_model_state_dict
            
            unwrapped = acc.unwrap_model(trainer.model)
            
            # Get full state dict and extract PEFT params
            state_dict_full = acc.get_state_dict(unwrapped)
            peft_state_dict = get_peft_model_state_dict(unwrapped, state_dict=state_dict_full)
            
            if is_main:
                logger.info(f"Extracted {len(peft_state_dict)} PEFT parameters")
                unwrapped.save_pretrained(
                    args.output_dir,
                    state_dict=peft_state_dict,
                    safe_serialization=True
                )
                logger.info(f"Saved adapter to: {args.output_dir}")
        
        # Verify saved file on main process
        if is_main:
            try:
                from safetensors.torch import safe_open
                adapter_path = Path(args.output_dir) / "adapter_model.safetensors"
                
                if not adapter_path.exists():
                    logger.error(f"❌ CRITICAL: {adapter_path} was not created!")
                    raise FileNotFoundError(f"Adapter file not found: {adapter_path}")
                
                file_size = adapter_path.stat().st_size
                logger.info(f"Adapter file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                
                with safe_open(str(adapter_path), framework="pt") as f:
                    keys = list(f.keys())
                    total_params = sum(f.get_tensor(k).numel() for k in keys)
                
                logger.info(f"Adapter verification: {len(keys)} keys, {total_params:,} total parameters")
                
                if len(keys) == 0:
                    logger.error("❌ CRITICAL: Saved adapter has 0 keys!")
                    raise ValueError("Adapter save failed - empty state dict")
                
                # Check first few tensors for zero-norm
                sample_keys = keys[:min(5, len(keys))]
                with safe_open(str(adapter_path), framework="pt") as f:
                    for k in sample_keys:
                        tensor = f.get_tensor(k)
                        norm = tensor.float().norm().item()
                        logger.info(f"  {k}: shape={list(tensor.shape)}, norm={norm:.6f}")
                        if norm < 1e-8:
                            logger.warning(f"    ⚠️  Parameter has near-zero norm!")
                
                logger.info("✅ Adapter verification PASSED")
                
            except Exception as e:
                logger.error(f"❌ Adapter verification FAILED: {e}", exc_info=True)
                raise
                
    except Exception as e:
        logger.error(f"❌ Failed to save adapter: {e}", exc_info=True)
        if is_main:
            logger.error("Attempting fallback save method...")
            try:
                # Last resort: use trainer's save_model (may still fail with ZeRO-3)
                trainer.save_model(args.output_dir)
                logger.warning("Fallback save completed (may have issues with ZeRO-3)")
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}", exc_info=True)
                raise

    if is_main:
        processor.save_pretrained(args.output_dir)
        logger.info(f"✅ Training complete. Adapter & processor saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
