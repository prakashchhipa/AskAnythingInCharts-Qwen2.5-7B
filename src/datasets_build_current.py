# src/datasets_build.py
import os
import random
from datasets import load_dataset, DatasetDict
from typing import Optional

def build_datasets(
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
    seed: int = 42
) -> DatasetDict:
    """
    Build training and evaluation datasets for ChartQA.
    This function loads the dataset and returns it for caching.
    
    Args:
        max_train: Maximum number of training samples (None for all)
        max_eval: Maximum number of evaluation samples (None for all)
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    print("Loading ChartQA dataset...")
    
    # Load from HuggingFace cache (this is what was working this morning)
    try:
        dataset = load_dataset("HuggingFaceM4/chart_qa")
        print("✓ Loaded ChartQA from HuggingFace cache")
    except Exception as e:
        print(f"Failed to load ChartQA dataset: {e}")
        raise RuntimeError("Could not load ChartQA dataset. Please check your internet connection and try again.")
    
    # Ensure we have the expected splits
    if 'train' not in dataset:
        raise ValueError("Dataset does not contain 'train' split")
    if 'test' not in dataset:
        raise ValueError("Dataset does not contain 'test' split")
    
    # Apply size limits if specified
    if max_train is not None and len(dataset['train']) > max_train:
        print(f"Limiting training set to {max_train} samples")
        dataset['train'] = dataset['train'].select(range(max_train))
    
    if max_eval is not None and len(dataset['test']) > max_eval:
        print(f"Limiting test set to {max_eval} samples")
        dataset['test'] = dataset['test'].select(range(max_eval))
    
    # Print dataset info
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    return dataset