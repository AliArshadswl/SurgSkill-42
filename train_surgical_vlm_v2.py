#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SURGICAL VLM TRAINING - SigLip2 + Qwen3 (Version 2.0)
================================================================================

An improved Vision-Language Model training script for surgical procedure 
understanding with:
- Proper train/validation/test splits (NO data leakage)
- Comprehensive ablation study support
- Detailed logging and dataset statistics
- Extensive documentation and comments

Architecture:
-------------
- Vision Encoder: SigLip2 (frozen) - extracts visual features from surgical frames
- Language Model: Qwen3-0.6B (LoRA fine-tuned) - generates text responses
- Visual Projector: MLP (fully trained) - bridges vision and language spaces

Ablation Parameters:
--------------------
- visual_tokens: Number of visual tokens (compression level)
- lora_r: LoRA rank (model capacity)
- lora_alpha: LoRA alpha (scaling factor)
- learning_rate: Optimizer learning rate
- num_frames: Number of video frames to use
- projector_type: Type of visual projector (mlp, linear, attention)

Data Split Strategy:
--------------------
- Training set: Used for gradient updates (backpropagation)
- Validation set: Used for model selection and early stopping
- Test set: Held out completely, used ONLY for final evaluation

Usage:
------
# Single run:
python train_surgical_vlm_v2.py \\
    --train_data /path/to/train.json \\
    --val_data /path/to/val.json \\
    --test_data /path/to/test.json \\
    --output_dir /path/to/checkpoints

# Ablation study:
python train_surgical_vlm_v2.py \\
    --run_ablation \\
    --ablation_param visual_tokens \\
    --ablation_values 16 32 64 128 256

Author: [Your Name]
Date: 2024
License: MIT
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
# Disable tokenizer parallelism to avoid deadlocks in DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
import time
import random
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import Counter, defaultdict
from datetime import datetime
import copy

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    SiglipImageProcessor,
    SiglipVisionModel,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(output_dir: Optional[Path] = None, log_level: int = logging.INFO):
    """
    Configure logging to both console and file.
    
    Args:
        output_dir: Directory to save log file (optional)
        log_level: Logging level (default: INFO)
    
    Returns:
        Logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if output_dir provided)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "training.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger (will be reconfigured with file handler later)
logger = setup_logging()


# ==============================================================================
# CONFIGURATION DATACLASS
# ==============================================================================

@dataclass
class TrainConfig:
    """
    Complete training configuration.
    
    This dataclass holds all hyperparameters and settings for training.
    Organized into logical groups for clarity.
    
    Attributes are documented inline with their purpose and typical values.
    """
    
    # ==========================================================================
    # DATA PATHS
    # ==========================================================================
    
    # Path to training data JSON file
    # Format: List of dicts with "frames", "conversations", "main_tag" keys
    train_data: str = "/mnt/share/ali/VLM_Project/hospital_data/train.json"
    
    # Path to validation data JSON file (used for model selection - NOT test!)
    # If not provided, will split from training data
    val_data: Optional[str] = None
    
    # Path to test data JSON file (held out - used ONLY for final evaluation)
    # This ensures NO data leakage during training
    test_data: str = "/mnt/share/ali/VLM_Project/hospital_data/test.json"
    
    # Output directory for checkpoints, logs, and results
    output_dir: str = "/mnt/share/ali/VLM_Project/checkpoints"
    
    # ==========================================================================
    # MODEL PATHS
    # ==========================================================================
    
    # Path to SigLip2 vision encoder
    # SigLip2 provides strong visual features with efficient processing
    vision_model_path: str = "/mnt/share/ali/VLMs/hf_cache/hub/siglip2-model/"
    
    # Path to Qwen3 language model
    # Qwen3-0.6B is a compact but capable LLM for generation
    language_model_path: str = "/mnt/share/ali/VLMs/hf_cache/hub/Qwen3-0.6B/"
    
    # ==========================================================================
    # TASK CONFIGURATION
    # ==========================================================================
    
    # List of tasks to train on (None = all tasks)
    # Available: step_classification, stage_classification, completion_status, full_description
    target_tasks: Optional[List[str]] = field(default_factory=lambda: [
        "step_classification",
        "stage_classification"
    ])
    
    # ==========================================================================
    # TRAINING HYPERPARAMETERS
    # ==========================================================================
    
    # Number of training epochs (full passes through data)
    epochs: int = 20
    
    # Batch size per GPU (actual batch = batch_size * grad_accum)
    batch_size: int = 2
    
    # Gradient accumulation steps (simulates larger batch size)
    # Effective batch size = batch_size * grad_accum = 2 * 8 = 16
    grad_accum: int = 8
    
    # Learning rate for AdamW optimizer
    # Typical range for LoRA: 1e-5 to 5e-4
    learning_rate: float = 2e-4
    
    # Weight decay for regularization (prevents overfitting)
    weight_decay: float = 0.01
    
    # Warmup ratio (fraction of total steps for LR warmup)
    warmup_ratio: float = 0.03
    
    # Maximum gradient norm for clipping (prevents exploding gradients)
    max_grad_norm: float = 1.0
    
    # ==========================================================================
    # SEQUENCE SETTINGS
    # ==========================================================================
    
    # Maximum sequence length for text (prompt + response)
    max_seq_len: int = 512
    
    # Number of video frames to use per sample
    # More frames = more context but more memory
    num_frames: int = 8
    
    # ==========================================================================
    # VISUAL TOKEN SETTINGS (ABLATION PARAMETER)
    # ==========================================================================
    
    # Number of visual tokens after projection
    # This controls the compression of visual information
    # Higher = more visual detail but longer sequences
    # Lower = more compression but faster processing
    # Ablation values: [16, 32, 64, 128, 256, 512]
    visual_tokens: int = 256
    
    # ==========================================================================
    # LORA SETTINGS (ABLATION PARAMETERS)
    # ==========================================================================
    
    # LoRA rank (dimensionality of low-rank matrices)
    # Higher = more capacity but more parameters
    # Ablation values: [8, 16, 32, 64, 128]
    lora_r: int = 32
    
    # LoRA alpha (scaling factor)
    # Typically set to 2x lora_r
    # The actual scaling is: lora_alpha / lora_r
    lora_alpha: int = 64
    
    # LoRA dropout (regularization)
    # Ablation values: [0.0, 0.05, 0.1, 0.15]
    lora_dropout: float = 0.05
    
    # ==========================================================================
    # PROJECTOR SETTINGS (ABLATION PARAMETER)
    # ==========================================================================
    
    # Type of visual projector
    # Options: "mlp", "linear", "attention"
    # - mlp: Two-layer MLP with GELU (default, good balance)
    # - linear: Single linear layer (faster, less capacity)
    # - attention: Cross-attention pooling (more complex, better for long sequences)
    projector_type: str = "mlp"
    
    # Hidden dimension multiplier for MLP projector
    # projector_hidden = llm_hidden * projector_hidden_mult
    projector_hidden_mult: float = 1.0
    
    # ==========================================================================
    # MODEL SETTINGS
    # ==========================================================================
    
    # Whether to freeze vision encoder (recommended: True)
    # Frozen = faster training, less memory, prevents catastrophic forgetting
    freeze_vision: bool = True
    
    # Use 4-bit quantization for LLM (saves memory)
    use_4bit: bool = True
    
    # Use bfloat16 precision (better numerical stability than float16)
    use_bf16: bool = True
    
    # ==========================================================================
    # LOGGING & CHECKPOINTING
    # ==========================================================================
    
    # Log metrics every N optimizer steps
    log_every_steps: int = 10
    
    # Run validation every N optimizer steps
    eval_every_steps: int = 500
    
    # Save checkpoint every N optimizer steps
    save_every_steps: int = 500
    
    # Number of samples to use for validation (for speed)
    # Set to -1 to use all validation samples
    eval_samples: int = 100
    
    # ==========================================================================
    # DATA SPLIT SETTINGS
    # ==========================================================================
    
    # Fraction of training data to use for validation (if val_data not provided)
    val_split_ratio: float = 0.1
    
    # Random seed for validation split (ensures reproducibility)
    val_split_seed: int = 42
    
    # ==========================================================================
    # HARDWARE SETTINGS
    # ==========================================================================
    
    # Number of DataLoader workers
    num_workers: int = 2
    
    # Random seed for reproducibility
    seed: int = 42
    
    # ==========================================================================
    # ABLATION STUDY SETTINGS
    # ==========================================================================
    
    # Whether to run ablation study
    run_ablation: bool = False
    
    # Parameter to ablate
    # Options: visual_tokens, lora_r, lora_alpha, learning_rate, num_frames, projector_type, lora_dropout
    ablation_param: Optional[str] = None
    
    # Values to try for ablation parameter
    ablation_values: Optional[List[Any]] = None
    
    # ==========================================================================
    # EARLY STOPPING SETTINGS
    # ==========================================================================
    
    # Enable early stopping based on validation loss
    early_stopping: bool = True
    
    # Number of evaluations without improvement before stopping
    early_stopping_patience: int = 5
    
    # Minimum improvement to count as "better"
    early_stopping_min_delta: float = 0.001


# ==============================================================================
# SPECIAL TOKENS
# ==============================================================================

# Task-specific tokens to help the model understand the task type
# These are added to the vocabulary and can be used in prompts
TASK_TOKENS = [
    "<STEP_CLASS>",      # Step classification task
    "<STAGE_CLASS>",     # Stage classification task
    "<COMPLETION>",      # Completion status task
    "<DESCRIPTION>",     # Full description task
    "<IMAGE>",           # Image placeholder token
    "<VIDEO>",           # Video placeholder token
]


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This ensures that:
    - Data shuffling is reproducible
    - Weight initialization is reproducible
    - Dropout is reproducible
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional settings for full reproducibility (may slow down training)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory info in GB
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - allocated, 2),
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ==============================================================================
# DATASET STATISTICS
# ==============================================================================

class DatasetAnalyzer:
    """
    Analyze and display comprehensive dataset statistics.
    
    This class provides detailed insights into the dataset composition
    to help identify potential issues like class imbalance.
    """
    
    def __init__(self, samples: List[Dict], name: str = "Dataset"):
        """
        Initialize analyzer with samples.
        
        Args:
            samples: List of sample dictionaries
            name: Name of the dataset for display
        """
        self.samples = samples
        self.name = name
        self.stats = self._compute_stats()
    
    def _compute_stats(self) -> Dict[str, Any]:
        """Compute comprehensive statistics."""
        stats = {
            "total_samples": len(self.samples),
            "task_distribution": Counter(),
            "frames_per_sample": [],
            "question_lengths": [],
            "answer_lengths": [],
            "unique_answers": set(),
            "missing_frames": 0,
            "empty_conversations": 0,
        }
        
        for sample in self.samples:
            # Task distribution
            task = sample.get("main_tag", "unknown")
            stats["task_distribution"][task] += 1
            
            # Frame statistics
            frames = sample.get("frames", [])
            stats["frames_per_sample"].append(len(frames))
            
            # Check for missing frames
            for frame_path in frames:
                frame_path = frame_path.replace("\\", "/")
                if not os.path.exists(frame_path):
                    stats["missing_frames"] += 1
                    break  # Count once per sample
            
            # Conversation statistics
            convs = sample.get("conversations", [])
            if len(convs) < 2:
                stats["empty_conversations"] += 1
            else:
                question = convs[0].get("value", "")
                answer = convs[1].get("value", "")
                stats["question_lengths"].append(len(question))
                stats["answer_lengths"].append(len(answer))
                stats["unique_answers"].add(answer.strip().lower())
        
        return stats
    
    def print_report(self, logger: logging.Logger) -> None:
        """Print formatted statistics report."""
        logger.info("=" * 70)
        logger.info(f"DATASET ANALYSIS: {self.name}")
        logger.info("=" * 70)
        
        # Basic counts
        logger.info(f"Total samples: {self.stats['total_samples']}")
        logger.info(f"Samples with missing frames: {self.stats['missing_frames']}")
        logger.info(f"Samples with empty conversations: {self.stats['empty_conversations']}")
        
        # Task distribution
        logger.info("\nTask Distribution:")
        for task, count in sorted(self.stats['task_distribution'].items()):
            pct = 100 * count / self.stats['total_samples']
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            logger.info(f"  {task:25s}: {count:6d} ({pct:5.1f}%) {bar}")
        
        # Frame statistics
        if self.stats['frames_per_sample']:
            frames = self.stats['frames_per_sample']
            logger.info(f"\nFrames per sample:")
            logger.info(f"  Min: {min(frames)}, Max: {max(frames)}, "
                       f"Mean: {np.mean(frames):.1f}, Median: {np.median(frames):.1f}")
        
        # Text statistics
        if self.stats['question_lengths']:
            q_lens = self.stats['question_lengths']
            a_lens = self.stats['answer_lengths']
            logger.info(f"\nQuestion length (chars):")
            logger.info(f"  Min: {min(q_lens)}, Max: {max(q_lens)}, Mean: {np.mean(q_lens):.1f}")
            logger.info(f"\nAnswer length (chars):")
            logger.info(f"  Min: {min(a_lens)}, Max: {max(a_lens)}, Mean: {np.mean(a_lens):.1f}")
            logger.info(f"\nUnique answers: {len(self.stats['unique_answers'])}")
        
        logger.info("=" * 70)
    
    def get_class_weights(self) -> Dict[str, float]:
        """
        Compute class weights for balanced training.
        
        Returns:
            Dictionary mapping task names to weights
        """
        total = self.stats['total_samples']
        n_classes = len(self.stats['task_distribution'])
        
        weights = {}
        for task, count in self.stats['task_distribution'].items():
            # Inverse frequency weighting
            weights[task] = total / (n_classes * count)
        
        return weights


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class SurgicalVLMDataset(Dataset):
    """
    PyTorch Dataset for surgical VLM training.
    
    This dataset:
    - Loads video frames from paths
    - Processes images for the vision encoder
    - Handles missing frames gracefully
    - Supports task filtering
    
    Attributes:
        samples: List of sample dictionaries
        image_processor: SigLip image processor
        num_frames: Number of frames to use per sample
    """
    
    def __init__(
        self,
        samples: List[Dict],
        image_processor: SiglipImageProcessor,
        num_frames: int = 8,
        augment: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of sample dictionaries (pre-filtered)
            image_processor: Hugging Face image processor for SigLip
            num_frames: Number of frames to sample per video
            augment: Whether to apply data augmentation (not implemented yet)
        """
        self.samples = samples
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.augment = augment
        
        # Validate samples have required fields
        self._validate_samples()
    
    def _validate_samples(self) -> None:
        """Validate that samples have required fields."""
        valid_samples = []
        invalid_count = 0
        
        for sample in self.samples:
            # Check for required fields
            has_frames = "frames" in sample and len(sample["frames"]) > 0
            has_convs = "conversations" in sample and len(sample["conversations"]) >= 2
            
            if has_frames and has_convs:
                valid_samples.append(sample)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid samples (missing frames or conversations)")
        
        self.samples = valid_samples
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def _load_frames(self, frame_paths: List[str]) -> List[Image.Image]:
        """
        Load frames from file paths.
        
        Handles:
        - Path normalization (Windows -> Unix)
        - Missing files (creates gray placeholder)
        - Padding to num_frames
        
        Args:
            frame_paths: List of frame file paths
        
        Returns:
            List of PIL Images (exactly num_frames images)
        """
        images = []
        
        # Load available frames (up to num_frames)
        for path in frame_paths[:self.num_frames]:
            # Normalize path separators
            path = path.replace("\\", "/")
            
            try:
                if os.path.exists(path):
                    # Load and convert to RGB
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                else:
                    # Create gray placeholder for missing frames
                    logger.debug(f"Missing frame: {path}")
                    images.append(Image.new("RGB", (384, 384), color=(128, 128, 128)))
            except Exception as e:
                logger.debug(f"Error loading frame {path}: {e}")
                images.append(Image.new("RGB", (384, 384), color=(128, 128, 128)))
        
        # Pad with last frame if needed (or gray if no frames loaded)
        while len(images) < self.num_frames:
            if images:
                # Repeat last frame (common practice for video padding)
                images.append(images[-1].copy())
            else:
                # No frames at all - create placeholder
                images.append(Image.new("RGB", (384, 384), color=(128, 128, 128)))
        
        return images[:self.num_frames]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with:
                - pixel_values: Processed frames tensor [N, C, H, W]
                - question: Question string
                - answer: Answer string
                - main_tag: Task type
                - sample_id: Unique sample identifier
        """
        sample = self.samples[idx]
        
        # Load and process frames
        frames = self._load_frames(sample.get("frames", []))
        
        # Process through image processor
        processed = self.image_processor(images=frames, return_tensors="pt")
        pixel_values = processed["pixel_values"]  # Shape: [N, C, H, W]
        
        # Extract conversation
        conversations = sample.get("conversations", [])
        question = conversations[0]["value"] if conversations else ""
        answer = conversations[1]["value"] if len(conversations) > 1 else ""
        
        return {
            "pixel_values": pixel_values,
            "question": question,
            "answer": answer,
            "main_tag": sample.get("main_tag", "unknown"),
            "sample_id": sample.get("id", f"sample_{idx}"),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Stacks pixel values and collects other fields into lists.
    
    Args:
        batch: List of sample dictionaries from __getitem__
    
    Returns:
        Batched dictionary with stacked tensors and lists
    """
    # Stack pixel values: [B, N, C, H, W]
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    
    return {
        "pixel_values": pixel_values,
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "main_tags": [b["main_tag"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
    }


# ==============================================================================
# DATA LOADING AND SPLITTING
# ==============================================================================

def load_and_split_data(config: TrainConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load data and create train/validation/test splits.
    
    IMPORTANT: This function ensures NO data leakage by:
    1. Using separate files for train and test
    2. Creating validation split from training data (not test!)
    3. Using fixed seed for reproducible splits
    
    Args:
        config: Training configuration
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    logger.info("=" * 70)
    logger.info("LOADING AND SPLITTING DATA")
    logger.info("=" * 70)
    
    # =========================================================================
    # Load training data
    # =========================================================================
    logger.info(f"Loading training data from: {config.train_data}")
    with open(config.train_data, "r", encoding="utf-8") as f:
        all_train_samples = json.load(f)
    logger.info(f"Loaded {len(all_train_samples)} total training samples")
    
    # Filter by target tasks if specified
    if config.target_tasks:
        train_samples = [s for s in all_train_samples if s.get("main_tag") in config.target_tasks]
        logger.info(f"Filtered to {len(train_samples)} samples for tasks: {config.target_tasks}")
    else:
        train_samples = all_train_samples
    
    # =========================================================================
    # Create validation split from training data (NOT from test!)
    # =========================================================================
    if config.val_data and os.path.exists(config.val_data):
        # Use provided validation file
        logger.info(f"Loading validation data from: {config.val_data}")
        with open(config.val_data, "r", encoding="utf-8") as f:
            all_val_samples = json.load(f)
        
        if config.target_tasks:
            val_samples = [s for s in all_val_samples if s.get("main_tag") in config.target_tasks]
        else:
            val_samples = all_val_samples
        
        logger.info(f"Loaded {len(val_samples)} validation samples")
    else:
        # Split from training data
        logger.info(f"Creating validation split from training data (ratio: {config.val_split_ratio})")
        
        # Use fixed seed for reproducible split
        rng = random.Random(config.val_split_seed)
        
        # Shuffle indices
        indices = list(range(len(train_samples)))
        rng.shuffle(indices)
        
        # Calculate split point
        val_size = int(len(train_samples) * config.val_split_ratio)
        
        # Split (stratified by task if possible)
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])
        
        val_samples = [train_samples[i] for i in val_indices]
        train_samples = [train_samples[i] for i in train_indices]
        
        logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} validation")
    
    # =========================================================================
    # Load test data (completely separate - NO leakage)
    # =========================================================================
    test_samples = []
    if config.test_data and os.path.exists(config.test_data):
        logger.info(f"Loading test data from: {config.test_data}")
        with open(config.test_data, "r", encoding="utf-8") as f:
            all_test_samples = json.load(f)
        
        if config.target_tasks:
            test_samples = [s for s in all_test_samples if s.get("main_tag") in config.target_tasks]
        else:
            test_samples = all_test_samples
        
        logger.info(f"Loaded {len(test_samples)} test samples")
    else:
        logger.warning("No test data provided - final evaluation will be skipped")
    
    # =========================================================================
    # Verify no overlap (sanity check for data leakage)
    # =========================================================================
    train_ids = {s.get("id", str(i)) for i, s in enumerate(train_samples)}
    val_ids = {s.get("id", str(i)) for i, s in enumerate(val_samples)}
    test_ids = {s.get("id", str(i)) for i, s in enumerate(test_samples)}
    
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    if train_val_overlap:
        logger.error(f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} samples in both train and val!")
    if train_test_overlap:
        logger.error(f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} samples in both train and test!")
    if val_test_overlap:
        logger.error(f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} samples in both val and test!")
    
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        logger.info("✓ No data leakage detected - splits are clean")
    
    # =========================================================================
    # Print dataset statistics
    # =========================================================================
    DatasetAnalyzer(train_samples, "TRAINING SET").print_report(logger)
    DatasetAnalyzer(val_samples, "VALIDATION SET").print_report(logger)
    if test_samples:
        DatasetAnalyzer(test_samples, "TEST SET").print_report(logger)
    
    return train_samples, val_samples, test_samples


# ==============================================================================
# TEXT PROCESSING
# ==============================================================================

def build_batch_text(
    tokenizer,
    questions: List[str],
    answers: List[str],
    max_seq_len: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build tokenized batch with proper label masking.
    
    This function:
    1. Tokenizes questions and answers separately
    2. Concatenates them with separator
    3. Masks question tokens in labels (set to -100)
    4. Pads to uniform length
    
    The label masking is CRITICAL - it ensures the model only learns
    to predict answers, not to memorize questions.
    
    Args:
        tokenizer: Hugging Face tokenizer
        questions: List of question strings
        answers: List of answer strings
        max_seq_len: Maximum sequence length
        device: Target device for tensors
    
    Returns:
        Tuple of (input_ids, attention_mask, labels) tensors
    """
    input_ids_list = []
    labels_list = []
    
    for question, answer in zip(questions, answers):
        # Format prompt with clear separator
        prompt = f"{question}\nAnswer:"
        answer_text = f" {answer}{tokenizer.eos_token}"
        
        # Tokenize separately to know where to mask
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        
        # Concatenate and truncate
        full_ids = (prompt_ids + answer_ids)[:max_seq_len]
        
        # Create labels: -100 for prompt tokens (masked), actual ids for answer
        # The -100 index is ignored by CrossEntropyLoss
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_len]
        
        input_ids_list.append(full_ids)
        labels_list.append(labels)
    
    # Pad to maximum length in batch
    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    padded_ids = []
    padded_labels = []
    attention_masks = []
    
    for ids, labels in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        
        # Pad with pad_token_id for inputs
        padded_ids.append(ids + [pad_id] * pad_len)
        
        # Pad with -100 for labels (ignored in loss)
        padded_labels.append(labels + [-100] * pad_len)
        
        # Attention mask: 1 for real tokens, 0 for padding
        attention_masks.append([1] * len(ids) + [0] * pad_len)
    
    return (
        torch.tensor(padded_ids, dtype=torch.long, device=device),
        torch.tensor(attention_masks, dtype=torch.long, device=device),
        torch.tensor(padded_labels, dtype=torch.long, device=device),
    )


# ==============================================================================
# VISUAL PROJECTOR VARIANTS (FOR ABLATION)
# ==============================================================================

class MLPProjector(nn.Module):
    """
    Two-layer MLP projector with GELU activation.
    
    This is the default projector that provides a good balance between
    capacity and efficiency. The MLP learns to transform visual features
    into the language model's embedding space.
    
    Architecture:
        Linear(vision_hidden, llm_hidden) -> GELU -> Linear(llm_hidden, llm_hidden)
    """
    
    def __init__(self, vision_hidden: int, llm_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearProjector(nn.Module):
    """
    Simple linear projection.
    
    Fastest option with minimal parameters. May underfit for complex
    visual-language alignment tasks.
    """
    
    def __init__(self, vision_hidden: int, llm_hidden: int):
        super().__init__()
        self.linear = nn.Linear(vision_hidden, llm_hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AttentionProjector(nn.Module):
    """
    Cross-attention based projector with learnable queries.
    
    Uses learnable query tokens that attend to visual features.
    More expressive than MLP but slower and more memory-intensive.
    
    This is similar to the Q-Former in BLIP-2.
    """
    
    def __init__(self, vision_hidden: int, llm_hidden: int, num_queries: int = 64):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_hidden) * 0.02)
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_hidden,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        # Project visual features to LLM dimension
        self.visual_proj = nn.Linear(vision_hidden, llm_hidden)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(llm_hidden),
            nn.Linear(llm_hidden, llm_hidden),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual features [B, N_patches, vision_hidden]
        
        Returns:
            Projected features [B, num_queries, llm_hidden]
        """
        B = x.shape[0]
        
        # Project visual features
        visual_tokens = self.visual_proj(x)  # [B, N, llm_hidden]
        
        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # [B, Q, llm_hidden]
        
        # Cross-attention: queries attend to visual tokens
        attended, _ = self.cross_attn(
            query=queries,
            key=visual_tokens,
            value=visual_tokens,
        )  # [B, Q, llm_hidden]
        
        # Output projection
        output = self.output_proj(attended)
        
        return output


class VisualProjector(nn.Module):
    """
    Visual projector that maps vision encoder outputs to LLM embedding space.
    
    This module:
    1. Takes visual features from multiple frames
    2. Optionally pools them using different strategies
    3. Projects to a fixed number of visual tokens
    4. Adds positional embeddings
    
    The number of visual tokens controls the compression level:
    - More tokens = more detail, longer sequences
    - Fewer tokens = more compression, faster processing
    """
    
    def __init__(
        self,
        vision_hidden: int,
        llm_hidden: int,
        num_visual_tokens: int = 256,
        projector_type: str = "mlp",
    ):
        """
        Initialize projector.
        
        Args:
            vision_hidden: Vision encoder hidden dimension
            llm_hidden: Language model hidden dimension
            num_visual_tokens: Number of output visual tokens
            projector_type: Type of projector ("mlp", "linear", "attention")
        """
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.projector_type = projector_type
        
        # Select projector based on type
        if projector_type == "mlp":
            self.projector = MLPProjector(vision_hidden, llm_hidden)
        elif projector_type == "linear":
            self.projector = LinearProjector(vision_hidden, llm_hidden)
        elif projector_type == "attention":
            self.projector = AttentionProjector(vision_hidden, llm_hidden, num_visual_tokens)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
        
        # Positional embeddings for visual tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_visual_tokens, llm_hidden))
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project visual features to LLM space.
        
        Args:
            vision_features: [B, N_frames, N_patches, V_hidden]
        
        Returns:
            visual_embeds: [B, num_visual_tokens, llm_hidden]
        """
        B, N, P, V = vision_features.shape
        
        # Flatten frames: [B, N*P, V]
        flat_features = vision_features.view(B, N * P, V)
        
        # Project features
        if self.projector_type == "attention":
            # Attention projector outputs fixed number of tokens directly
            result = self.projector(flat_features)
        else:
            # MLP/Linear projector: project then interpolate
            projected = self.projector(flat_features)  # [B, N*P, H]
            
            # Interpolate to fixed number of tokens
            # Transpose for interpolation: [B, H, N*P]
            projected_t = projected.transpose(1, 2)
            
            # Linear interpolation to target size
            pooled = F.interpolate(
                projected_t,
                size=self.num_visual_tokens,
                mode='linear',
                align_corners=True
            )
            
            # Transpose back: [B, K, H]
            result = pooled.transpose(1, 2)
        
        # Add positional embeddings
        result = result + self.pos_embed.to(result.device)
        
        return result


# ==============================================================================
# VLM MODEL
# ==============================================================================

class SurgicalVLM(nn.Module):
    """
    Vision-Language Model for surgical procedure understanding.
    
    Architecture:
    1. Vision Encoder (SigLip2): Extracts visual features from video frames
    2. Visual Projector: Maps visual features to LLM embedding space
    3. Language Model (Qwen3): Generates text responses
    
    During forward pass:
    1. Encode all video frames with vision encoder
    2. Project visual features to LLM space
    3. Concatenate [visual_tokens, text_tokens]
    4. Forward through LLM with causal attention
    5. Compute loss only on answer tokens
    """
    
    def __init__(
        self,
        vision_encoder: SiglipVisionModel,
        image_processor: SiglipImageProcessor,
        llm: nn.Module,
        config: TrainConfig,
    ):
        """
        Initialize VLM.
        
        Args:
            vision_encoder: Pretrained SigLip vision model
            image_processor: SigLip image processor
            llm: Language model (with LoRA applied)
            config: Training configuration
        """
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.image_processor = image_processor
        self.llm = llm
        self.config = config
        
        # Get hidden dimensions
        self.vision_hidden = vision_encoder.config.hidden_size
        self.llm_hidden = llm.config.hidden_size
        
        logger.info(f"Vision hidden dim: {self.vision_hidden}")
        logger.info(f"LLM hidden dim: {self.llm_hidden}")
        logger.info(f"Visual tokens: {config.visual_tokens}")
        logger.info(f"Projector type: {config.projector_type}")
        
        # Initialize visual projector
        self.visual_projector = VisualProjector(
            vision_hidden=self.vision_hidden,
            llm_hidden=self.llm_hidden,
            num_visual_tokens=config.visual_tokens,
            projector_type=config.projector_type,
        )
        
        # Freeze vision encoder if configured
        if config.freeze_vision:
            logger.info("Freezing vision encoder")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
    
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames through vision encoder.
        
        Args:
            pixel_values: [B, N_frames, C, H, W] - batch of video frames
        
        Returns:
            vision_features: [B, N_frames, N_patches, V_hidden]
        """
        B, N, C, H, W = pixel_values.shape
        
        # Flatten batch and frames for efficient processing
        flat_pixels = pixel_values.view(B * N, C, H, W)
        
        # Encode (no gradients if frozen)
        context = torch.no_grad() if self.config.freeze_vision else torch.enable_grad()
        with context:
            outputs = self.vision_encoder(pixel_values=flat_pixels)
            features = outputs.last_hidden_state  # [B*N, P, V]
        
        # Reshape back to separate frames
        _, P, V = features.shape
        return features.view(B, N, P, V)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            pixel_values: [B, N, C, H, W] - video frames
            input_ids: [B, T] - tokenized text
            attention_mask: [B, T] - attention mask for text
            labels: [B, T] - labels for loss computation
        
        Returns:
            loss: Scalar loss tensor
        """
        device = input_ids.device
        
        # Step 1: Encode images
        vision_features = self.encode_images(pixel_values)
        
        # Step 2: Project to LLM space
        visual_embeds = self.visual_projector(vision_features)
        visual_embeds = visual_embeds.to(self.llm.dtype)
        
        # Step 3: Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Step 4: Concatenate [visual; text]
        full_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # Step 5: Build full attention mask
        B, K, _ = visual_embeds.shape
        visual_attn = torch.ones(B, K, dtype=attention_mask.dtype, device=device)
        full_attn = torch.cat([visual_attn, attention_mask], dim=1)
        
        # Step 6: Build full labels (mask visual tokens)
        visual_labels = torch.full((B, K), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([visual_labels, labels], dim=1)
        
        # Step 7: Forward through LLM
        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            labels=full_labels,
            use_cache=False,  # Disable KV cache during training
            return_dict=True,
        )
        
        return outputs.loss
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate response for inference.
        
        Args:
            pixel_values: [B, N, C, H, W] - video frames
            input_ids: [B, T] - tokenized prompt
            attention_mask: [B, T] - attention mask
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            output_ids: Generated token ids
        """
        self.eval()
        device = input_ids.device
        
        # Encode and project visual features
        vision_features = self.encode_images(pixel_values)
        visual_embeds = self.visual_projector(vision_features)
        visual_embeds = visual_embeds.to(self.llm.dtype)
        
        # Get text embeddings and concatenate
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        full_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # Build attention mask
        B, K, _ = visual_embeds.shape
        visual_attn = torch.ones(B, K, dtype=attention_mask.dtype, device=device)
        full_attn = torch.cat([visual_attn, attention_mask], dim=1)
        
        # Generate
        return self.llm.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.llm.config.pad_token_id,
            **kwargs
        )


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_models(config: TrainConfig, device: torch.device):
    """
    Load and prepare all model components.
    
    This function:
    1. Loads tokenizer and adds special tokens
    2. Loads LLM with optional 4-bit quantization
    3. Applies LoRA for efficient fine-tuning
    4. Loads vision encoder
    
    Args:
        config: Training configuration
        device: Target device
    
    Returns:
        Tuple of (tokenizer, llm, image_processor, vision_encoder, compute_dtype)
    """
    logger.info("=" * 70)
    logger.info("LOADING MODELS")
    logger.info("=" * 70)
    
    # =========================================================================
    # Load Tokenizer
    # =========================================================================
    logger.info(f"Loading tokenizer: {config.language_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.language_model_path,
        trust_remote_code=True
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Right padding for training (left padding better for generation)
    tokenizer.padding_side = "right"
    
    # Add task-specific special tokens
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": TASK_TOKENS})
    logger.info(f"Added {num_added} special tokens, vocab size: {len(tokenizer)}")
    
    # =========================================================================
    # Determine compute dtype
    # =========================================================================
    if config.use_bf16 and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        logger.info("Using bfloat16 precision")
    else:
        compute_dtype = torch.float16
        logger.info("Using float16 precision")
    
    # =========================================================================
    # Load Language Model
    # =========================================================================
    if config.use_4bit:
        logger.info("Loading LLM with 4-bit quantization...")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,  # Nested quantization for more memory savings
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - optimal for neural networks
        )
        
        llm = AutoModelForCausalLM.from_pretrained(
            config.language_model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map=None,  # Let us handle device placement
        )
        
        # Prepare model for k-bit training (handles gradient checkpointing setup)
        llm = prepare_model_for_kbit_training(llm)
    else:
        logger.info("Loading LLM in full precision...")
        llm = AutoModelForCausalLM.from_pretrained(
            config.language_model_path,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
    
    # Resize token embeddings for new special tokens
    llm.resize_token_embeddings(len(tokenizer))
    
    # =========================================================================
    # Apply LoRA
    # =========================================================================
    logger.info(f"Applying LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Target all attention and MLP projections
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
    )
    
    llm = get_peft_model(llm, lora_config)
    llm.config.use_cache = False  # Disable KV cache during training
    llm.to(device)
    
    # Print trainable parameter summary
    llm.print_trainable_parameters()
    
    # =========================================================================
    # Load Vision Encoder
    # =========================================================================
    logger.info(f"Loading SigLip vision encoder: {config.vision_model_path}")
    
    image_processor = SiglipImageProcessor.from_pretrained(config.vision_model_path)
    vision_encoder = SiglipVisionModel.from_pretrained(config.vision_model_path)
    vision_encoder.to(device).eval()
    
    logger.info(f"Vision encoder loaded - hidden size: {vision_encoder.config.hidden_size}")
    
    # =========================================================================
    # Memory info
    # =========================================================================
    mem_info = get_gpu_memory_info()
    logger.info(f"GPU memory after loading: {mem_info}")
    
    return tokenizer, llm, image_processor, vision_encoder, compute_dtype


# ==============================================================================
# CHECKPOINT MANAGEMENT
# ==============================================================================

def save_checkpoint(
    model: SurgicalVLM,
    tokenizer,
    config: TrainConfig,
    output_dir: Path,
    name: str,
    step: int = 0,
    metrics: Optional[Dict] = None
) -> Path:
    """
    Save model checkpoint.
    
    Saves:
    - Tokenizer (for vocab with special tokens)
    - LoRA weights (adapter weights only - small)
    - Visual projector weights
    - Training info (config, step, metrics)
    
    Args:
        model: VLM model
        tokenizer: Tokenizer
        config: Training config
        output_dir: Base output directory
        name: Checkpoint name (e.g., "best", "step_1000")
        step: Current training step
        metrics: Evaluation metrics
    
    Returns:
        Path to checkpoint directory
    """
    ckpt_dir = output_dir / f"checkpoint_{name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")
    
    # Save LoRA weights (small, contains only adapter parameters)
    model.llm.save_pretrained(ckpt_dir / "lora_weights")
    
    # Save visual projector weights
    torch.save(
        model.visual_projector.state_dict(),
        ckpt_dir / "visual_projector.pt"
    )
    
    # Save training info
    info = {
        "config": asdict(config),
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.now().isoformat(),
    }
    with open(ckpt_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Saved checkpoint: {ckpt_dir}")
    return ckpt_dir


def load_checkpoint(
    model: SurgicalVLM,
    checkpoint_path: Path,
) -> Dict:
    """
    Load model from checkpoint.
    
    Args:
        model: VLM model to load weights into
        checkpoint_path: Path to checkpoint directory
    
    Returns:
        Info dict with training state
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load visual projector
    projector_path = checkpoint_path / "visual_projector.pt"
    if projector_path.exists():
        model.visual_projector.load_state_dict(
            torch.load(projector_path, map_location="cpu")
        )
    
    # Load info
    info_path = checkpoint_path / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}
    
    return info


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(
    model: SurgicalVLM,
    tokenizer,
    samples: List[Dict],
    config: TrainConfig,
    device: torch.device,
    max_samples: int = 100,
    split_name: str = "validation"
) -> Dict[str, Any]:
    """
    Evaluate model on a set of samples.
    
    This function:
    1. Sets model to eval mode
    2. Generates predictions for each sample
    3. Computes accuracy metrics (exact match)
    4. Computes per-task metrics
    
    IMPORTANT: Uses separate random generator to avoid affecting training randomness.
    
    Args:
        model: VLM model
        tokenizer: Tokenizer
        samples: List of evaluation samples
        config: Training config
        device: Target device
        max_samples: Maximum samples to evaluate (for speed)
        split_name: Name of split for logging
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Use separate random generator for evaluation (no effect on training)
    eval_rng = random.Random(config.seed + 12345)
    
    # Sample subset if needed
    if max_samples > 0 and len(samples) > max_samples:
        eval_subset = eval_rng.sample(samples, max_samples)
    else:
        eval_subset = samples
    
    # Track results
    correct = 0
    total = 0
    task_results = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": []})
    all_predictions = []
    
    for sample in tqdm(eval_subset, desc=f"Evaluating ({split_name})", leave=False):
        try:
            # Load and process frames
            frames = []
            for path in sample.get("frames", [])[:config.num_frames]:
                path = path.replace("\\", "/")
                if os.path.exists(path):
                    frames.append(Image.open(path).convert("RGB"))
                else:
                    frames.append(Image.new("RGB", (384, 384), (128, 128, 128)))
            
            # Pad frames
            while len(frames) < config.num_frames:
                frames.append(frames[-1] if frames else Image.new("RGB", (384, 384)))
            frames = frames[:config.num_frames]
            
            # Process frames
            processed = model.image_processor(images=frames, return_tensors="pt")
            pixel_values = processed["pixel_values"].unsqueeze(0).to(device)
            
            # Get Q&A
            convs = sample.get("conversations", [])
            question = convs[0]["value"] if convs else ""
            ground_truth = convs[1]["value"] if len(convs) > 1 else ""
            
            # Tokenize prompt
            prompt = f"{question}\nAnswer:"
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            
            # Generate
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=64,
            )
            
            # Decode prediction
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract answer part
            if "Answer:" in prediction:
                prediction = prediction.split("Answer:")[-1].strip()
            
            # Check correctness (exact match, case-insensitive)
            pred_clean = prediction.strip().lower()
            gt_clean = ground_truth.strip().lower()
            is_correct = pred_clean == gt_clean
            
            # Update metrics
            task = sample.get("main_tag", "unknown")
            task_results[task]["correct"] += int(is_correct)
            task_results[task]["total"] += 1
            task_results[task]["predictions"].append({
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": is_correct,
            })
            
            correct += int(is_correct)
            total += 1
            
            all_predictions.append({
                "sample_id": sample.get("id", "unknown"),
                "task": task,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": is_correct,
            })
            
        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            continue
    
    # Compute final metrics
    metrics = {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "split": split_name,
    }
    
    # Per-task metrics
    for task, results in task_results.items():
        task_acc = results["correct"] / results["total"] if results["total"] > 0 else 0.0
        metrics[f"acc_{task}"] = task_acc
        metrics[f"total_{task}"] = results["total"]
    
    # Restore training mode
    model.train()
    
    return metrics


def final_test_evaluation(
    model: SurgicalVLM,
    tokenizer,
    test_samples: List[Dict],
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run comprehensive final evaluation on test set.
    
    This is run ONLY ONCE at the end of training to report final performance.
    Uses ALL test samples (not a subset).
    
    Args:
        model: Trained VLM model
        tokenizer: Tokenizer
        test_samples: Complete test set
        config: Training config
        device: Target device
        output_dir: Directory to save results
    
    Returns:
        Dictionary with comprehensive test metrics
    """
    logger.info("=" * 70)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Evaluating on {len(test_samples)} test samples")
    logger.info("NOTE: Test set was held out during training - no data leakage")
    
    # Evaluate on full test set
    metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        config=config,
        device=device,
        max_samples=-1,  # Use all samples
        split_name="test",
    )
    
    # Log results
    logger.info("=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    for key, value in metrics.items():
        if key.startswith("acc_"):
            task = key.replace("acc_", "")
            total_key = f"total_{task}"
            total = metrics.get(total_key, 0)
            logger.info(f"  {task}: {value:.4f} (n={total})")
    
    logger.info("=" * 70)
    
    # Save results to file
    results_path = output_dir / "final_test_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved test results to: {results_path}")
    
    return metrics


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(config: TrainConfig) -> Tuple[str, Dict]:
    """
    Main training function.
    
    This function orchestrates the complete training process:
    1. Setup (seeds, directories, logging)
    2. Data loading and splitting
    3. Model initialization
    4. Training loop with gradient accumulation
    5. Periodic validation (on validation set - NOT test!)
    6. Checkpointing
    7. Final test evaluation (ONLY at the end)
    
    Args:
        config: Training configuration
    
    Returns:
        Tuple of (output_directory, final_metrics)
    """
    # Declare global logger at the start of function (before any usage)
    global logger
    
    # =========================================================================
    # Setup
    # =========================================================================
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {get_gpu_memory_info()}")
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reconfigure logger with file handler
    logger = setup_logging(output_dir)
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Saved config to: {config_path}")
    
    # =========================================================================
    # Load and split data (NO DATA LEAKAGE)
    # =========================================================================
    train_samples, val_samples, test_samples = load_and_split_data(config)
    
    if len(train_samples) == 0:
        raise ValueError("No training samples! Check data path and target_tasks.")
    
    # =========================================================================
    # Load models
    # =========================================================================
    tokenizer, llm, image_processor, vision_encoder, compute_dtype = load_models(config, device)
    
    # =========================================================================
    # Create VLM
    # =========================================================================
    model = SurgicalVLM(
        vision_encoder=vision_encoder,
        image_processor=image_processor,
        llm=llm,
        config=config,
    ).to(device)
    
    # =========================================================================
    # Create datasets and dataloaders
    # =========================================================================
    train_dataset = SurgicalVLMDataset(
        samples=train_samples,
        image_processor=image_processor,
        num_frames=config.num_frames,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    # =========================================================================
    # Setup optimizer and scheduler
    # =========================================================================
    
    # Collect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Calculate total steps
    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = int(config.warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # =========================================================================
    # Setup AMP (Automatic Mixed Precision)
    # =========================================================================
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and not config.use_bf16))
    
    # =========================================================================
    # Print training summary
    # =========================================================================
    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    logger.info(f"Test samples: {len(test_samples)} (held out until final eval)")
    logger.info(f"Tasks: {config.target_tasks}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size} x {config.grad_accum} = {config.batch_size * config.grad_accum}")
    logger.info(f"Total optimizer steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Visual tokens: {config.visual_tokens}")
    logger.info(f"Projector type: {config.projector_type}")
    logger.info(f"LoRA rank: {config.lora_r}")
    logger.info("=" * 70)
    
    # =========================================================================
    # Training loop
    # =========================================================================
    global_step = 0
    best_val_acc = 0.0
    epochs_without_improvement = 0
    training_history = []
    
    model.train()
    optimizer.zero_grad()
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            pixel_values = batch["pixel_values"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]
            
            # Tokenize text
            input_ids, attention_mask, labels = build_batch_text(
                tokenizer, questions, answers, config.max_seq_len, device
            )
            
            # Forward pass with AMP
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(pixel_values, input_ids, attention_mask, labels)
                loss = loss / config.grad_accum  # Scale for gradient accumulation
            
            # Backward pass
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += loss.item() * config.grad_accum
            epoch_batches += 1
            
            # Optimizer step (after gradient accumulation)
            if (batch_idx + 1) % config.grad_accum == 0:
                # Gradient clipping
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                
                # Optimizer step
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                # Logging
                if global_step % config.log_every_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / epoch_batches
                    elapsed = time.time() - start_time
                    
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "step": global_step,
                    })
                    
                    training_history.append({
                        "step": global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "elapsed": elapsed,
                    })
                
                # Validation (on VALIDATION set - NOT test!)
                if val_samples and config.eval_every_steps > 0 and global_step % config.eval_every_steps == 0:
                    val_metrics = evaluate(
                        model=model,
                        tokenizer=tokenizer,
                        samples=val_samples,
                        config=config,
                        device=device,
                        max_samples=config.eval_samples,
                        split_name="validation",
                    )
                    
                    logger.info(f"[Step {global_step}] Validation: {val_metrics}")
                    
                    # Check for improvement
                    if val_metrics["accuracy"] > best_val_acc + config.early_stopping_min_delta:
                        best_val_acc = val_metrics["accuracy"]
                        epochs_without_improvement = 0
                        save_checkpoint(
                            model, tokenizer, config, output_dir,
                            "best", global_step, val_metrics
                        )
                    else:
                        epochs_without_improvement += 1
                    
                    # Early stopping
                    if config.early_stopping and epochs_without_improvement >= config.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epochs_without_improvement} evaluations without improvement")
                        break
                    
                    model.train()
                
                # Periodic checkpoint
                if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
                    save_checkpoint(
                        model, tokenizer, config, output_dir,
                        f"step_{global_step}", global_step
                    )
        
        # End of epoch
        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
        
        save_checkpoint(
            model, tokenizer, config, output_dir,
            f"epoch_{epoch+1}", global_step
        )
        
        # Check early stopping at epoch level too
        if config.early_stopping and epochs_without_improvement >= config.early_stopping_patience:
            break
    
    # =========================================================================
    # Save final checkpoint
    # =========================================================================
    save_checkpoint(
        model, tokenizer, config, output_dir,
        "final", global_step
    )
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    
    # =========================================================================
    # Final test evaluation (ONLY HERE - no leakage!)
    # =========================================================================
    final_metrics = {}
    if test_samples:
        # Load best checkpoint for final evaluation
        best_ckpt = output_dir / "checkpoint_best"
        if best_ckpt.exists():
            load_checkpoint(model, best_ckpt)
            logger.info("Loaded best checkpoint for final evaluation")
        
        final_metrics = final_test_evaluation(
            model=model,
            tokenizer=tokenizer,
            test_samples=test_samples,
            config=config,
            device=device,
            output_dir=output_dir,
        )
    
    # =========================================================================
    # Training complete
    # =========================================================================
    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {format_time(total_time)}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    if final_metrics:
        logger.info(f"Final test accuracy: {final_metrics.get('accuracy', 'N/A'):.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 70)
    
    return str(output_dir), final_metrics


# ==============================================================================
# ABLATION STUDY
# ==============================================================================

def run_ablation_study(base_config: TrainConfig) -> Dict[str, Any]:
    """
    Run ablation study over specified parameter values.
    
    This function systematically varies one hyperparameter while keeping
    others constant to understand its impact on model performance.
    
    Args:
        base_config: Base configuration to modify
    
    Returns:
        Dictionary with ablation results
    """
    logger.info("=" * 70)
    logger.info("ABLATION STUDY")
    logger.info("=" * 70)
    logger.info(f"Parameter: {base_config.ablation_param}")
    logger.info(f"Values: {base_config.ablation_values}")
    logger.info("=" * 70)
    
    results = {
        "parameter": base_config.ablation_param,
        "values": base_config.ablation_values,
        "runs": [],
    }
    
    for value in base_config.ablation_values:
        logger.info(f"\n{'='*70}")
        logger.info(f"ABLATION RUN: {base_config.ablation_param} = {value}")
        logger.info(f"{'='*70}\n")
        
        # Create modified config
        run_config = copy.deepcopy(base_config)
        setattr(run_config, base_config.ablation_param, value)
        run_config.run_ablation = False  # Prevent recursion
        
        # Update output directory
        run_config.output_dir = str(
            Path(base_config.output_dir) / f"ablation_{base_config.ablation_param}" / f"{value}"
        )
        
        try:
            # Run training
            output_dir, final_metrics = train(run_config)
            
            results["runs"].append({
                "value": value,
                "output_dir": output_dir,
                "metrics": final_metrics,
                "success": True,
            })
            
        except Exception as e:
            logger.error(f"Ablation run failed for {base_config.ablation_param}={value}: {e}")
            results["runs"].append({
                "value": value,
                "error": str(e),
                "success": False,
            })
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Parameter: {base_config.ablation_param}")
    logger.info("-" * 70)
    
    for run in results["runs"]:
        if run["success"]:
            acc = run["metrics"].get("accuracy", "N/A")
            logger.info(f"  {run['value']:>10}: Test Accuracy = {acc:.4f}")
        else:
            logger.info(f"  {run['value']:>10}: FAILED - {run.get('error', 'Unknown error')}")
    
    logger.info("=" * 70)
    
    # Save ablation results
    results_path = Path(base_config.output_dir) / f"ablation_{base_config.ablation_param}" / "summary.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Surgical VLM with ablation support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Basic training:
python train_surgical_vlm_v2.py --train_data train.json --output_dir ./checkpoints

# With separate validation set:
python train_surgical_vlm_v2.py --train_data train.json --val_data val.json --test_data test.json

# Ablation on visual tokens:
python train_surgical_vlm_v2.py --run_ablation --ablation_param visual_tokens --ablation_values 16 32 64 128 256

# Ablation on LoRA rank:
python train_surgical_vlm_v2.py --run_ablation --ablation_param lora_r --ablation_values 8 16 32 64 128

# Ablation on learning rate:
python train_surgical_vlm_v2.py --run_ablation --ablation_param learning_rate --ablation_values 1e-5 5e-5 1e-4 2e-4 5e-4
        """
    )
    
    # Data paths
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument("--train_data", type=str, default="/mnt/share/ali/VLM_Project/hospital_data/train.json",
                           help="Path to training data JSON")
    data_group.add_argument("--val_data", type=str, default=None,
                           help="Path to validation data JSON (optional, splits from train if not provided)")
    data_group.add_argument("--test_data", type=str, default="/mnt/share/ali/VLM_Project/hospital_data/test.json",
                           help="Path to test data JSON (held out until final eval)")
    data_group.add_argument("--output_dir", type=str, default="/mnt/share/ali/VLM_Project/checkpoints",
                           help="Output directory for checkpoints and logs")
    
    # Model paths
    model_group = parser.add_argument_group("Model paths")
    model_group.add_argument("--vision_model", type=str, default="/mnt/share/ali/VLMs/hf_cache/hub/siglip2-model/",
                            help="Path to SigLip2 vision encoder")
    model_group.add_argument("--language_model", type=str, default="/mnt/share/ali/VLMs/hf_cache/hub/Qwen3-0.6B/",
                            help="Path to Qwen3 language model")
    
    # Tasks
    parser.add_argument("--tasks", nargs="+", default=["step_classification", "stage_classification"],
                       help="Tasks to train on")
    
    # Training hyperparameters
    train_group = parser.add_argument_group("Training hyperparameters")
    train_group.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    train_group.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    train_group.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Model settings
    model_settings = parser.add_argument_group("Model settings")
    model_settings.add_argument("--visual_tokens", type=int, default=256, help="Number of visual tokens")
    model_settings.add_argument("--num_frames", type=int, default=8, help="Number of video frames")
    model_settings.add_argument("--projector_type", type=str, default="mlp",
                               choices=["mlp", "linear", "attention"], help="Visual projector type")
    model_settings.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    model_settings.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    model_settings.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Ablation settings
    ablation_group = parser.add_argument_group("Ablation study")
    ablation_group.add_argument("--run_ablation", action="store_true", help="Run ablation study")
    ablation_group.add_argument("--ablation_param", type=str, default=None,
                               choices=["visual_tokens", "lora_r", "lora_alpha", "learning_rate",
                                       "num_frames", "projector_type", "lora_dropout"],
                               help="Parameter to ablate")
    ablation_group.add_argument("--ablation_values", nargs="+", default=None,
                               help="Values to try for ablation parameter")
    
    # Other settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_split_ratio", type=float, default=0.1,
                       help="Fraction of train data to use for validation (if no val_data provided)")
    parser.add_argument("--eval_every_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_every_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples for validation")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                       help="Early stopping patience (evaluations without improvement)")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Convert ablation values to appropriate types
    ablation_values = None
    if args.ablation_values:
        if args.ablation_param == "learning_rate":
            ablation_values = [float(v) for v in args.ablation_values]
        elif args.ablation_param == "projector_type":
            ablation_values = args.ablation_values  # strings
        elif args.ablation_param == "lora_dropout":
            ablation_values = [float(v) for v in args.ablation_values]
        else:
            ablation_values = [int(v) for v in args.ablation_values]
    
    # Build config
    config = TrainConfig(
        # Data paths
        train_data=args.train_data,
        val_data=args.val_data,
        test_data=args.test_data,
        output_dir=args.output_dir,
        
        # Model paths
        vision_model_path=args.vision_model,
        language_model_path=args.language_model,
        
        # Tasks
        target_tasks=args.tasks,
        
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        
        # Model settings
        visual_tokens=args.visual_tokens,
        num_frames=args.num_frames,
        projector_type=args.projector_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        
        # Ablation
        run_ablation=args.run_ablation,
        ablation_param=args.ablation_param,
        ablation_values=ablation_values,
        
        # Other
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
        eval_samples=args.eval_samples,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    # Run ablation or single training
    if config.run_ablation:
        if not config.ablation_param or not config.ablation_values:
            logger.error("Ablation requires --ablation_param and --ablation_values")
            return
        run_ablation_study(config)
    else:
        train(config)


if __name__ == "__main__":
    main()