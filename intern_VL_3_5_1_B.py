#!/usr/bin/env python3
"""
InternVL2.5-8B QLoRA Fine-tuning for Surgical Video Classification
==================================================================

This script performs multi-task classification (step and stage) on surgical video
frames using InternVL2.5-8B with QLoRA (Quantized Low-Rank Adaptation).

The pipeline:
1. Loads InternVL2.5-8B in 4-bit quantization (NF4) to reduce VRAM usage
2. Applies LoRA adapters to the language model for efficient fine-tuning
3. Uses a multi-task classification head for:
   - Step classification: 13 surgical procedure steps
   - Stage classification: 3 surgical stages
4. Pools features from the last image context token position
5. Trains with gradient accumulation and early stopping based on validation score

Key Features:
- 4-bit quantization with bfloat16 compute dtype for memory efficiency
- LoRA fine-tuning (rank=8, alpha=16) on attention layers (wqkv, wo)
- Visual-aligned pooling: extracts features from the last IMG_CONTEXT token
- Subject-wise train/val split to prevent data leakage
- Tracks best model based on average macro-F1 across both tasks
- Comprehensive metrics: accuracy, macro-F1, Cohen's kappa
- Class-wise F1 and confusion matrices for final test evaluation

Requirements:
- torch >= 2.0
- transformers >= 4.50
- peft (for LoRA)
- bitsandbytes (for 4-bit quantization)
- PIL, torchvision
- scikit-learn
- tqdm

Usage:
    python internvl_surgical_classification.py [--runs N] [--epochs E] [--seed S]

Author: Converted from Jupyter notebook
"""

import os
import sys
import json
import copy
import time
import random
import argparse
import importlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment setup - must be done before importing certain libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths - modify these to match your setup
MODEL_DIR = Path("/home/ali/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-1B/")
TRAIN_JSON = Path("/mnt/share/ali/VLM_Project/hospital_data/train.cleaned.json")
TEST_JSON = Path("/mnt/share/ali/VLM_Project/hospital_data/test.cleaned.json")

# Model and training hyperparameters
IMG_SIZE = 448              # InternVL2.5 expected image size
N_FRAMES = 8                # Number of frames to sample per video
CTX_PER_IMAGE = 256         # Number of context tokens per image in InternVL2.5
HIDDEN_DIM = 2048           # LLM hidden dimension
LORA_RANK = 8               # LoRA rank (lower = fewer parameters)
LORA_ALPHA = 16             # LoRA scaling factor
LORA_DROPOUT = 0.05         # Dropout in LoRA layers
LEARNING_RATE = 2e-4        # AdamW learning rate
WEIGHT_DECAY = 0.01         # L2 regularization
ACCUM_STEPS = 8             # Gradient accumulation steps
VAL_SUBJECT_RATIO = 0.2     # Fraction of subjects for validation

# Allowed classification tasks
ALLOWED_TAGS = {"step_classification", "stage_classification"}


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_json(path: Path) -> List[Dict]:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_subject(sample: Dict) -> str:
    """
    Extract subject identifier from a sample.
    
    The subject is used for train/val splitting to ensure that frames from
    the same subject don't appear in both train and validation sets.
    This prevents data leakage and gives a more realistic generalization estimate.
    """
    meta = sample.get("meta", {}) or {}
    if meta.get("subject"):
        return meta["subject"]
    # Fallback: extract from sample ID
    _id = sample.get("id", "")
    return _id.split("__")[0] if "__" in _id else "UNKNOWN"


def get_label(sample: Dict) -> str:
    """Extract the label (step_id or stage_id) from a sample."""
    meta = sample.get("meta", {}) or {}
    if sample["main_tag"] == "step_classification":
        return meta.get("step_id")
    else:
        return meta.get("stage_id")


def prepare_data(seed: int = 42) -> Tuple[List, List, List, Dict, Dict]:
    """
    Load and prepare train/val/test splits with subject-wise validation split.
    
    Subject-wise splitting is crucial for medical/surgical data because:
    - Frames from the same surgery share visual characteristics
    - Patient/surgeon variations should be captured in evaluation
    - Prevents overfitting to subject-specific patterns
    
    Returns:
        train_split: Training samples
        val_split: Validation samples (used for model selection)
        test_kept: Test samples (final evaluation)
        label_list: Dict mapping task names to list of class labels
        label2id: Dict mapping (task, label) to integer index
    """
    random.seed(seed)
    
    # Load raw data
    train_all = load_json(TRAIN_JSON)
    test_all = load_json(TEST_JSON)
    
    # Filter to only allowed tasks (step and stage classification)
    train_kept = [s for s in train_all if s.get("main_tag") in ALLOWED_TAGS]
    test_kept = [s for s in test_all if s.get("main_tag") in ALLOWED_TAGS]
    
    print(f"train_kept: {len(train_kept)} {Counter([s['main_tag'] for s in train_kept])}")
    print(f"test_kept:  {len(test_kept)} {Counter([s['main_tag'] for s in test_kept])}")
    
    # Build subject-wise index for train/val split
    idx_by_subj = defaultdict(list)
    for i, s in enumerate(train_kept):
        idx_by_subj[get_subject(s)].append(i)
    
    subjects = sorted(idx_by_subj.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)
    
    # Split subjects into train and validation
    n_val_subj = max(1, int(round(len(subjects) * VAL_SUBJECT_RATIO)))
    val_subjects = set(subjects[:n_val_subj])
    
    train_idx, val_idx = [], []
    for subj, idxs in idx_by_subj.items():
        (val_idx if subj in val_subjects else train_idx).extend(idxs)
    
    train_split = [train_kept[i] for i in train_idx]
    val_split = [train_kept[i] for i in val_idx]
    
    print(f"train_split: {len(train_split)} {Counter([s['main_tag'] for s in train_split])}")
    print(f"val_split:   {len(val_split)} {Counter([s['main_tag'] for s in val_split])}")
    print(f"val subjects: {sorted(list(val_subjects))[:10]}...")
    
    # Build label mappings from full training set
    # This ensures all classes are represented even if some are missing from val
    label_list = {
        "step_classification": sorted(
            {get_label(s) for s in train_kept if s["main_tag"] == "step_classification"}
        ),
        "stage_classification": sorted(
            {get_label(s) for s in train_kept if s["main_tag"] == "stage_classification"}
        ),
    }
    label2id = {t: {lab: i for i, lab in enumerate(label_list[t])} for t in label_list}
    
    print(f"step classes:  {len(label_list['step_classification'])} {label_list['step_classification']}")
    print(f"stage classes: {len(label_list['stage_classification'])} {label_list['stage_classification']}")
    
    return train_split, val_split, test_kept, label_list, label2id


# ============================================================================
# MODEL SETUP
# ============================================================================

def load_model_and_tokenizer():
    """
    Load InternVL2.5-8B in 4-bit quantization with LoRA adapters.
    
    Quantization Strategy:
    - NF4 (Normal Float 4-bit): Better quality than INT4 for neural networks
    - Double quantization: Quantizes the quantization constants for more savings
    - bfloat16 compute: Fast computation with good numerical stability
    
    This reduces VRAM from ~16GB (fp16) to ~5GB while maintaining most quality.
    
    LoRA Strategy:
    - Only adapts attention layers (wqkv, wo) in the language model
    - Vision encoder is frozen (transfer learning from pretrained features)
    - Rank 8 with alpha 16 gives 2x scaling (alpha/rank)
    - Small dropout (0.05) for regularization
    """
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Check bitsandbytes is available
    bnb_spec = importlib.util.find_spec("bitsandbytes")
    if bnb_spec is None:
        raise ImportError("bitsandbytes is required for 4-bit quantization")
    print("bitsandbytes installed: True")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # Normal Float 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
        bnb_4bit_use_double_quant=True,      # Quantize the quantization constants
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    # Load model with quantization
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )
    model.eval()
    
    print(f"Loaded model: {type(model)}")
    print(f"cuda mem allocated (GB): {torch.cuda.memory_allocated()/1024**3:.2f}")
    print(f"cuda mem reserved  (GB): {torch.cuda.memory_reserved()/1024**3:.2f}")
    
    # Recover img_context_token_id (required for visual feature injection)
    # InternVL uses this special token to mark where image features go in the sequence
    candidates = ["<IMG_CONTEXT>", "<img_context>", "<image_context>", "<im_context>"]
    found = []
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            found.append((tok, tid))
    
    if len(found) == 0:
        img_ctx_id = 92546  # Fallback to known value
        print(f"⚠️ No token string found; falling back to img_context_token_id = {img_ctx_id}")
    else:
        img_ctx_id = found[0][1]
        print(f"✅ Using img_context_token_id from tokenizer: {img_ctx_id}")
    
    model.img_context_token_id = img_ctx_id
    
    # Attach LoRA adapters to the language model
    # We only fine-tune the LLM's attention layers; vision encoder stays frozen
    lm = model.language_model
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,                    # Low rank for parameter efficiency
        lora_alpha=LORA_ALPHA,          # Scaling factor
        lora_dropout=LORA_DROPOUT,      # Regularization
        target_modules=["wqkv", "wo"],  # InternLM2 attention layer names
        bias="none",                    # Don't add bias to LoRA layers
    )
    lm = get_peft_model(lm, lora_cfg)
    model.language_model = lm
    
    # Patch for cache handling during generation (prevents errors)
    if not hasattr(model.language_model, "_orig_prepare_inputs_for_generation"):
        model.language_model._orig_prepare_inputs_for_generation = (
            model.language_model.prepare_inputs_for_generation
        )
    
    import types
    def safe_prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Safely handle potentially corrupted KV cache."""
        bad_cache = False
        try:
            if past_key_values is not None:
                if len(past_key_values) == 0:
                    bad_cache = True
                else:
                    first = past_key_values[0]
                    if first is None or (isinstance(first, (tuple, list)) and len(first) > 0 and first[0] is None):
                        bad_cache = True
        except Exception:
            bad_cache = True
        if bad_cache:
            past_key_values = None
        return self._orig_prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    model.language_model.prepare_inputs_for_generation = types.MethodType(
        safe_prepare_inputs_for_generation, model.language_model
    )
    
    return model, tokenizer, img_ctx_id


# ============================================================================
# DATASET AND DATALOADER
# ============================================================================

class SurgicalDataset(Dataset):
    """
    Dataset for surgical video frame classification.
    
    Each sample contains:
    - Multiple frames (up to N_FRAMES) from a surgical video
    - Task type (step or stage classification)
    - Ground truth label
    
    The frames are processed with:
    - Resize to IMG_SIZE x IMG_SIZE
    - ImageNet normalization (mean, std from training on ImageNet)
    - Stack into tensor of shape [N_frames, 3, H, W]
    
    The input sequence is constructed as:
    [BOS] [IMG_CTX x (N_frames * CTX_PER_IMAGE)] [prompt_tokens] [EOS]
    
    Where IMG_CTX tokens will be replaced by visual features during forward pass.
    """
    
    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        label2id: Dict,
        img_ctx_id: int,
        n_frames: int = N_FRAMES,
        img_size: int = IMG_SIZE,
        ctx_per_image: int = CTX_PER_IMAGE,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.img_ctx_id = img_ctx_id
        self.n_frames = n_frames
        self.ctx_per_image = ctx_per_image
        
        # Standard ImageNet normalization
        # InternVL expects this preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _build_inputs(self, n_frames: int, tag: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build input_ids, attention_mask, and image_flags for a sample.
        
        The prompt is task-specific:
        - Step classification: "Classify step. Answer step_XXX."
        - Stage classification: "Classify stage. Answer stage_XX."
        
        Image context tokens are placeholders that get replaced by visual features.
        """
        bos = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        eos = self.tokenizer.eos_token_id
        n_ctx = n_frames * self.ctx_per_image
        
        # Task-specific prompts
        if tag == "step_classification":
            prompt = "Classify step. Answer step_XXX."
        else:
            prompt = "Classify stage. Answer stage_XX."
        
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        
        # Construct sequence: [BOS] [IMG_CTX * n_ctx] [prompt] [EOS]
        ids = [bos] + [self.img_ctx_id] * n_ctx + prompt_ids
        if eos is not None:
            ids.append(eos)
        
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        image_flags = torch.ones((n_frames,), dtype=torch.long)  # All frames are real images
        
        return input_ids, attention_mask, image_flags
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        tag = s["main_tag"]
        meta = s.get("meta", {}) or {}
        
        # Get ground truth label index
        label_key = "step_id" if tag == "step_classification" else "stage_id"
        y = self.label2id[tag][meta[label_key]]
        
        # Load and preprocess frames
        frames = s["frames"][:self.n_frames]
        imgs = [Image.open(p).convert("RGB") for p in frames]
        pixel_values = torch.stack([self.transform(im) for im in imgs], dim=0)  # [N, 3, H, W]
        
        # Build input sequence
        input_ids, attention_mask, image_flags = self._build_inputs(pixel_values.shape[0], tag)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
            "tag": tag,
            "y": y,
        }


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Handles variable-length sequences by padding to the maximum length in the batch.
    Note: With batch_size=1 (memory constraints), this mainly handles consistency.
    """
    max_len = max(b["input_ids"].shape[0] for b in batch)
    
    def pad(x: torch.Tensor, fill_value: int) -> torch.Tensor:
        if x.shape[0] == max_len:
            return x
        padding = torch.full((max_len - x.shape[0],), fill_value, dtype=x.dtype)
        return torch.cat([x, padding], dim=0)
    
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),  # [B, N, 3, H, W]
        "input_ids": torch.stack([pad(b["input_ids"], pad_token_id) for b in batch], dim=0),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch], dim=0),
        "image_flags": torch.stack([b["image_flags"] for b in batch], dim=0),  # [B, N]
        "tags": [b["tag"] for b in batch],
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
    }


# ============================================================================
# CLASSIFICATION HEAD
# ============================================================================

class MultiTaskHead(nn.Module):
    """
    Multi-task classification head for step and stage prediction.
    
    Takes the pooled hidden state from the language model and produces
    logits for both tasks simultaneously. During training, only the
    relevant task's loss is computed based on the sample's tag.
    
    Architecture:
    - Two separate linear layers (no shared parameters between tasks)
    - No activation (logits are passed to softmax/cross-entropy)
    """
    
    def __init__(self, hidden_dim: int = HIDDEN_DIM, n_step: int = 13, n_stage: int = 3):
        super().__init__()
        self.step = nn.Linear(hidden_dim, n_step)
        self.stage = nn.Linear(hidden_dim, n_stage)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Pooled features of shape [B, hidden_dim]
        
        Returns:
            Dict with 'step_logits' [B, n_step] and 'stage_logits' [B, n_stage]
        """
        return {
            "step_logits": self.step(x),
            "stage_logits": self.stage(x),
        }


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def summarize(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute classification metrics for a single task."""
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "n": len(y_true),
    }


def forward_pooled(
    model,
    batch: Dict,
    device: torch.device,
    ctx_per_image: int = CTX_PER_IMAGE
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Forward pass with visual-aligned pooling.
    
    Pooling Strategy:
    We extract features from the last IMG_CONTEXT token position. This position
    contains aggregated visual information from all frames after being processed
    by the vision encoder and projected into the language model's embedding space.
    
    Why this position?
    - The IMG_CTX tokens are where visual features are injected
    - The last IMG_CTX position has seen all preceding visual context
    - Alternative: pooling from the final token (less visual-aligned)
    
    Returns:
        pooled: Features of shape [B, hidden_dim]
        tags: List of task tags
        y: Ground truth labels
    """
    B, N, C, H, W = batch["pixel_values"].shape
    
    # Convert to float16 for the vision encoder (matches loaded weights)
    pixel_values_bn = batch["pixel_values"].to(device=device, dtype=torch.float16).view(B*N, C, H, W)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    image_flags = batch["image_flags"].to(device).view(B*N)
    
    # Calculate position of last IMG_CTX token
    # Sequence structure: [BOS] [IMG_CTX * n_ctx] [prompt] [EOS]
    # Position = 1 (after BOS) + n_ctx - 1 (last CTX token)
    n_ctx = N * ctx_per_image
    img_ctx_pos = n_ctx  # 1 + n_ctx - 1 = n_ctx (0-indexed, so position n_ctx is the last IMG_CTX)
    
    # Forward through model
    out = model(
        pixel_values=pixel_values_bn,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,  # Disable KV cache during training
    )
    
    # Extract features from last layer, last IMG_CTX position
    hidden_states = out.hidden_states[-1]  # [B, seq_len, hidden_dim]
    pooled = hidden_states[:, img_ctx_pos, :].float()  # Convert to float32 for head
    
    return pooled, batch["tags"], batch["y"].to(device)


def eval_loader(
    model,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "eval"
) -> Tuple[Dict, Dict]:
    """
    Evaluate on a data loader and return metrics for both tasks.
    
    Returns:
        step_metrics: Dict with acc, macro_f1, kappa, n for step classification
        stage_metrics: Dict with same keys for stage classification
    """
    model.eval()
    head.eval()
    
    ys_step, ps_step = [], []
    ys_stage, ps_stage = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            pooled, tags, y = forward_pooled(model, batch, device)
            logits = head(pooled)
            
            tag = tags[0]  # Batch size is 1
            y_true = int(y.item())
            
            if tag == "step_classification":
                pred = int(torch.argmax(logits["step_logits"], dim=-1).item())
                ys_step.append(y_true)
                ps_step.append(pred)
            else:
                pred = int(torch.argmax(logits["stage_logits"], dim=-1).item())
                ys_stage.append(y_true)
                ps_stage.append(pred)
    
    return summarize(ys_step, ps_step), summarize(ys_stage, ps_stage)


def train_one_epoch(
    model,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    accum_steps: int = ACCUM_STEPS,
    epoch: int = 1,
    total_epochs: int = 20,
) -> float:
    """
    Train for one epoch with gradient accumulation.
    
    Gradient accumulation allows effective larger batch sizes when memory is limited.
    With accum_steps=8 and batch_size=1, effective batch size is 8.
    
    Loss is computed only for the relevant task based on the sample's tag.
    
    Returns:
        avg_loss: Average training loss over the epoch
    """
    model.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    
    running_loss = 0.0
    
    for i, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}/{total_epochs}"), start=1):
        pooled, tags, y = forward_pooled(model, batch, device)
        logits = head(pooled)
        
        # Task-specific loss
        if tags[0] == "step_classification":
            loss = F.cross_entropy(logits["step_logits"], y)
        else:
            loss = F.cross_entropy(logits["stage_logits"], y)
        
        # Gradient accumulation: scale loss
        (loss / accum_steps).backward()
        
        # Optimizer step every accum_steps iterations
        if i % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item()
    
    # Handle remaining gradients if not divisible by accum_steps
    if len(train_loader) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    return running_loss / len(train_loader)


def train_and_evaluate(
    model,
    tokenizer,
    img_ctx_id: int,
    train_split: List,
    val_split: List,
    test_kept: List,
    label_list: Dict,
    label2id: Dict,
    device: torch.device,
    epochs: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full training pipeline with early stopping and test evaluation.
    
    Model selection is based on the average macro-F1 across both tasks on validation set.
    The best model checkpoint (LoRA weights + head) is saved and used for final test evaluation.
    
    Returns:
        results: Dict containing best val score, test metrics, and per-epoch history
    """
    torch.manual_seed(seed)
    
    # Create classification head
    n_step = len(label_list["step_classification"])
    n_stage = len(label_list["stage_classification"])
    head = MultiTaskHead(hidden_dim=HIDDEN_DIM, n_step=n_step, n_stage=n_stage).to(device)
    
    # Create datasets and loaders
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    train_ds = SurgicalDataset(train_split, tokenizer, label2id, img_ctx_id)
    val_ds = SurgicalDataset(val_split, tokenizer, label2id, img_ctx_id)
    test_ds = SurgicalDataset(test_kept, tokenizer, label2id, img_ctx_id)
    
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    
    # Optimizer: only trainable parameters (LoRA + head)
    params = [p for p in list(model.language_model.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training loop with best model tracking
    best = {"score": -1.0, "epoch": 0, "head": None, "lora": None}
    history = []
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, head, optimizer, train_loader, device,
            accum_steps=ACCUM_STEPS, epoch=epoch, total_epochs=epochs
        )
        
        # Validate
        step_res, stage_res = eval_loader(model, head, val_loader, device, desc=f"val epoch {epoch}")
        
        # Model selection score: average macro-F1
        score = 0.5 * (step_res["macro_f1"] + stage_res["macro_f1"])
        
        dt = time.time() - t0
        print(
            f"\nEpoch {epoch:02d} | {dt/60:.1f} min | train_loss {train_loss:.4f} | "
            f"VAL step acc {step_res['acc']:.3f} f1 {step_res['macro_f1']:.3f} k {step_res['kappa']:.3f} | "
            f"VAL stage acc {stage_res['acc']:.3f} f1 {stage_res['macro_f1']:.3f} k {stage_res['kappa']:.3f} | "
            f"score {score:.3f}"
        )
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_step": step_res,
            "val_stage": stage_res,
            "score": score,
        })
        
        # Save best model
        if score > best["score"]:
            best["score"] = score
            best["epoch"] = epoch
            best["head"] = copy.deepcopy(head.state_dict())
            best["lora"] = copy.deepcopy(model.language_model.state_dict())
            print(f"✅ New best @ epoch {epoch} (score={score:.3f})")
    
    print(f"\n=== Best epoch === {best['epoch']} best score: {best['score']:.4f}")
    
    # Restore best model and evaluate on test set
    head.load_state_dict(best["head"])
    model.language_model.load_state_dict(best["lora"], strict=False)
    
    test_step, test_stage = eval_loader(model, head, test_loader, device, desc="TEST(best)")
    
    print(f"\nTEST step:  {test_step}")
    print(f"TEST stage: {test_stage}")
    
    # Detailed classification report on test set
    print_detailed_test_report(model, head, test_loader, device, label_list)
    
    return {
        "best_epoch": best["epoch"],
        "best_val_score": best["score"],
        "test_step": test_step,
        "test_stage": test_stage,
        "history": history,
    }


def print_detailed_test_report(
    model,
    head: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    label_list: Dict,
):
    """Print class-wise F1 scores and confusion matrices for test set."""
    model.eval()
    head.eval()
    
    ys_step, ps_step = [], []
    ys_stage, ps_stage = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="predict(TEST)"):
            pooled, tags, y = forward_pooled(model, batch, device)
            logits = head(pooled)
            
            tag = tags[0]
            y_true = int(y.item())
            
            if tag == "step_classification":
                pred = int(torch.argmax(logits["step_logits"], dim=-1).item())
                ys_step.append(y_true)
                ps_step.append(pred)
            else:
                pred = int(torch.argmax(logits["stage_logits"], dim=-1).item())
                ys_stage.append(y_true)
                ps_stage.append(pred)
    
    step_names = label_list["step_classification"]
    stage_names = label_list["stage_classification"]
    
    print("\n===== TEST: Step (13-way) class-wise F1 =====")
    print(classification_report(ys_step, ps_step, target_names=step_names, digits=4, zero_division=0))
    
    print("\n===== TEST: Stage (3-way) class-wise F1 =====")
    print(classification_report(ys_stage, ps_stage, target_names=stage_names, digits=4, zero_division=0))
    
    print("\nConfusion Matrix (Step):")
    print(confusion_matrix(ys_step, ps_step))
    
    print("\nConfusion Matrix (Stage):")
    print(confusion_matrix(ys_stage, ps_stage))


# ============================================================================
# MULTI-RUN EXPERIMENT
# ============================================================================

def run_multiple_experiments(
    n_runs: int = 5,
    epochs: int = 20,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run multiple experiments with different random seeds and aggregate results.
    
    This provides:
    - Mean and standard deviation of all metrics
    - Per-run results for analysis
    - Statistical robustness of the reported numbers
    
    Args:
        n_runs: Number of experiment runs
        epochs: Training epochs per run
        base_seed: Base seed (run i uses seed = base_seed + i)
    
    Returns:
        Aggregated results with means, stds, and per-run details
    """
    print(f"\n{'='*60}")
    print(f"Running {n_runs} experiments with {epochs} epochs each")
    print(f"{'='*60}\n")
    
    # Verify environment
    import transformers
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"cuda: {torch.cuda.is_available()} device: {torch.cuda.current_device()} {torch.cuda.get_device_name(0)}")
    print(f"MODEL_DIR exists: {MODEL_DIR.exists()}")
    print(f"TRAIN_JSON exists: {TRAIN_JSON.exists()}")
    print(f"TEST_JSON exists: {TEST_JSON.exists()}")
    
    device = torch.device("cuda:0")
    all_results = []
    
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{n_runs} (seed={seed})")
        print(f"{'='*60}\n")
        
        # Prepare data with this seed
        train_split, val_split, test_kept, label_list, label2id = prepare_data(seed=seed)
        
        # Load fresh model for each run (to ensure clean LoRA initialization)
        # Note: In practice, you might want to reinitialize just the LoRA weights
        # Here we reload to ensure clean state
        model, tokenizer, img_ctx_id = load_model_and_tokenizer()
        
        # Train and evaluate
        results = train_and_evaluate(
            model, tokenizer, img_ctx_id,
            train_split, val_split, test_kept,
            label_list, label2id,
            device=device,
            epochs=epochs,
            seed=seed,
        )
        
        results["seed"] = seed
        results["run"] = run_idx + 1
        all_results.append(results)
        
        # Memory cleanup between runs
        del model
        torch.cuda.empty_cache()
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}\n")
    
    # Extract metrics
    step_accs = [r["test_step"]["acc"] for r in all_results]
    step_f1s = [r["test_step"]["macro_f1"] for r in all_results]
    step_kappas = [r["test_step"]["kappa"] for r in all_results]
    
    stage_accs = [r["test_stage"]["acc"] for r in all_results]
    stage_f1s = [r["test_stage"]["macro_f1"] for r in all_results]
    stage_kappas = [r["test_stage"]["kappa"] for r in all_results]
    
    val_scores = [r["best_val_score"] for r in all_results]
    
    print("Step Classification (13-way):")
    print(f"  Accuracy:  {np.mean(step_accs):.4f} ± {np.std(step_accs):.4f}")
    print(f"  Macro-F1:  {np.mean(step_f1s):.4f} ± {np.std(step_f1s):.4f}")
    print(f"  Kappa:     {np.mean(step_kappas):.4f} ± {np.std(step_kappas):.4f}")
    
    print("\nStage Classification (3-way):")
    print(f"  Accuracy:  {np.mean(stage_accs):.4f} ± {np.std(stage_accs):.4f}")
    print(f"  Macro-F1:  {np.mean(stage_f1s):.4f} ± {np.std(stage_f1s):.4f}")
    print(f"  Kappa:     {np.mean(stage_kappas):.4f} ± {np.std(stage_kappas):.4f}")
    
    print(f"\nValidation Score: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
    
    # Per-run summary
    print("\nPer-run Results:")
    print("-" * 80)
    print(f"{'Run':<5} {'Seed':<8} {'Step Acc':<10} {'Step F1':<10} {'Stage Acc':<10} {'Stage F1':<10}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['run']:<5} {r['seed']:<8} "
            f"{r['test_step']['acc']:<10.4f} {r['test_step']['macro_f1']:<10.4f} "
            f"{r['test_stage']['acc']:<10.4f} {r['test_stage']['macro_f1']:<10.4f}"
        )
    print("-" * 80)
    
    return {
        "n_runs": n_runs,
        "epochs": epochs,
        "per_run": all_results,
        "aggregate": {
            "step": {
                "acc_mean": np.mean(step_accs),
                "acc_std": np.std(step_accs),
                "f1_mean": np.mean(step_f1s),
                "f1_std": np.std(step_f1s),
                "kappa_mean": np.mean(step_kappas),
                "kappa_std": np.std(step_kappas),
            },
            "stage": {
                "acc_mean": np.mean(stage_accs),
                "acc_std": np.std(stage_accs),
                "f1_mean": np.mean(stage_f1s),
                "f1_std": np.std(stage_f1s),
                "kappa_mean": np.mean(stage_kappas),
                "kappa_std": np.std(stage_kappas),
            },
            "val_score_mean": np.mean(val_scores),
            "val_score_std": np.std(val_scores),
        },
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="InternVL2.5-8B QLoRA Fine-tuning for Surgical Classification"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of experiment runs (default: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs per run (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Run a single experiment instead of multiple runs"
    )
    
    args = parser.parse_args()
    
    if args.single:
        # Single run mode
        print("Running single experiment...")
        device = torch.device("cuda:0")
        
        train_split, val_split, test_kept, label_list, label2id = prepare_data(seed=args.seed)
        model, tokenizer, img_ctx_id = load_model_and_tokenizer()
        
        results = train_and_evaluate(
            model, tokenizer, img_ctx_id,
            train_split, val_split, test_kept,
            label_list, label2id,
            device=device,
            epochs=args.epochs,
            seed=args.seed,
        )
        
        print(f"\nFinal Results:")
        print(f"  Test Step Acc:  {results['test_step']['acc']:.4f}")
        print(f"  Test Step F1:   {results['test_step']['macro_f1']:.4f}")
        print(f"  Test Stage Acc: {results['test_stage']['acc']:.4f}")
        print(f"  Test Stage F1:  {results['test_stage']['macro_f1']:.4f}")
    else:
        # Multiple runs mode
        results = run_multiple_experiments(
            n_runs=args.runs,
            epochs=args.epochs,
            base_seed=args.seed,
        )
        
        # Save results to file
        output_file = f"results_{args.runs}runs_{args.epochs}epochs.json"
        with open(output_file, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            json.dump(convert(results), f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()