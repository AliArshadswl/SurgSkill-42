#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SURGICAL VLM TRAINING WITH ABLATION STUDIES
================================================================================

Comprehensive training script for surgical procedure understanding with:
- Step classification (fine-grained, 13 classes)
- Stage classification (coarse-grained, 3 classes)

Features for Paper:
- Multiple ablation studies (visual tokens, frames, LoRA rank, etc.)
- Detailed metrics (Accuracy, F1, Precision, Recall)
- Resource tracking (GPU memory, inference time, training time)
- Per-class performance analysis
- Confusion matrix generation
- Statistical significance testing

Usage:
------
# Standard training
python train_surgical_vlm_ablation.py \\
    --train_data /mnt/share/ali/VLM_Project/hospital_data/train.json \\
    --test_data /mnt/share/ali/VLM_Project/hospital_data/test.json \\
    --output_dir /mnt/share/ali/VLM_Project/checkpoints

# Run ablation study
python train_surgical_vlm_ablation.py \\
    --ablation visual_tokens \\
    --ablation_values 3 5 7 10 14 28 \\
    --epochs 10
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import math
import time
import random
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    SiglipImageProcessor,
    SiglipVisionModel,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Data paths
    train_data: str = "/mnt/share/ali/VLM_Project/hospital_data/train.json"
    test_data: str = "/mnt/share/ali/VLM_Project/hospital_data/test.json"
    output_dir: str = "/mnt/share/ali/VLM_Project/checkpoints"
    
    # Model paths
    vision_model_path: str = "/mnt/share/ali/VLMs/hf_cache/hub/siglip2-model/"
    language_model_path: str = "/mnt/share/ali/VLMs/hf_cache/hub/Qwen3-0.6B/"
    
    # Tasks (classification only)
    target_tasks: List[str] = field(default_factory=lambda: [
        "step_classification",
        "stage_classification"
    ])
    
    # Training hyperparameters
    epochs: int = 10
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Sequence settings
    max_seq_len: int = 512  # Keep same as working code
    num_frames: int = 8
    
    # Visual token settings
    visual_tokens: int = 5
    
    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Model settings
    freeze_vision: bool = True
    use_4bit: bool = True
    use_bf16: bool = True
    
    # Evaluation
    eval_every_epochs: int = 0  # 0 = only final epoch, N = every N epochs
    
    # Hardware
    num_workers: int = 2
    seed: int = 42


# ==============================================================================
# TASK TOKENS
# ==============================================================================

TASK_TOKENS = [
    "<STEP_CLASS>",
    "<STAGE_CLASS>",
    "<COMPLETION>",
    "<DESCRIPTION>",
]


# ==============================================================================
# UTILITIES
# ==============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_gpu_memory_peak_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ==============================================================================
# METRICS
# ==============================================================================

def compute_classification_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Returns:
        Dictionary with accuracy, precision, recall, f1, per-class metrics, confusion matrix
    """
    # Normalize
    preds = [p.strip().lower() for p in predictions]
    gts = [g.strip().lower() for g in ground_truths]
    
    # Overall accuracy
    correct = sum(1 for p, g in zip(preds, gts) if p == g)
    accuracy = correct / len(preds) if preds else 0.0
    
    # Get all unique classes (from ground truth)
    all_classes = sorted(set(gts))
    num_classes = len(all_classes)
    
    # Per-class metrics
    per_class = {}
    class_tp, class_fp, class_fn = {}, {}, {}
    
    for cls in all_classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds, gts) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds, gts) if p != cls and g == cls)
        tn = sum(1 for p, g in zip(preds, gts) if p != cls and g != cls)
        
        class_tp[cls] = tp
        class_fp[cls] = fp
        class_fn[cls] = fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    # Macro averages (average across classes)
    macro_precision = np.mean([m["precision"] for m in per_class.values()])
    macro_recall = np.mean([m["recall"] for m in per_class.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class.values()])
    
    # Weighted averages (weighted by support)
    total_support = sum(m["support"] for m in per_class.values())
    if total_support > 0:
        weighted_precision = sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
        weighted_recall = sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
        weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    # Micro averages (global TP, FP, FN)
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Confusion matrix
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, g in zip(preds, gts):
        if g in class_to_idx:
            g_idx = class_to_idx[g]
            p_idx = class_to_idx.get(p, -1)
            if p_idx >= 0:
                confusion_matrix[g_idx, p_idx] += 1
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(preds),
        "num_classes": num_classes,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": all_classes,
    }


# ==============================================================================
# DATASET
# ==============================================================================

class SurgicalVLMDataset(Dataset):
    """Dataset for surgical VLM training."""
    
    def __init__(
        self,
        data_path: str,
        image_processor: SiglipImageProcessor,
        num_frames: int = 8,
        target_tasks: Optional[List[str]] = None,
    ):
        logger.info(f"Loading dataset from: {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)
        
        if target_tasks:
            self.samples = [s for s in all_samples if s.get("main_tag") in target_tasks]
            logger.info(f"Filtered: {len(self.samples)}/{len(all_samples)} samples for tasks: {target_tasks}")
        else:
            self.samples = all_samples
            logger.info(f"Loaded all {len(self.samples)} samples")
        
        self.image_processor = image_processor
        self.num_frames = num_frames
        
        # Log statistics
        task_counts = Counter(s.get("main_tag", "unknown") for s in self.samples)
        logger.info(f"Task distribution: {dict(task_counts)}")
        
        # Log class distribution per task
        for task in target_tasks or []:
            task_samples = [s for s in self.samples if s.get("main_tag") == task]
            if task_samples:
                answers = [s["conversations"][1]["value"] for s in task_samples if len(s.get("conversations", [])) > 1]
                class_dist = Counter(answers)
                logger.info(f"  {task}: {len(class_dist)} classes, samples per class: min={min(class_dist.values())}, max={max(class_dist.values())}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_frames(self, frame_paths: List[str]) -> List[Image.Image]:
        """Load frames from paths."""
        images = []
        
        for path in frame_paths[:self.num_frames]:
            path = path.replace("\\", "/")
            try:
                if os.path.exists(path):
                    img = Image.open(path).convert("RGB")
                else:
                    img = Image.new("RGB", (384, 384), color=(128, 128, 128))
            except Exception:
                img = Image.new("RGB", (384, 384), color=(128, 128, 128))
            images.append(img)
        
        while len(images) < self.num_frames:
            images.append(images[-1].copy() if images else Image.new("RGB", (384, 384)))
        
        return images[:self.num_frames]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        frames = self._load_frames(sample.get("frames", []))
        processed = self.image_processor(images=frames, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        
        conversations = sample.get("conversations", [])
        question = conversations[0]["value"] if conversations else ""
        answer = conversations[1]["value"] if len(conversations) > 1 else ""
        
        return {
            "pixel_values": pixel_values,
            "question": question,
            "answer": answer,
            "main_tag": sample.get("main_tag", "unknown"),
            "sample_id": sample.get("id", "unknown"),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate batch samples."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    
    return {
        "pixel_values": pixel_values,
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "main_tags": [b["main_tag"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
    }


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
    """Build tokenized batch with label masking."""
    input_ids_list = []
    labels_list = []
    
    for q, a in zip(questions, answers):
        prompt = f"{q}\nAnswer:"
        answer_text = f" {a}{tokenizer.eos_token}"
        
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        
        full_ids = (prompt_ids + answer_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_len]
        
        input_ids_list.append(full_ids)
        labels_list.append(labels)
    
    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0
    
    padded_ids, padded_labels, attn_masks = [], [], []
    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_labels.append(labs + [-100] * pad_len)
        attn_masks.append([1] * len(ids) + [0] * pad_len)
    
    return (
        torch.tensor(padded_ids, dtype=torch.long, device=device),
        torch.tensor(attn_masks, dtype=torch.long, device=device),
        torch.tensor(padded_labels, dtype=torch.long, device=device),
    )


# ==============================================================================
# VISUAL PROJECTOR
# ==============================================================================

class VisualProjector(nn.Module):
    """Project visual features to LLM embedding space."""
    
    def __init__(self, vision_hidden: int, llm_hidden: int, num_visual_tokens: int = 5):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_visual_tokens, llm_hidden))
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        B, N, P, V = vision_features.shape
        flat = vision_features.view(B, N * P, V)
        projected = self.projector(flat)
        projected_t = projected.transpose(1, 2)
        pooled = F.interpolate(projected_t, size=self.num_visual_tokens, mode='linear', align_corners=True)
        result = pooled.transpose(1, 2)
        result = result + self.pos_embed.to(result.device)
        return result


# ==============================================================================
# VLM MODEL
# ==============================================================================

class SurgicalVLM(nn.Module):
    """Vision-Language Model for surgical procedures."""
    
    def __init__(
        self,
        vision_encoder: SiglipVisionModel,
        image_processor: SiglipImageProcessor,
        llm: nn.Module,
        config: TrainConfig,
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.image_processor = image_processor
        self.llm = llm
        self.config = config
        
        self.vision_hidden = vision_encoder.config.hidden_size
        self.llm_hidden = llm.config.hidden_size
        
        self.visual_projector = VisualProjector(
            vision_hidden=self.vision_hidden,
            llm_hidden=self.llm_hidden,
            num_visual_tokens=config.visual_tokens,
        )
        
        if config.freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            self.vision_encoder.eval()
    
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = pixel_values.shape
        flat = pixel_values.view(B * N, C, H, W)
        
        with torch.no_grad() if self.config.freeze_vision else torch.enable_grad():
            outputs = self.vision_encoder(pixel_values=flat)
            features = outputs.last_hidden_state
        
        _, P, V = features.shape
        return features.view(B, N, P, V)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = input_ids.device
        
        vision_features = self.encode_images(pixel_values)
        visual_embeds = self.visual_projector(vision_features)
        visual_embeds = visual_embeds.to(self.llm.dtype)
        
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        full_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        B, K, _ = visual_embeds.shape
        visual_attn = torch.ones(B, K, dtype=attention_mask.dtype, device=device)
        full_attn = torch.cat([visual_attn, attention_mask], dim=1)
        
        visual_labels = torch.full((B, K), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([visual_labels, labels], dim=1)
        
        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            labels=full_labels,
            use_cache=False,
            return_dict=True,
        )
        
        return outputs.loss
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **kwargs
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        
        vision_features = self.encode_images(pixel_values)
        visual_embeds = self.visual_projector(vision_features)
        visual_embeds = visual_embeds.to(self.llm.dtype)
        
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        full_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        B, K, _ = visual_embeds.shape
        visual_attn = torch.ones(B, K, dtype=attention_mask.dtype, device=device)
        full_attn = torch.cat([visual_attn, attention_mask], dim=1)
        
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
    """Load all model components."""
    
    logger.info(f"Loading tokenizer: {config.language_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.language_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    added = tokenizer.add_special_tokens({"additional_special_tokens": TASK_TOKENS})
    logger.info(f"Added {added} task tokens, vocab size: {len(tokenizer)}")
    
    compute_dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using dtype: {compute_dtype}")
    
    if config.use_4bit:
        logger.info("Loading LLM with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        llm = AutoModelForCausalLM.from_pretrained(
            config.language_model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map=None,
        )
        llm = prepare_model_for_kbit_training(llm)
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            config.language_model_path,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
    
    llm.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Applying LoRA (r={config.lora_r}, alpha={config.lora_alpha})...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    llm = get_peft_model(llm, lora_config)
    llm.config.use_cache = False
    llm.to(device)
    llm.print_trainable_parameters()
    
    logger.info(f"Loading SigLip: {config.vision_model_path}")
    image_processor = SiglipImageProcessor.from_pretrained(config.vision_model_path)
    vision_encoder = SiglipVisionModel.from_pretrained(config.vision_model_path)
    vision_encoder.to(device).eval()
    
    return tokenizer, llm, image_processor, vision_encoder, compute_dtype


# ==============================================================================
# CHECKPOINT
# ==============================================================================

def save_checkpoint(model, tokenizer, config, output_dir, name, epoch=0, step=0, metrics=None):
    """Save checkpoint."""
    ckpt_dir = output_dir / f"checkpoint_{name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")
    model.llm.save_pretrained(ckpt_dir / "lora_weights")
    torch.save(model.visual_projector.state_dict(), ckpt_dir / "visual_projector.pt")
    
    # Save config (convert non-serializable fields)
    config_dict = asdict(config)
    
    info = {
        "config": config_dict,
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.now().isoformat(),
    }
    with open(ckpt_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Saved checkpoint: {ckpt_dir}")


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(
    model: SurgicalVLM,
    tokenizer,
    eval_samples: List[Dict],
    config: TrainConfig,
    device: torch.device,
    measure_inference_time: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive evaluation with detailed metrics.
    """
    model.eval()
    
    # Group by task
    task_predictions = defaultdict(list)
    task_ground_truths = defaultdict(list)
    inference_times = []
    
    for sample in tqdm(eval_samples, desc="Evaluating", leave=False):
        try:
            # Load frames
            frames = []
            for path in sample.get("frames", [])[:config.num_frames]:
                path = path.replace("\\", "/")
                if os.path.exists(path):
                    frames.append(Image.open(path).convert("RGB"))
                else:
                    frames.append(Image.new("RGB", (384, 384), (128, 128, 128)))
            while len(frames) < config.num_frames:
                frames.append(frames[-1] if frames else Image.new("RGB", (384, 384)))
            
            # Process
            processed = model.image_processor(images=frames, return_tensors="pt")
            pixel_values = processed["pixel_values"].unsqueeze(0).to(device)
            
            # Get Q&A
            convs = sample.get("conversations", [])
            question = convs[0]["value"] if convs else ""
            gt = convs[1]["value"] if len(convs) > 1 else ""
            task = sample.get("main_tag", "unknown")
            
            # Generate with timing
            prompt = f"{question}\nAnswer:"
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            
            if measure_inference_time:
                torch.cuda.synchronize()
                start_time = time.time()
            
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=64,
            )
            
            if measure_inference_time:
                torch.cuda.synchronize()
                inference_times.append(time.time() - start_time)
            
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if "Answer:" in pred:
                pred = pred.split("Answer:")[-1].strip()
            
            task_predictions[task].append(pred)
            task_ground_truths[task].append(gt)
            
        except Exception as e:
            logger.warning(f"Eval error: {e}")
    
    # Compute metrics per task
    results = {
        "total_samples": sum(len(v) for v in task_predictions.values()),
        "tasks": {},
    }
    
    all_preds = []
    all_gts = []
    
    for task in task_predictions.keys():
        preds = task_predictions[task]
        gts = task_ground_truths[task]
        
        all_preds.extend(preds)
        all_gts.extend(gts)
        
        metrics = compute_classification_metrics(preds, gts)
        results["tasks"][task] = metrics
        
        logger.info(f"\n[{task}]")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        logger.info(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        logger.info(f"  Classes: {metrics['num_classes']}")
    
    # Overall accuracy
    overall_correct = sum(1 for p, g in zip(all_preds, all_gts) if p.strip().lower() == g.strip().lower())
    results["overall_accuracy"] = overall_correct / len(all_preds) if all_preds else 0.0
    results["overall_correct"] = overall_correct
    
    # Inference time stats
    if inference_times:
        results["inference_time"] = {
            "mean_ms": np.mean(inference_times) * 1000,
            "std_ms": np.std(inference_times) * 1000,
            "min_ms": np.min(inference_times) * 1000,
            "max_ms": np.max(inference_times) * 1000,
            "samples_per_second": 1.0 / np.mean(inference_times),
        }
        logger.info(f"\nInference time: {results['inference_time']['mean_ms']:.1f} ± {results['inference_time']['std_ms']:.1f} ms/sample")
    
    model.train()
    return results


# ==============================================================================
# TRAINING
# ==============================================================================

def train_single_run(config: TrainConfig, run_name: str = None) -> Dict[str, Any]:
    """
    Single training run with comprehensive logging.
    
    Returns:
        Dictionary with training history and final metrics
    """
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Reset GPU stats
    reset_gpu_memory_stats()
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        output_dir = Path(config.output_dir) / f"{run_name}_{timestamp}"
    else:
        output_dir = Path(config.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    
    # Load models
    load_start = time.time()
    tokenizer, llm, image_processor, vision_encoder, compute_dtype = load_models(config, device)
    load_time = time.time() - load_start
    
    # Create VLM
    model = SurgicalVLM(
        vision_encoder=vision_encoder,
        image_processor=image_processor,
        llm=llm,
        config=config,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Dataset
    train_dataset = SurgicalVLMDataset(
        config.train_data,
        image_processor,
        num_frames=config.num_frames,
        target_tasks=config.target_tasks,
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
    
    # Eval samples
    eval_samples = []
    if config.test_data and os.path.exists(config.test_data):
        with open(config.test_data, "r", encoding="utf-8") as f:
            all_eval = json.load(f)
        eval_samples = [s for s in all_eval if s.get("main_tag") in config.target_tasks]
        logger.info(f"Eval samples: {len(eval_samples)}")
    
    # Optimizer
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Scheduler
    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = int(config.warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # AMP
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and not config.use_bf16))
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_samples)}")
    logger.info(f"Tasks: {config.target_tasks}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size} x {config.grad_accum} = {config.batch_size * config.grad_accum}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Visual tokens: {config.visual_tokens}")
    logger.info(f"Num frames: {config.num_frames}")
    logger.info(f"LR: {config.learning_rate}")
    logger.info(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    logger.info("=" * 70)
    
    # Training history
    history = {
        "config": asdict(config),
        "train_loss": [],
        "eval_metrics": [],
        "resource_usage": {},
    }
    
    best_accuracy = 0.0
    best_epoch = 0
    global_step = 0
    train_start_time = time.time()
    
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]
            
            input_ids, attention_mask, labels = build_batch_text(
                tokenizer, questions, answers, config.max_seq_len, device
            )
            
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(pixel_values, input_ids, attention_mask, labels)
                loss = loss / config.grad_accum
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += loss.item() * config.grad_accum
            epoch_batches += 1
            
            if (batch_idx + 1) % config.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params_list, config.max_grad_norm)
                
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / epoch_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        
        history["train_loss"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "time_seconds": epoch_time,
        })
        
        logger.info(f"\nEpoch {epoch+1} completed in {format_time(epoch_time)}, avg loss: {avg_loss:.4f}")
        
        # Evaluation (if enabled during training)
        if eval_samples and config.eval_every_epochs > 0 and (epoch + 1) % config.eval_every_epochs == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"EVALUATION - Epoch {epoch+1}")
            logger.info(f"{'='*50}")
            
            metrics = evaluate(
                model, tokenizer, eval_samples, config, device,
                measure_inference_time=False
            )
            
            metrics["epoch"] = epoch + 1
            metrics["train_loss"] = avg_loss
            history["eval_metrics"].append(metrics)
            
            logger.info(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
            
            # Check for best model
            if metrics["overall_accuracy"] > best_accuracy:
                best_accuracy = metrics["overall_accuracy"]
                best_epoch = epoch + 1
                save_checkpoint(model, tokenizer, config, output_dir, "best", epoch + 1, global_step, metrics)
                logger.info(f"*** New best model! ***")
            
            model.train()
    
    total_train_time = time.time() - train_start_time
    
    # Final evaluation with inference time measurement
    logger.info(f"\n{'='*50}")
    logger.info("FINAL EVALUATION")
    logger.info(f"{'='*50}")
    
    final_metrics = evaluate(model, tokenizer, eval_samples, config, device, measure_inference_time=True)
    
    # Resource usage
    history["resource_usage"] = {
        "gpu_memory_peak_mb": get_gpu_memory_peak_mb(),
        "total_train_time_seconds": total_train_time,
        "time_per_epoch_seconds": total_train_time / config.epochs,
        "model_load_time_seconds": load_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params,
    }
    
    if "inference_time" in final_metrics:
        history["resource_usage"]["inference_time_ms"] = final_metrics["inference_time"]["mean_ms"]
        history["resource_usage"]["samples_per_second"] = final_metrics["inference_time"]["samples_per_second"]
    
    # Save final checkpoint and history
    save_checkpoint(model, tokenizer, config, output_dir, "final", config.epochs, global_step, final_metrics)
    
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total training time: {format_time(total_train_time)}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Peak GPU memory: {history['resource_usage']['gpu_memory_peak_mb']:.0f} MB")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    return {
        "output_dir": str(output_dir),
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "final_metrics": final_metrics,
        "history": history,
    }


# ==============================================================================
# ABLATION STUDIES
# ==============================================================================

def run_ablation_study(
    base_config: TrainConfig,
    ablation_param: str,
    ablation_values: List[Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run ablation study over a parameter.
    
    Args:
        base_config: Base training configuration
        ablation_param: Parameter to vary (e.g., "visual_tokens", "num_frames", "lora_r")
        ablation_values: Values to test
        output_dir: Output directory for results
    
    Returns:
        Dictionary with ablation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"ABLATION STUDY: {ablation_param}")
    logger.info(f"Values to test: {ablation_values}")
    logger.info("=" * 70)
    
    results = {
        "ablation_param": ablation_param,
        "ablation_values": ablation_values,
        "runs": [],
    }
    
    for value in ablation_values:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {ablation_param} = {value}")
        logger.info(f"{'='*50}")
        
        # Create config copy with modified parameter
        config = TrainConfig(**asdict(base_config))
        setattr(config, ablation_param, value)
        config.output_dir = str(output_dir)
        
        # Run training
        run_name = f"ablation_{ablation_param}_{value}"
        
        try:
            run_result = train_single_run(config, run_name=run_name)
            
            results["runs"].append({
                "value": value,
                "best_accuracy": run_result["best_accuracy"],
                "final_metrics": run_result["final_metrics"],
                "resource_usage": run_result["history"]["resource_usage"],
                "output_dir": run_result["output_dir"],
                "status": "success",
            })
        except Exception as e:
            logger.error(f"Run failed for {ablation_param}={value}: {e}")
            results["runs"].append({
                "value": value,
                "status": "failed",
                "error": str(e),
            })
        
        # Clear GPU memory between runs
        reset_gpu_memory_stats()
    
    # Save ablation results
    with open(output_dir / f"ablation_{ablation_param}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"ABLATION RESULTS: {ablation_param}")
    logger.info("=" * 70)
    
    print(f"\n{'Value':<10} {'Step Acc':<12} {'Stage Acc':<12} {'Overall':<12} {'GPU (MB)':<12} {'Time/Epoch':<12}")
    print("-" * 70)
    
    for run in results["runs"]:
        if run["status"] == "success":
            step_acc = run["final_metrics"]["tasks"].get("step_classification", {}).get("accuracy", 0)
            stage_acc = run["final_metrics"]["tasks"].get("stage_classification", {}).get("accuracy", 0)
            overall = run["best_accuracy"]
            gpu_mb = run["resource_usage"].get("gpu_memory_peak_mb", 0)
            time_epoch = run["resource_usage"].get("time_per_epoch_seconds", 0)
            
            print(f"{run['value']:<10} {step_acc:<12.4f} {stage_acc:<12.4f} {overall:<12.4f} {gpu_mb:<12.0f} {format_time(time_epoch):<12}")
        else:
            print(f"{run['value']:<10} FAILED: {run.get('error', 'Unknown error')[:40]}")
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Surgical VLM with Ablation Studies")
    
    # Data paths
    parser.add_argument("--train_data", type=str, default="/mnt/share/ali/VLM_Project/hospital_data/train.json")
    parser.add_argument("--test_data", type=str, default="/mnt/share/ali/VLM_Project/hospital_data/test.json")
    parser.add_argument("--output_dir", type=str, default="/mnt/share/ali/VLM_Project/checkpoints")
    
    # Model paths
    parser.add_argument("--vision_model", type=str, default="/mnt/share/ali/VLMs/hf_cache/hub/siglip2-model/")
    parser.add_argument("--language_model", type=str, default="/mnt/share/ali/VLMs/hf_cache/hub/Qwen3-0.6B/")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--visual_tokens", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=0, 
                        help="Evaluate every N epochs (0=final only)")
    
    # Ablation study
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["visual_tokens", "num_frames", "lora_r", "learning_rate", "epochs"],
                        help="Parameter to ablate")
    parser.add_argument("--ablation_values", nargs="+", type=str, default=None,
                        help="Values to test in ablation")
    
    args = parser.parse_args()
    
    # Create base config
    config = TrainConfig(
        train_data=args.train_data,
        test_data=args.test_data,
        output_dir=args.output_dir,
        vision_model_path=args.vision_model,
        language_model_path=args.language_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        visual_tokens=args.visual_tokens,
        num_frames=args.num_frames,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        eval_every_epochs=args.eval_every,
    )
    
    if args.ablation:
        # Run ablation study
        if args.ablation_values is None:
            # Default ablation values
            default_values = {
                "visual_tokens": [3, 5, 7, 10, 14, 28],
                "num_frames": [1, 2, 4, 8],
                "lora_r": [8, 16, 32, 64],
                "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3],
                "epochs": [5, 10, 15, 20],
            }
            ablation_values = default_values.get(args.ablation, [])
        else:
            # Parse values (convert to appropriate type)
            if args.ablation in ["visual_tokens", "num_frames", "lora_r", "epochs"]:
                ablation_values = [int(v) for v in args.ablation_values]
            elif args.ablation == "learning_rate":
                ablation_values = [float(v) for v in args.ablation_values]
            else:
                ablation_values = args.ablation_values
        
        run_ablation_study(config, args.ablation, ablation_values, args.output_dir)
    else:
        # Single training run
        train_single_run(config)


if __name__ == "__main__":
    main()