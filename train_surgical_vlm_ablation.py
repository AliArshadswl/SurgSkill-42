#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SURGICAL VLM TRAINING - ABLATION: VISUAL TOKEN RESAMPLING METHODS (5 RUNS)
================================================================================

This script is based on your optimized training code, with additions:

1) Visual token resampling ablations (replace interpolation baseline):
   - interpolate       : 1D linear interpolation over token axis (your current)
   - adaptive_avg      : AdaptiveAvgPool1d over token axis (cheap baseline)
   - adaptive_max      : AdaptiveMaxPool1d over token axis (cheap baseline)
   - frame_interpolate : resample per-frame tokens to K/N, then concat (keeps frame boundary)
   - perceiver         : Perceiver-style latent resampler (content-aware, recommended)

2) Multi-run experiment:
   - Run N runs (default 5) with different seeds
   - Collect Accuracy, Macro-F1, Cohen's Kappa, inference speed (samples/sec)

3) Metrics:
   - Builds label vocab per task from TRAIN split answers (string-normalized)
   - Converts generated prediction strings into label IDs for metrics
   - Reports overall + per-task metrics

Usage examples
--------------
# Run 5 runs for one method (interpolate):
python train_surgical_vlm_ablation.py --resampler interpolate --num_runs 5

# Run 5 runs for perceiver resampler:
python train_surgical_vlm_ablation.py --resampler perceiver --num_runs 5

# Run all resamplers (each with 5 runs):
python train_surgical_vlm_ablation.py --resampler all --num_runs 5

Notes
-----
- This script assumes your answers are discrete class strings for each task
  (e.g., "Step 3", "Stage: Preparation", etc.). It uses exact normalized match.
- If your labels are more complex text, you can replace the mapping logic with
  regex or semantic matching.

================================================================================
"""

import os
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
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict
from datetime import datetime

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

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(output_dir: Optional[Path] = None, log_level: int = logging.INFO):
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "training.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logging()


# ==============================================================================
# CONFIG
# ==============================================================================

TASK_TOKENS = [
    "<STEP_CLASS>",
    "<STAGE_CLASS>",
    "<COMPLETION>",
    "<DESCRIPTION>",
    "<IMAGE>",
    "<VIDEO>",
]


@dataclass
class OptimizedConfig:
    # Data paths
    train_data: str = "/mnt/share/ali/VLM_Project/hospital_data/train.json"
    val_data: Optional[str] = None
    test_data: str = "/mnt/share/ali/VLM_Project/hospital_data/test.json"
    output_dir: str = "/mnt/share/ali/VLM_Project/checkpoints_ablation"

    # Model paths
    vision_model_path: str = "/mnt/share/ali/VLMs/hf_cache/hub/siglip2-model/"
    language_model_path: str = "/mnt/share/ali/VLMs/hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/"

    # Tasks
    target_tasks: Optional[List[str]] = field(default_factory=lambda: [
        "step_classification",
        "stage_classification"
    ])

    # Training
    visual_tokens: int = 1024   # fix: use 1024 consistently
    epochs: int = 20
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Sequence
    max_seq_len: int = 512
    num_frames: int = 8

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Model
    freeze_vision: bool = True
    use_4bit: bool = True
    use_bf16: bool = True

    # Ablation: visual resampler method
    # choices: interpolate | adaptive_avg | adaptive_max | frame_interpolate | perceiver
    resampler: str = "interpolate"
    perceiver_layers: int = 1
    perceiver_heads: int = 8
    perceiver_dropout: float = 0.0

    # Logging & eval
    log_every_steps: int = 10
    eval_every_steps: int = 500
    save_every_steps: int = 500
    eval_samples: int = 200  # recommend >100 for stable metrics

    # Split
    val_split_ratio: float = 0.1
    val_split_seed: int = 42

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001

    # Hardware
    num_workers: int = 2

    # Seed (will be overwritten per run)
    seed: int = 42

    # Inference speed measurement
    speed_warmup: int = 10
    speed_samples: int = 100
    max_new_tokens_eval: int = 16  # for classification, short generation is enough


# ==============================================================================
# UTILS
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_memory_info() -> Dict[str, float]:
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
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def normalize_text(s: str) -> str:
    """Normalize label/prediction strings for robust exact-match classification."""
    if s is None:
        return ""
    s = s.strip().lower()
    # collapse whitespace
    s = " ".join(s.split())
    # remove trailing punctuation
    s = s.strip(" \t\n\r.,;:!?'\"")
    return s


# ==============================================================================
# DATASET
# ==============================================================================

class DatasetAnalyzer:
    def __init__(self, samples: List[Dict], name: str = "Dataset"):
        self.samples = samples
        self.name = name
        self.stats = self._compute_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        stats = {
            "total_samples": len(self.samples),
            "task_distribution": Counter(),
            "frames_per_sample": [],
            "unique_answers_by_task": defaultdict(set),
        }
        for sample in self.samples:
            task = sample.get("main_tag", "unknown")
            stats["task_distribution"][task] += 1
            frames = sample.get("frames", [])
            stats["frames_per_sample"].append(len(frames))

            convs = sample.get("conversations", [])
            if len(convs) >= 2:
                ans = normalize_text(convs[1].get("value", ""))
                stats["unique_answers_by_task"][task].add(ans)
        return stats

    def print_report(self, log: logging.Logger) -> None:
        log.info("=" * 70)
        log.info(f"DATASET: {self.name}")
        log.info("=" * 70)
        log.info(f"Total samples: {self.stats['total_samples']}")
        log.info("\nTask Distribution:")
        for task, count in sorted(self.stats["task_distribution"].items()):
            pct = 100 * count / max(1, self.stats["total_samples"])
            log.info(f"  {task:25s}: {count:6d} ({pct:5.1f}%)")
        if self.stats["frames_per_sample"]:
            frames = self.stats["frames_per_sample"]
            log.info(f"\nFrames per sample: min={min(frames)}, max={max(frames)}, mean={np.mean(frames):.1f}")
        for task, uniq in self.stats["unique_answers_by_task"].items():
            log.info(f"Unique labels ({task}): {len(uniq)}")
        log.info("=" * 70)


class SurgicalVLMDataset(Dataset):
    def __init__(self, samples: List[Dict], image_processor: SiglipImageProcessor, num_frames: int = 8):
        self.samples = self._validate_samples(samples)
        self.image_processor = image_processor
        self.num_frames = num_frames

    def _validate_samples(self, samples: List[Dict]) -> List[Dict]:
        valid = []
        for s in samples:
            has_frames = "frames" in s and len(s["frames"]) > 0
            has_convs = "conversations" in s and len(s["conversations"]) >= 2
            if has_frames and has_convs:
                valid.append(s)
        if len(valid) < len(samples):
            logger.warning(f"Filtered {len(samples) - len(valid)} invalid samples")
        return valid

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, frame_paths: List[str]) -> List[Image.Image]:
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
        pixel_values = processed["pixel_values"]  # [N, C, H, W]

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
    pixel_values = torch.stack([b["pixel_values"] for b in batch])  # [B, N, C, H, W]
    return {
        "pixel_values": pixel_values,
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "main_tags": [b["main_tag"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
    }


def load_and_split_data(config: OptimizedConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    logger.info(f"Loading training data: {config.train_data}")
    with open(config.train_data, "r", encoding="utf-8") as f:
        all_train = json.load(f)

    if config.target_tasks:
        train_samples = [s for s in all_train if s.get("main_tag") in config.target_tasks]
        logger.info(f"Filtered to {len(train_samples)} samples for tasks: {config.target_tasks}")
    else:
        train_samples = all_train

    if config.val_data and os.path.exists(config.val_data):
        logger.info(f"Loading validation data: {config.val_data}")
        with open(config.val_data, "r", encoding="utf-8") as f:
            all_val = json.load(f)
        if config.target_tasks:
            val_samples = [s for s in all_val if s.get("main_tag") in config.target_tasks]
        else:
            val_samples = all_val
        logger.info(f"Loaded {len(val_samples)} validation samples from file")
    else:
        logger.info("Creating validation split from training data")
        rng = random.Random(config.val_split_seed)
        indices = list(range(len(train_samples)))
        rng.shuffle(indices)
        val_size = int(len(train_samples) * config.val_split_ratio)
        val_indices = set(indices[:val_size])
        val_samples = [train_samples[i] for i in val_indices]
        train_samples = [train_samples[i] for i in range(len(train_samples)) if i not in val_indices]
        logger.info(f"  - Training samples: {len(train_samples)}")
        logger.info(f"  - Validation samples: {len(val_samples)}")

    test_samples = []
    if config.test_data and os.path.exists(config.test_data):
        logger.info(f"Loading test data: {config.test_data}")
        with open(config.test_data, "r", encoding="utf-8") as f:
            all_test = json.load(f)
        if config.target_tasks:
            test_samples = [s for s in all_test if s.get("main_tag") in config.target_tasks]
        else:
            test_samples = all_test
        logger.info(f"Loaded {len(test_samples)} test samples (held out for final evaluation)")
    else:
        logger.warning("No test data file found - final evaluation will be skipped")

    DatasetAnalyzer(train_samples, "TRAIN").print_report(logger)
    DatasetAnalyzer(val_samples, "VAL").print_report(logger)
    if test_samples:
        DatasetAnalyzer(test_samples, "TEST").print_report(logger)

    return train_samples, val_samples, test_samples


# ==============================================================================
# LABEL VOCAB (for classification metrics)
# ==============================================================================

def build_label_vocab(train_samples: List[Dict], tasks: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Build task->(label_string->id) vocab from TRAIN split answers.
    """
    vocab: Dict[str, Dict[str, int]] = {}
    for t in tasks:
        labels = []
        for s in train_samples:
            if s.get("main_tag") != t:
                continue
            convs = s.get("conversations", [])
            if len(convs) < 2:
                continue
            labels.append(normalize_text(convs[1].get("value", "")))
        uniq = sorted(set([x for x in labels if x != ""]))
        vocab[t] = {lab: i for i, lab in enumerate(uniq)}
        logger.info(f"[LabelVocab] task={t} | classes={len(uniq)}")
    return vocab


# ==============================================================================
# TEXT PROCESSING
# ==============================================================================

def build_batch_text(tokenizer, questions: List[str], answers: List[str], max_seq_len: int, device: torch.device):
    input_ids_list, labels_list = [], []

    for question, answer in zip(questions, answers):
        prompt = f"{question}\nAnswer:"
        answer_text = f" {answer}{tokenizer.eos_token}"

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
# METRICS (no sklearn dependency)
# ==============================================================================

def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def cohen_kappa_from_cm(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = float((row_marg * col_marg).sum())
    denom = (1.0 - pe)
    return float((po - pe) / denom) if denom > 1e-12 else 0.0


# ==============================================================================
# VISUAL RESAMPLERS (ABLATIONS)
# ==============================================================================

class MLPProjector(nn.Module):
    """Token-wise projection V->H."""
    def __init__(self, vision_hidden: int, llm_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverResampler(nn.Module):
    """
    Content-aware token bottleneck:
    - K learnable latent tokens cross-attend to T visual tokens.

    Input:  visual tokens [B, T, H]
    Output: resampled     [B, K, H]
    """
    def __init__(self, hidden: int, num_latents: int, num_heads: int = 8, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden) * 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True),
                "ln1": nn.LayerNorm(hidden),
                "ff": nn.Sequential(
                    nn.Linear(hidden, 4 * hidden),
                    nn.GELU(),
                    nn.Linear(4 * hidden, hidden),
                ),
                "ln2": nn.LayerNorm(hidden),
            }))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, H = tokens.shape
        lat = self.latents.expand(B, -1, -1)  # [B, K, H]
        for layer in self.layers:
            # Cross-attention: queries=latents, keys/values=tokens
            attn_out, _ = layer["attn"](query=layer["ln1"](lat), key=tokens, value=tokens, need_weights=False)
            lat = lat + attn_out
            lat = lat + layer["ff"](layer["ln2"](lat))
        return lat


class VisualProjectorAblation(nn.Module):
    """
    Implements:
    1) Flatten [B, N, P, V] -> [B, T, V] where T=N*P
    2) Project token-wise V->H using MLP
    3) Resample token axis to K using selected resampler
    4) Add learned pos embedding for K tokens
    """
    def __init__(self, vision_hidden: int, llm_hidden: int, num_frames: int, num_visual_tokens: int, resampler: str,
                 perceiver_layers: int = 1, perceiver_heads: int = 8, perceiver_dropout: float = 0.0):
        super().__init__()
        self.num_frames = num_frames
        self.num_visual_tokens = num_visual_tokens
        self.resampler = resampler

        self.project = MLPProjector(vision_hidden, llm_hidden)

        if resampler == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool1d(num_visual_tokens)
        elif resampler == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool1d(num_visual_tokens)
        elif resampler == "perceiver":
            self.pool = PerceiverResampler(
                hidden=llm_hidden,
                num_latents=num_visual_tokens,
                num_heads=perceiver_heads,
                num_layers=perceiver_layers,
                dropout=perceiver_dropout,
            )
        else:
            self.pool = None  # interpolate or frame_interpolate uses functional ops

        self.pos_embed = nn.Parameter(torch.zeros(1, num_visual_tokens, llm_hidden))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        vision_features: [B, N, P, V]
        returns:         [B, K, H]
        """
        B, N, P, V = vision_features.shape
        H = self.pos_embed.shape[-1]
        K = self.num_visual_tokens

        # 1) flatten to token seq
        flat = vision_features.view(B, N * P, V)          # [B, T, V]
        proj = self.project(flat)                         # [B, T, H]

        if self.resampler == "interpolate":
            # 2) token-axis interpolation: [B, T, H] -> [B, K, H]
            proj_t = proj.transpose(1, 2)  # [B, H, T]
            pooled = F.interpolate(proj_t, size=K, mode="linear", align_corners=True)
            out = pooled.transpose(1, 2)   # [B, K, H]

        elif self.resampler in ("adaptive_avg", "adaptive_max"):
            proj_t = proj.transpose(1, 2)  # [B, H, T]
            pooled = self.pool(proj_t)     # [B, H, K]
            out = pooled.transpose(1, 2)   # [B, K, H]

        elif self.resampler == "frame_interpolate":
            # keep frame boundary: per-frame interpolate to k_per_frame then concat
            assert N > 0
            kpf = K // N
            remainder = K - kpf * N
            outs = []
            for i in range(N):
                # select tokens for this frame [B, P, H]
                frame_tokens = proj[:, i * P:(i + 1) * P, :]
                # per-frame interpolate P -> kpf (and distribute remainder to first frames)
                this_k = kpf + (1 if i < remainder else 0)
                ft = frame_tokens.transpose(1, 2)  # [B, H, P]
                pooled = F.interpolate(ft, size=this_k, mode="linear", align_corners=True)
                outs.append(pooled.transpose(1, 2))  # [B, this_k, H]
            out = torch.cat(outs, dim=1)  # [B, K, H] (exact)

        elif self.resampler == "perceiver":
            out = self.pool(proj)  # [B, K, H]

        else:
            raise ValueError(f"Unknown resampler: {self.resampler}")

        out = out + self.pos_embed.to(out.device)
        return out


# ==============================================================================
# VLM
# ==============================================================================

class SurgicalVLM(nn.Module):
    def __init__(self, vision_encoder: SiglipVisionModel, image_processor: SiglipImageProcessor, llm: nn.Module, config: OptimizedConfig):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.image_processor = image_processor
        self.llm = llm
        self.config = config

        self.vision_hidden = vision_encoder.config.hidden_size
        self.llm_hidden = llm.config.hidden_size

        logger.info(f"Vision hidden: {self.vision_hidden}, LLM hidden: {self.llm_hidden}")
        logger.info(f"Visual tokens K: {config.visual_tokens} | Resampler: {config.resampler}")

        self.visual_projector = VisualProjectorAblation(
            vision_hidden=self.vision_hidden,
            llm_hidden=self.llm_hidden,
            num_frames=config.num_frames,
            num_visual_tokens=config.visual_tokens,
            resampler=config.resampler,
            perceiver_layers=config.perceiver_layers,
            perceiver_heads=config.perceiver_heads,
            perceiver_dropout=config.perceiver_dropout,
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
            features = outputs.last_hidden_state  # [B*N, P, V]
        _, P, V = features.shape
        return features.view(B, N, P, V)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        vision_features = self.encode_images(pixel_values)
        visual_embeds = self.visual_projector(vision_features).to(self.llm.dtype)  # [B, K, H]

        text_embeds = self.llm.get_input_embeddings()(input_ids)                   # [B, L, H]
        full_embeds = torch.cat([visual_embeds, text_embeds], dim=1)               # [B, K+L, H]

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
    def generate(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 32, **kwargs) -> torch.Tensor:
        self.eval()
        device = input_ids.device

        vision_features = self.encode_images(pixel_values)
        visual_embeds = self.visual_projector(vision_features).to(self.llm.dtype)

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

def load_models(config: OptimizedConfig, device: torch.device):
    logger.info("=" * 70)
    logger.info("LOADING MODELS")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(config.language_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": TASK_TOKENS})
    logger.info(f"Added {num_added} special tokens, vocab size: {len(tokenizer)}")

    if config.use_bf16 and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        logger.info("Using bfloat16 precision")
    else:
        compute_dtype = torch.float16
        logger.info("Using float16 precision")

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

    logger.info(f"GPU memory: {get_gpu_memory_info()}")
    return tokenizer, llm, image_processor, vision_encoder, compute_dtype


# ==============================================================================
# CHECKPOINTS
# ==============================================================================

def save_checkpoint(model, tokenizer, config, output_dir: Path, name: str, step: int = 0, metrics: Optional[Dict] = None):
    ckpt_dir = output_dir / f"checkpoint_{name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")
    model.llm.save_pretrained(ckpt_dir / "lora_weights")
    torch.save(model.visual_projector.state_dict(), ckpt_dir / "visual_projector.pt")
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


def load_checkpoint(model, checkpoint_path: Path) -> Dict:
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    projector_path = checkpoint_path / "visual_projector.pt"
    if projector_path.exists():
        model.visual_projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
    info_path = checkpoint_path / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return {}


# ==============================================================================
# EVALUATION + SPEED
# ==============================================================================

@torch.no_grad()
def measure_inference_speed(model: SurgicalVLM, tokenizer, samples: List[Dict], config: OptimizedConfig, device: torch.device) -> Dict[str, float]:
    """
    Measures generation throughput on a subset of samples.
    Reports samples/sec (end-to-end per sample: vision encode + projector + generate).
    """
    model.eval()
    rng = random.Random(config.seed + 999)
    subset = samples
    if config.speed_samples > 0 and len(samples) > config.speed_samples:
        subset = rng.sample(samples, config.speed_samples)

    # warmup
    warm = subset[:min(config.speed_warmup, len(subset))]
    for s in warm:
        frames = []
        for path in s.get("frames", [])[:config.num_frames]:
            path = path.replace("\\", "/")
            frames.append(Image.open(path).convert("RGB") if os.path.exists(path) else Image.new("RGB", (384, 384), (128, 128, 128)))
        while len(frames) < config.num_frames:
            frames.append(frames[-1] if frames else Image.new("RGB", (384, 384)))
        processed = model.image_processor(images=frames[:config.num_frames], return_tensors="pt")
        pixel_values = processed["pixel_values"].unsqueeze(0).to(device)
        prompt = f"{s.get('conversations',[{'value':''}])[0].get('value','')}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        _ = model.generate(pixel_values, enc["input_ids"], enc["attention_mask"], max_new_tokens=config.max_new_tokens_eval)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # timed
    t0 = time.time()
    count = 0
    for s in subset:
        frames = []
        for path in s.get("frames", [])[:config.num_frames]:
            path = path.replace("\\", "/")
            frames.append(Image.open(path).convert("RGB") if os.path.exists(path) else Image.new("RGB", (384, 384), (128, 128, 128)))
        while len(frames) < config.num_frames:
            frames.append(frames[-1] if frames else Image.new("RGB", (384, 384)))
        processed = model.image_processor(images=frames[:config.num_frames], return_tensors="pt")
        pixel_values = processed["pixel_values"].unsqueeze(0).to(device)
        prompt = f"{s.get('conversations',[{'value':''}])[0].get('value','')}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        _ = model.generate(pixel_values, enc["input_ids"], enc["attention_mask"], max_new_tokens=config.max_new_tokens_eval)
        count += 1
        if device.type == "cuda":
            torch.cuda.synchronize()

    t1 = time.time()
    dt = max(1e-9, t1 - t0)
    return {"samples_per_sec": float(count / dt), "seconds_total": float(dt), "samples_measured": int(count)}


@torch.no_grad()
def evaluate_classification(model: SurgicalVLM, tokenizer, samples: List[Dict], config: OptimizedConfig, device: torch.device,
                            label_vocab: Dict[str, Dict[str, int]], max_samples: int = -1, split_name: str = "test") -> Dict[str, Any]:
    """
    Evaluates by generating a short answer and mapping it to a class ID using label_vocab.
    Reports overall + per-task Accuracy, Macro-F1, Cohen's Kappa.
    """
    model.eval()
    rng = random.Random(config.seed + 12345)
    eval_subset = samples
    if max_samples > 0 and len(samples) > max_samples:
        eval_subset = rng.sample(samples, max_samples)

    per_task_true: Dict[str, List[int]] = defaultdict(list)
    per_task_pred: Dict[str, List[int]] = defaultdict(list)

    unknown_count = 0
    total_count = 0

    for s in tqdm(eval_subset, desc=f"Evaluating ({split_name})", leave=False):
        task = s.get("main_tag", "unknown")
        if task not in label_vocab:
            continue

        # ground-truth label
        convs = s.get("conversations", [])
        if len(convs) < 2:
            continue
        gt = normalize_text(convs[1].get("value", ""))
        if gt not in label_vocab[task]:
            # unseen label in train vocab; skip or count as unknown
            continue
        y_true = label_vocab[task][gt]

        # load frames
        frames = []
        for path in s.get("frames", [])[:config.num_frames]:
            path = path.replace("\\", "/")
            frames.append(Image.open(path).convert("RGB") if os.path.exists(path) else Image.new("RGB", (384, 384), (128, 128, 128)))
        while len(frames) < config.num_frames:
            frames.append(frames[-1] if frames else Image.new("RGB", (384, 384)))
        frames = frames[:config.num_frames]

        processed = model.image_processor(images=frames, return_tensors="pt")
        pixel_values = processed["pixel_values"].unsqueeze(0).to(device)

        question = convs[0].get("value", "")
        prompt = f"{question}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        out_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=config.max_new_tokens_eval,
        )

        pred_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if "Answer:" in pred_text:
            pred_text = pred_text.split("Answer:")[-1]
        pred_text = normalize_text(pred_text)

        # map to label id
        if pred_text in label_vocab[task]:
            y_pred = label_vocab[task][pred_text]
        else:
            # unknown prediction => treat as incorrect by mapping to -1 and counting unknowns
            y_pred = -1
            unknown_count += 1

        per_task_true[task].append(y_true)
        per_task_pred[task].append(y_pred)
        total_count += 1

    # compute metrics
    metrics: Dict[str, Any] = {
        "split": split_name,
        "n_eval": total_count,
        "unknown_predictions": unknown_count,
    }

    # overall (micro over tasks)
    all_true, all_pred = [], []
    for t in per_task_true.keys():
        all_true.extend(per_task_true[t])
        all_pred.extend(per_task_pred[t])

    # for overall, handle unknown preds by counting them as always wrong:
    # build cm with an extra "unknown" class? We'll instead clamp unknown to 0 and mark wrong via accuracy,
    # and compute f1/kappa only over valid class IDs by ignoring unknown preds.
    # Reviewer-safe: report unknown rate + metrics computed on valid preds.
    valid_idx = [i for i, p in enumerate(all_pred) if p >= 0]
    if len(valid_idx) > 0:
        vt = [all_true[i] for i in valid_idx]
        vp = [all_pred[i] for i in valid_idx]
        num_classes = max(vt) + 1
        cm = confusion_matrix(vt, vp, num_classes)
        metrics["overall_macro_f1_valid"] = macro_f1_from_cm(cm)
        metrics["overall_kappa_valid"] = cohen_kappa_from_cm(cm)
        metrics["overall_valid_frac"] = float(len(valid_idx) / max(1, len(all_pred)))
    else:
        metrics["overall_macro_f1_valid"] = 0.0
        metrics["overall_kappa_valid"] = 0.0
        metrics["overall_valid_frac"] = 0.0

    # accuracy counts unknown as incorrect
    correct = sum([1 for t, p in zip(all_true, all_pred) if p == t])
    metrics["overall_accuracy"] = float(correct / max(1, len(all_true)))

    # per-task
    for task in per_task_true.keys():
        yt = per_task_true[task]
        yp = per_task_pred[task]
        correct_t = sum([1 for t, p in zip(yt, yp) if p == t])
        acc = correct_t / max(1, len(yt))

        valid = [(t, p) for t, p in zip(yt, yp) if p >= 0]
        if valid:
            vt = [t for t, _ in valid]
            vp = [p for _, p in valid]
            num_classes = max(vt) + 1
            cm = confusion_matrix(vt, vp, num_classes)
            mf1 = macro_f1_from_cm(cm)
            kap = cohen_kappa_from_cm(cm)
            vfrac = len(valid) / max(1, len(yt))
        else:
            mf1, kap, vfrac = 0.0, 0.0, 0.0

        metrics[f"{task}_accuracy"] = float(acc)
        metrics[f"{task}_macro_f1_valid"] = float(mf1)
        metrics[f"{task}_kappa_valid"] = float(kap)
        metrics[f"{task}_valid_frac"] = float(vfrac)
        metrics[f"{task}_n"] = int(len(yt))

    model.train()
    return metrics


# ==============================================================================
# TRAINING
# ==============================================================================

def train_one_run(config: OptimizedConfig) -> Tuple[str, Dict[str, Any]]:
    """
    Train one run with config.seed and config.resampler, evaluate on test set,
    and return (run_dir, metrics_dict).
    """
    global logger
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {get_gpu_memory_info()}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.resampler}_seed{config.seed}_{timestamp}"
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    train_samples, val_samples, test_samples = load_and_split_data(config)
    if len(train_samples) == 0:
        raise ValueError("No training samples!")

    tokenizer, llm, image_processor, vision_encoder, _ = load_models(config, device)

    model = SurgicalVLM(
        vision_encoder=vision_encoder,
        image_processor=image_processor,
        llm=llm,
        config=config,
    ).to(device)

    train_dataset = SurgicalVLMDataset(train_samples, image_processor=image_processor, num_frames=config.num_frames)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # label vocab from TRAIN split
    label_vocab = build_label_vocab(train_samples, config.target_tasks or [])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = int(config.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and not config.use_bf16))

    logger.info("=" * 70)
    logger.info("TRAINING CONFIG")
    logger.info("=" * 70)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Resampler: {config.resampler}")
    logger.info(f"Epochs: {config.epochs} | Batch: {config.batch_size} | Accum: {config.grad_accum}")
    logger.info(f"Visual tokens K: {config.visual_tokens}")
    logger.info(f"Total steps: {total_steps} | LR: {config.learning_rate}")
    logger.info("=" * 70)

    global_step = 0
    best_val_acc = 0.0
    epochs_wo = 0
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]

            input_ids, attention_mask, labels = build_batch_text(tokenizer, questions, answers, config.max_seq_len, device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(pixel_values, input_ids, attention_mask, labels) / config.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * config.grad_accum
            epoch_batches += 1

            if (batch_idx + 1) % config.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % config.log_every_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / max(1, epoch_batches)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step": global_step})

                if val_samples and config.eval_every_steps > 0 and global_step % config.eval_every_steps == 0:
                    val_metrics = evaluate_classification(
                        model=model,
                        tokenizer=tokenizer,
                        samples=val_samples,
                        config=config,
                        device=device,
                        label_vocab=label_vocab,
                        max_samples=config.eval_samples,
                        split_name="val",
                    )
                    logger.info(f"[Step {global_step}] VAL metrics: {val_metrics}")

                    # use overall_accuracy as early stopping signal
                    if val_metrics["overall_accuracy"] > best_val_acc + config.early_stopping_min_delta:
                        best_val_acc = val_metrics["overall_accuracy"]
                        epochs_wo = 0
                        save_checkpoint(model, tokenizer, config, output_dir, "best", global_step, val_metrics)
                    else:
                        epochs_wo += 1

                    if config.early_stopping and epochs_wo >= config.early_stopping_patience:
                        logger.info(f"Early stopping after {epochs_wo} evals without improvement")
                        break

                    model.train()

                if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
                    save_checkpoint(model, tokenizer, config, output_dir, f"step_{global_step}", global_step)

        avg_loss = epoch_loss / max(1, epoch_batches)
        logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
        save_checkpoint(model, tokenizer, config, output_dir, f"epoch_{epoch+1}", global_step)

        if config.early_stopping and epochs_wo >= config.early_stopping_patience:
            break

    save_checkpoint(model, tokenizer, config, output_dir, "final", global_step)

    # Load best for final test
    best_ckpt = output_dir / "checkpoint_best"
    if best_ckpt.exists():
        load_checkpoint(model, best_ckpt)
        logger.info("Loaded best checkpoint for final evaluation")

    final_metrics = {}
    if test_samples:
        test_metrics = evaluate_classification(
            model=model,
            tokenizer=tokenizer,
            samples=test_samples,
            config=config,
            device=device,
            label_vocab=label_vocab,
            max_samples=-1,
            split_name="test",
        )
        speed = measure_inference_speed(model, tokenizer, test_samples, config, device)
        final_metrics = {**test_metrics, **{f"speed_{k}": v for k, v in speed.items()}}

        with open(output_dir / "final_test_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info("=" * 70)
        logger.info(f"TEST metrics: {final_metrics}")
        logger.info("=" * 70)

    total_time = time.time() - start_time
    logger.info(f"Run complete | time={format_time(total_time)} | best_val_acc={best_val_acc:.4f}")

    return str(output_dir), final_metrics


# ==============================================================================
# MULTI-RUN ABLATION DRIVER
# ==============================================================================

def summarize_runs(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in keys:
        vals = [r.get(k, None) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)}
    return out


def run_ablation(config: OptimizedConfig, resamplers: List[str], seeds: List[int]) -> Path:
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    exp_stamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = root / f"ABLAT_{exp_stamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for res in resamplers:
        for sd in seeds:
            cfg = OptimizedConfig(**asdict(config))
            cfg.resampler = res
            cfg.seed = sd
            cfg.output_dir = str(exp_dir)  # nest runs inside this ablation folder

            logger.info("=" * 70)
            logger.info(f"START RUN | resampler={res} | seed={sd}")
            logger.info("=" * 70)

            run_dir, metrics = train_one_run(cfg)
            rec = {"resampler": res, "seed": sd, "run_dir": run_dir, **metrics}
            all_results.append(rec)

            with open(exp_dir / "all_runs.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # summarize per resampler
    summary = {}
    metric_keys = [
        "overall_accuracy",
        "overall_macro_f1_valid",
        "overall_kappa_valid",
        "overall_valid_frac",
        "speed_samples_per_sec",
    ]
    # also per-task metrics if present
    for t in config.target_tasks or []:
        metric_keys.extend([
            f"{t}_accuracy",
            f"{t}_macro_f1_valid",
            f"{t}_kappa_valid",
            f"{t}_valid_frac",
        ])

    for res in resamplers:
        rows = [r for r in all_results if r["resampler"] == res]
        summary[res] = summarize_runs(rows, metric_keys)

    with open(exp_dir / "summary_mean_std.json", "w") as f:
        json.dump(summary, f, indent=2)

    # simple CSV
    csv_path = exp_dir / "all_runs.csv"
    if all_results:
        keys = sorted(set().union(*[r.keys() for r in all_results]))
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in all_results:
                f.write(",".join([str(r.get(k, "")) for k in keys]) + "\n")

    logger.info("=" * 70)
    logger.info(f"ABLATION DONE | results at: {exp_dir}")
    logger.info(f"Saved: all_runs.json, all_runs.csv, summary_mean_std.json")
    logger.info("=" * 70)

    # print compact summary
    for res in resamplers:
        s = summary[res]
        acc = s.get("overall_accuracy", {})
        f1 = s.get("overall_macro_f1_valid", {})
        kap = s.get("overall_kappa_valid", {})
        spd = s.get("speed_samples_per_sec", {})
        logger.info(
            f"[{res}] Acc={acc.get('mean', float('nan')):.4f}±{acc.get('std', float('nan')):.4f} | "
            f"F1={f1.get('mean', float('nan')):.4f}±{f1.get('std', float('nan')):.4f} | "
            f"Kappa={kap.get('mean', float('nan')):.4f}±{kap.get('std', float('nan')):.4f} | "
            f"Speed={spd.get('mean', float('nan')):.2f}±{spd.get('std', float('nan')):.2f} samp/s"
        )

    return exp_dir


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser("Surgical VLM Ablation: Visual Resampling (5 runs)")
    p.add_argument("--train_data", type=str, default="/mnt/share/ali/VLM_Project/hospital_data/train.json")
    p.add_argument("--val_data", type=str, default=None)
    p.add_argument("--test_data", type=str, default="/mnt/share/ali/VLM_Project/hospital_data/test.json")
    p.add_argument("--output_dir", type=str, default="/mnt/share/ali/VLM_Project/checkpoints_ablation")

    p.add_argument("--vision_model", type=str, default="/mnt/share/ali/VLMs/hf_cache/hub/siglip2-model/")
    p.add_argument("--language_model", type=str, default="/mnt/share/ali/VLMs/hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/")

    p.add_argument("--tasks", nargs="+", default=["step_classification", "stage_classification"])

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--visual_tokens", type=int, default=1024)
    p.add_argument("--num_frames", type=int, default=8)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)

    # resampler selection
    p.add_argument("--resampler", type=str, default="all",
                   choices=["all", "interpolate", "adaptive_avg", "adaptive_max", "frame_interpolate", "perceiver"])

    # perceiver params
    p.add_argument("--perceiver_layers", type=int, default=1)
    p.add_argument("--perceiver_heads", type=int, default=8)
    p.add_argument("--perceiver_dropout", type=float, default=0.0)

    # multi-run
    p.add_argument("--num_runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # eval/speed
    p.add_argument("--eval_samples", type=int, default=200)
    p.add_argument("--max_new_tokens_eval", type=int, default=16)
    p.add_argument("--speed_samples", type=int, default=100)

    # early stopping
    p.add_argument("--no_early_stopping", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = OptimizedConfig(
        train_data=args.train_data,
        val_data=args.val_data,
        test_data=args.test_data,
        output_dir=args.output_dir,
        vision_model_path=args.vision_model,
        language_model_path=args.language_model,
        target_tasks=args.tasks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        visual_tokens=args.visual_tokens,
        num_frames=args.num_frames,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        resampler="interpolate",
        perceiver_layers=args.perceiver_layers,
        perceiver_heads=args.perceiver_heads,
        perceiver_dropout=args.perceiver_dropout,
        eval_samples=args.eval_samples,
        max_new_tokens_eval=args.max_new_tokens_eval,
        speed_samples=args.speed_samples,
        early_stopping=not args.no_early_stopping,
        seed=args.seed,
    )

    # seeds for N runs
    seeds = [args.seed + i for i in range(args.num_runs)]

    if args.resampler == "all":
        resamplers = ["interpolate", "adaptive_avg", "adaptive_max", "frame_interpolate", "perceiver"]
    else:
        resamplers = [args.resampler]

    exp_dir = run_ablation(cfg, resamplers, seeds)
    print(f"\nAblation complete. Results saved in:\n{exp_dir}\n")


if __name__ == "__main__":
    main()
