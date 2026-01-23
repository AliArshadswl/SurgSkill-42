# SurgSkill-42: Vision-Language Models for Surgical Procedural Understanding

[![Paper](https://img.shields.io/badge/Paper-EMBC%202025-blue)](https://github.com/AliArshadswl/SurgSkill-42)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](https://github.com/AliArshadswl/SurgSkill-42)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official repository for the paper: **"Do Larger Vision-Language Models Help Under Domain Shift? A Study on Surgical Procedural Understanding"**

## 📋 Abstract

Vision-language models (VLMs) have achieved remarkable success in visual understanding, with larger models typically outperforming smaller ones. But does this scaling advantage hold under significant domain shift? We investigate this in surgical procedural understanding, specifically stage and step recognition for trainee competency evaluation. We introduce **SurgSkill-42**, a multi-view video dataset of 42 trainees performing sterile gowning and gloving procedures, annotated with 3 stages and 13 fine-grained steps.

**Key Findings:**
- Scaling provides no benefit and can degrade performance under domain shift
- A 1B-parameter model achieves 87.1%±4.2% accuracy, matching its 8B counterpart
- Within the Qwen family, a 0.6B model achieves 84.8% while a 14B model achieves only 58.5%
- Architectural quality and vision encoder strength yield greater returns than scaling language model size

## 🗂️ Dataset Overview

### Statistics

| Split | Subjects | Temporal Segments | Stage Classes | Step Classes |
|-------|----------|-------------------|---------------|--------------|
| Train | 33 | 379 | 3 | 13 |
| Test | 9 | 112 | 3 | 13 |

### Procedural Stages and Steps

| Stage | Stage Name | Steps |
|-------|------------|-------|
| 1 | Gown Donning | Pick up gown, Unfold gown, Insert arms into sleeves, Assistant helps with gown, Untie and pass belt to assistant, Turn and tie belt at waist |
| 2 | Glove Donning | Pick up right glove, Insert right hand into glove, Insert left hand into glove, Adjust right glove cuff |
| 3 | Gown and Glove Removal | Adjust left glove cuff, Remove gloves, Tie waist belt |

### Multi-View Setup

The dataset features synchronized four-camera recordings:
- Close-up view (hand-object interactions)
- Overhead wide-angle view (full room context)
- Frontal view (upper-body motion)
- Top-down view (instrument table)

Views are spatially concatenated into 2×2 composite frames.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/AliArshadswl/SurgSkill-42.git
cd SurgSkill-42

# Create conda environment
conda create -n surgskill python=3.10
conda activate surgskill

# Install dependencies
pip install -r requirements.txt
```


## 🏋️ Training

### InternVL3.5-1B (Best Configuration)

```bash
python scripts/train.py \
    --model internvl3.5-1b \
    --data_path ./data \
    --output_dir ./checkpoints \
    --num_frames 8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --learning_rate 2e-4 \
    --num_epochs 20 \
    --batch_size 4
```

### SigLIP2 + Qwen3-0.6B

```bash
python scripts/train.py \
    --model siglip2_qwen3_0.6b \
    --data_path ./data \
    --output_dir ./checkpoints \
    --num_frames 8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --learning_rate 2e-4 \
    --num_epochs 20
```

## 📊 Evaluation

```bash
# Evaluate on SurgSkill-42
python scripts/evaluate.py \
    --model_path ./checkpoints/best_model \
    --data_path ./data/test \
    --task both  # Options: stage, step, both

# Cross-dataset evaluation on Cholec80
python scripts/evaluate.py \
    --model_path ./checkpoints/best_model \
    --data_path ./data/cholec80 \
    --task phase
```

## 📈 Results

### Main Results on SurgSkill-42

| Model | Params | Overall (%) | Step (%) | Stage (%) |
|-------|--------|-------------|----------|-----------|
| InternVL3.5-1B | 1B | **87.05** | **78.57** | **95.54** |
| InternVL2.5-8B | 8B | 87.05 | 79.46 | 94.64 |
| SigLIP2 + Qwen3-0.6B | 0.6B | 84.82 | 75.89 | 93.75 |
| SigLIP2 + Qwen2.5-14B | 14B | 58.48 | 37.50 | 79.46 |

### Cross-Dataset Evaluation

| Model | EndoVis-18 Acc (%) | EndoVis-17 Acc (%) | CoPESD Acc (%) |
|-------|-------------------|-------------------|----------------|
| InternVL3.5-1B | **74.3** | 50.0 | **79.1** |
| EndoChat (~13B) | 71.5 | **55.5** | 75.3 |

## 🔧 Model Zoo

| Model | Download | Size |
|-------|----------|------|
| InternVL3.5-1B (Best) | [Coming Soon]() | ~2GB |
| SigLIP2 + Qwen3-0.6B | [Coming Soon]() | ~1.5GB |

## 📖 Citation

Our work is submitted to 
```bibtex
@inproceedings{arshad2025surgskill,
  title={Do Larger Vision-Language Models Help Under Domain Shift? A Study on Surgical Procedural Understanding}
  author={Arshad, Muhammad Ali and Li, Wei and Yan, Yan and Liu, Yu-Shi and Wang, Lei},
  year={2025}
}
```

## 🙏 Acknowledgments

- Clinical staff at Zhujiang Hospital for data collection and annotation
- Surgical educators who participated in the expert evaluation
- Authors of [EndoChat](https://github.com/xxx/EndoChat) for providing the Surg396K dataset

## 📧 Contact

- **Muhammad Ali Arshad**: arshad@siat.ac.cn

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [InternVL](https://github.com/OpenGVLab/InternVL)
- [SigLIP](https://github.com/google-research/big_vision)
- [EndoChat](https://github.com/xxx/EndoChat)
- [Cholec80](http://camma.u-strasbg.fr/datasets)
