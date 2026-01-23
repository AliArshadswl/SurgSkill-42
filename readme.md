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

### Procedural Stages (3 Classes)

| Stage ID | Stage Name (Chinese) | Stage Name (English) |
|----------|---------------------|----------------------|
| stage_01 | 穿手术衣 | Gown Donning |
| stage_02 | 戴手套 | Glove Donning |
| stage_03 | 脱手术衣和手套 | Gown and Glove Removal |

### Procedural Steps (13 Classes)

| Step ID | Step Description (Chinese) | Step Description (English) |
|---------|---------------------------|----------------------------|
| step_001 | 拿起叠放着的手术衣，双手不能接触下面的手术衣 | Pick up gown without touching gowns below |
| step_002 | 打开手术衣 | Unfold gown |
| step_003 | 将双手插入袖筒 | Insert arms into sleeves |
| step_004 | 巡回护士协助穿好手术衣 | Assistant helps with gown |
| step_005 | 系腰带 | Tie waist belt |
| step_006 | 拿起右手手套 | Pick up right glove |
| step_007 | 右手戴手套 | Insert right hand into glove |
| step_008 | 左手戴手套 | Insert left hand into glove |
| step_009 | 解下腰带并递给巡回护士 | Untie and pass belt to assistant |
| step_010 | 整理右手手套袖口 | Adjust right glove cuff |
| step_011 | 整理左手手套袖口 | Adjust left glove cuff |
| step_012 | 脱手套 | Remove gloves |
| step_013 | 转身并系腰带 | Turn and tie belt at waist |

### Multi-View Setup

The dataset features synchronized four-camera recordings:
- Close-up view (hand-object interactions)
- Overhead wide-angle view (full room context)
- Frontal view (upper-body motion)
- Top-down view (instrument table)

Views are spatially concatenated into 2×2 composite frames.

## 📁 Data Structure

```
SurgSkill-42/
├── frames/
│   ├── subject_0001/
│   │   ├── step_001/
│   │   │   └── subject_0001__step_001__36.000_38.000/
│   │   │       ├── frame_0000.jpg
│   │   │       ├── frame_0001.jpg
│   │   │       ├── frame_0002.jpg
│   │   │       ├── frame_0003.jpg
│   │   │       ├── frame_0004.jpg
│   │   │       ├── frame_0005.jpg
│   │   │       ├── frame_0006.jpg
│   │   │       └── frame_0007.jpg
│   │   ├── step_002/
│   │   └── ...
│   ├── subject_0002/
│   └── ...
├── segments/
│   └── subject_0002__step_001__20.000_25.000__ec88ecff.mp4
├── splits/
│   ├── formatted_train.json
│   ├── formatted_val.json
│   └── formatted_test.json
└── README.md
```

### Annotation Format

Each sample in the JSON files follows this format:

```json
{
  "id": "subject_0002__step_001__20.000_25.000__step_classification",
  "task": "step_classification",
  "images": [
    "frames/subject_0002/step_001/subject_0002__step_001__20.000_25.000/frame_0000.jpg",
    "frames/subject_0002/step_001/subject_0002__step_001__20.000_25.000/frame_0001.jpg",
    "frames/subject_0002/step_001/subject_0002__step_001__20.000_25.000/frame_0002.jpg",
    "frames/subject_0002/step_001/subject_0002__step_001__20.000_25.000/frame_0003.jpg",
    "frames/subject_0002/step_001/subject_0002__step_001__20.000_25.000/frame_0004.jpg",
    "frames/subject_0002/step_001/subject_0002__step_001__20.000_25.000/frame_0005.jpg",
    "frames/subject_0002/step_001/subject_0002__step_001__20.000_25.000/frame_0006.jpg",
    "frames/subject_0002/step_001/subject_0002__step_001__20.000_25.000/frame_0007.jpg"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "以下视频帧展示了哪个步骤？请选择正确的步骤类别。\n<image>\n\n<image>\n..."
    },
    {
      "from": "gpt",
      "value": "拿起叠放着的手术衣，双手不能接触下面的手术衣"
    }
  ],
  "label": "拿起叠放着的手术衣，双手不能接触下面的手术衣",
  "label_idx": 8,
  "meta": {
    "subject": "subject_0002",
    "stage_id": "stage_01",
    "stage_text": "穿手术衣",
    "step_id": "step_001",
    "step_text": "拿起叠放着的手术衣，双手不能接触下面的手术衣",
    "segment_path": "segments/subject_0002__step_001__20.000_25.000__ec88ecff.mp4",
    "start_sec": 20,
    "end_sec": 25,
    "duration_sec": 5,
    "task_type": "step_classification"
  }
}
```

### Task Types

Each sample has two task variants:
- **step_classification**: Classify into one of 13 fine-grained steps
- **stage_classification**: Classify into one of 3 procedural stages

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
    --data_path ./splits/formatted_train.json \
    --val_path ./splits/formatted_val.json \
    --frames_dir ./frames \
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
    --data_path ./splits/formatted_train.json \
    --val_path ./splits/formatted_val.json \
    --frames_dir ./frames \
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
    --data_path ./splits/formatted_test.json \
    --frames_dir ./frames \
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

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{arshad2025surgskill,
  title={Do Larger Vision-Language Models Help Under Domain Shift? A Study on Surgical Procedural Understanding},
  author={Arshad, Muhammad Ali and Li, Wei and Yan, Yan and Liu, Yu-Shi and Wang, Lei},
  booktitle={IEEE Engineering in Medicine and Biology Conference (EMBC)},
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
