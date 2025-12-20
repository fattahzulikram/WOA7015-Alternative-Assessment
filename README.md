# Benchmarking CNN-LSTM baseline against BLIP-based approach on SLAKE dataset

A comparison between traditional discriminative models with modern pre-trained generative models for medical-VQA on the SLAKE dataset.

This project is still ongoing. The file will be updated accordingly.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## Overview

This project provides a comparison between two distinct approaches to medical Visual Question Answering.

1. **CNN-LSTM Baseline** - A discriminative model that uses ResNet-34 as a baseline visual encoder and a Bidirectional LSTM with self-attention mechanism as the textual
encoder.

2. **BioMedBLIP** - A vision-language model based on BLIP that is specifically trained on domain-specific medical dataset.

### Key Features
- The implementation of the CNN-biLSTM baseline
- Hyperparameter tuning for the baseline
- Evaluation with different classification and generative metrics alongside per-answer-type metrics

## Dataset

### SLAKE Dataset
The **SLAKE (Semantic Label for Anatomical Knowledge Evaluation)** dataset is used for this benchmark.

- **Images**: 642 radiological images (CT, MRI, X-ray)
- **Question-Answers**: 14,028 total question-answer pairs with 7,033 pairs in English
- **Answer Types**: Closed-ended and Open-ended
- **Data Split**: 70% training, 15% validation, and 15% test

### Dataset Download
```bash
# Using huggingface library
from datasets import load_dataset
dataset = load_dataset("BoKelvin/SLAKE")
```

The images zip file needs to be downloaded and extracted separately.

Or download the full dataset manually from [HuggingFace SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)

## Models

### 1. The baseline: CNN-LSTM

**Architecture:**
- **Visual Encoder**: ResNet-34 (pretrained on ImageNet)
  - Output: 512-dimensional visual features
- **Question Encoder**: Bidirectional LSTM (512 hidden units per direction)
  - Self-attention mechanism for keyword emphasis
  - Output: 1024-dimensional question features
- **Fusion Network**: Multi-layer perceptron with dropout
  - Input: 1536-dim concatenated features
  - Output: Probability distribution over answer vocabulary

**Key Features:**
- End-to-end trainable
- Discriminative approach, treats each answer as a class

### 2. BioMedBLIP

**Architecture:**
- **Vision Encoder**: Vision Transformer (ViT)
- **Text Encoder**: Transformer
- **Multimodal Decoder**: Cross-attention fusion

**Key Features:**
- Pre-trained on large-scale biomedical data
- Generative approach with free-form text output
- State-of-the-art performance

## Installation and Usage

### Prerequisites
- Python (3.14 used in this project)
- CUDA-capable GPU (recommended)
- At lease 16GB RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/fattahzulikram/WOA7015-Alternative-Assessment.git
cd WOA7015-Alternative-Assessment
```

2. Create a virtual environment (Windows OS used here):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

4. Open and run **baseline_final.ipynb** for the baseline model.

## Results

### Overall Performance

#### Accuracy Based Metrics

| Model | Overall Acc | Exact Match |
|-------|-------------|-------------|
| **CNN-LSTM** | 81.53% | 81.53% |
| **BioMedBLIP** | TBD | TBD |

#### F1 Scores

| Model | Macro F1 | Weighted F1 |
|-------|----------|-------------|
| **CNN-LSTM** | 54.53% | 81.46% |
| **BioMedBLIP** | TBD | TBD |

#### Generative Metrics

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|-------|--------|--------|--------|--------|---------|---------|---------|--------|
| **CNN-LSTM** | 83.79% | 35.82% | 24.57% | 19.23% | 84.42% | 15.89% | 84.08% | 48.56% |
| **BioMedBLIP** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 

| Model | BERTScore Precision | BERTScore Recall | BERTScore F1 |
|-------|---------------------|------------------|--------------|
| **CNN-LSTM** | 96.74% | 96.75% | 96.74% |
| **BioMedBLIP** | TBD | TBD | TBD |

### Performance by Question Type

#### Open-Ended Questions
| Model | Accuracy | F1 Score | Exact Match |
|-------|----------|----------|-------------|
| **CNN-LSTM** | 80.16% | 53.98% | 80.16% |
| **BioMedBLIP** | TBD | TBD | TBD |

#### Closed-Ended Questions
| Model | Accuracy | F1 Score | Exact Match |
|-------|----------|----------|-------------|
| **CNN-LSTM** | 83.65% | 64.75% | 83.65% |
| **BioMedBLIP** | TBD | TBD | TBD |

### Key Findings

#### CNN-LSTM Baseline

- **Accuracy**: Achieves 81.53% overall accuracy with solid performance on both open-ended and closed-ended questions.
- **Class Imbalance**: Almost 27% gap between weighted and macro F1 scores. The model struggles on rare answer types.
- **Generative Metrics**: Solid BERT-1 and ROUGE-1 scores, which drop down for higher orders. This is expected as most of the answers are short.
- **Semantic Understanding**: An outstanding BERTScore (96.75%) indicates good semantic alignment.

## Requirements

See `requirements.txt` for complete list.


## Acknowledgments

- SLAKE dataset creators for providing the benchmark
- BioMedBLIP researchers for the models
- HuggingFace for datasets and model hosting

---

**Note**: This is an academic project. Results are preliminary and models should be thoroughly validated before any further application.
