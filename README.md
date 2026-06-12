# Multimodal Conversational Emotion Recognition (MELD)

This project implements a multimodal emotion recognition system for conversational data using the MELD dataset. The model combines textual and acoustic information to classify emotions at the utterance level in multi-party dialogues.

## Overview

The system explores progressively more complex architectures for emotion recognition:

1. Utterance-level text-only baseline
2. Contextual text model using sliding window context
3. Multimodal model combining text and speech embeddings

The goal is to evaluate how contextual and acoustic information impacts emotion classification performance in conversational settings.

---

## Dataset

**MELD (Multimodal EmotionLines Dataset)**  
Source: TV series *Friends*

Modalities:
- Text transcripts
- Audio clips (converted to `.flac` format)
- (Visual modality not used in this project)

Emotion labels:
- anger
- disgust
- fear
- joy
- neutral
- sadness
- surprise

---

## Model Architecture

### Text Encoder
- RoBERTa transformer model
- Used for:
  - Utterance-only baseline
  - Contextual window-based inputs

### Audio Encoder
- Pretrained HuBERT model
- Extracts contextual acoustic embeddings from speech

### Fusion Strategy
- Feature-level concatenation of:
  - Text embeddings
  - Audio embeddings
- Followed by a classification head

---

## Experiments

### 1. Utterance-only Baseline
- Single utterance classification
- No contextual information

### 2. Contextual Text Model
- Sliding window of previous utterances (k=2)
- Incorporates dialogue context

### 3. Multimodal Model
- Combines RoBERTa (text) + HuBERT (audio)
- Feature fusion via concatenation

---

## Training Details

- Framework: PyTorch + HuggingFace Transformers
- Environment: Google Colab (T4 GPU)
- Optimizer: AdamW
- Learning rate: `1e-5`
- Epochs: 10 (early stopping enabled with patience of 2, validation set weighted f1 used to determine best model)
- Mixed precision training (fp16 enabled)

---

## Evaluation Metrics

- Accuracy
- Weighted F1-score (primary metric)
- Macro F1-score
- Per-class F1-score

---

## Key Findings

- Contextual information improves performance over utterance-only models.
- Audio features provide additional but modest improvements over text-only models.
- Model performance is heavily affected by class imbalance in MELD.
- Neutral and joy dominate predictions due to dataset distribution.
- Multimodal fusion improves robustness but gains are incremental.

---

## Limitations

- Strong class imbalance in MELD dataset
- Simple feature fusion strategy (no cross-attention)
- Audio quality depends on preprocessing pipeline
- Limited contextual window size due to sequence length constraints

---

## Future Work

- Improved multimodal fusion (cross-attention / transformer fusion)
- Advanced imbalance handling strategies
- Larger pretrained speech encoders
- Speaker-aware modeling for dialogue context

---

## Acknowledgements

- MELD dataset by Poria et al.
- HuggingFace Transformers library
- HuBERT and RoBERTa pretrained models
