# Baby Name Transformer

A character-level generative Transformer built and trained from scratch using Python and PyTorch to generate new baby-name sequences.

## Overview

This project demonstrates an end-to-end machine learning workflow, including data preparation, model architecture, training, validation, checkpointing, and generative inference.

The Transformer learns character-level patterns from a dataset of baby names and autoregressively generates new name-like sequences.

## Machine Learning Concepts

- PyTorch
- Transformer architecture
- Character/token embeddings
- Positional embeddings
- Multi-head self-attention
- Feed-forward neural network layers
- Causal attention masking
- Training and validation split
- Cross-entropy loss
- AdamW optimization
- Gradient clipping
- Learning-rate scheduling
- Model checkpointing
- Autoregressive inference
- Temperature-based sampling

## Architecture

Names Dataset
↓
Character Encoding
↓
Token + Positional Embeddings
↓
Transformer Blocks
↓
Multi-Head Self-Attention
↓
Feed-Forward Layers
↓
Next-Character Prediction
↓
Generated Names

## Training

The model is trained from scratch using PyTorch.

The training pipeline includes:

- 80/20 training and validation split
- AdamW optimizer
- Cross-entropy loss
- Gradient clipping
- Cosine learning-rate scheduling
- Validation-loss monitoring
- Best-model checkpointing

Run training with:

```bash
python3 train.py

