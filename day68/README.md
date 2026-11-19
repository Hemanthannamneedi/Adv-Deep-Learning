# Day 68 – Transformer Encoder

## Overview

Built a complete Transformer Encoder classifier using TensorFlow and Keras. The project implements positional encoding, multi-head self-attention, residual connections, layer normalization and feed-forward networks.

The Transformer Encoder is evaluated against an LSTM classifier using the IMDB movie review sentiment dataset.

## Concepts Covered

- Transformer Encoder Architecture
- Sinusoidal Positional Encoding
- Multi-Head Self-Attention
- Residual Connections
- Layer Normalization
- Feed-Forward Networks
- Sequence Classification
- Transformer vs LSTM Comparison

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## Installation

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Run the Project

```bash
python transformer_encoder.py
```

The script downloads the IMDB dataset when needed, trains both models, evaluates their performance and generates comparison visualizations.

## Output Files

- `outputs/model_comparison.csv`
- `outputs/accuracy_comparison.png`
- `outputs/training_time_comparison.png`
- `outputs/training_accuracy.png`

Trained models are saved locally in the `models/` directory and are excluded from Git because of their file size.

## Learning Outcomes

- Understood Transformer Encoder architecture.
- Implemented sinusoidal positional encoding.
- Built a custom Transformer Encoder block.
- Applied residual connections and layer normalization.
- Implemented an LSTM baseline.
- Compared Transformer and LSTM classification performance.
