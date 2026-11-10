# Day 64 — Text Generation with LSTM

## Overview

This project implements character-level text generation using a Long Short-Term Memory (LSTM) neural network. The model learns character patterns from a text corpus and predicts the next character in a sequence.

## Objectives

- Prepare a text corpus for character-level modelling.
- Convert characters into numerical representations.
- Generate input sequences and next-character targets.
- Build and train an LSTM model.
- Generate text using different temperature values.
- Compare creativity and coherence across sampling settings.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Model Architecture

1. Embedding layer
2. LSTM layer with 128 units
3. Dropout layer
4. LSTM layer with 128 units
5. Dense output layer with softmax activation

## Text Generation

The model predicts a probability distribution over the available characters. Temperature sampling controls how those probabilities are used to select the next character.

Experiments:

- Temperature 0.2
- Temperature 0.5
- Temperature 0.8
- Temperature 1.2

Top-k sampling is also used to restrict the candidate characters to the most likely options.

## Project Structure

```text
day64/
├── corpus.txt
├── day64.py
├── requirements.txt
├── assignment.txt
├── README.md
├── models/
│   ├── character_lstm.keras
│   └── character_mapping.json
└── outputs/
    ├── temperature_experiments.txt
    ├── training_history.json
    ├── evaluation_report.txt
    └── training_loss.png
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Execution

```bash
python day64.py
```

## Generated Results

- Trained character-level LSTM model
- Character mapping JSON
- Generated text for four temperature values
- Training history
- Training and validation loss plot
- Evaluation report

## Learning Outcomes

- Character-level text encoding
- Sliding-window sequence preparation
- LSTM-based language modelling
- Probability-based character sampling
- Temperature and top-k sampling
- Text generation evaluation

## Note

The generated output depends on the training corpus, model training, and sampling parameters. A small corpus may produce repetitive or incoherent text. This project is an educational implementation, not a production language model.
