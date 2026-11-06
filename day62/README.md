# Day 62 — Advanced Deep Learning: GRU (Gated Recurrent Unit)

## Overview

This project implements a GRU-based sentiment classification model and compares it with an LSTM model using the IMDB Movie Reviews dataset.

Both models classify movie reviews as positive or negative sentiment.

## Objectives

- Understand the GRU architecture.
- Learn about update and reset gates.
- Understand the differences between GRU and LSTM.
- Build GRU and LSTM models using TensorFlow/Keras.
- Compare parameter counts, training time, and test accuracy.
- Analyze the computational efficiency of both architectures.

## Dataset

**IMDB Movie Reviews Dataset**

- Positive sentiment: 1
- Negative sentiment: 0
- Maximum vocabulary size: 10,000 words
- Maximum review length: 200 tokens

The dataset is loaded using TensorFlow/Keras.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Keras

## Model Architectures

### LSTM

- Embedding layer
- LSTM layer with 64 units
- Dropout layer
- Dense sigmoid output layer

### GRU

- Embedding layer
- GRU layer with 64 units
- Dropout layer
- Dense sigmoid output layer

## Project Structure

```text
day62/
├── day62.py
├── requirements.txt
├── assignment.txt
├── README.md
├── model_comparison.csv
├── evaluation_report.txt
├── models/
│   ├── lstm_sentiment.keras
│   └── gru_sentiment.keras
└── plots/
    ├── accuracy_comparison.png
    ├── test_accuracy_comparison.png
    ├── training_time_comparison.png
    └── parameter_comparison.png
```

## Installation

```cmd
pip install -r requirements.txt
```

## Run the Project

```cmd
python day62.py
```

An internet connection is required for the initial IMDB dataset download.

## Evaluation Metrics

The models are compared using:

- Parameter count
- Training time
- Test accuracy
- Test loss

Results are saved in `model_comparison.csv`.

## Key Learning

GRU uses two gates:

- Update Gate
- Reset Gate

Unlike LSTM, GRU does not have a separate cell state. It uses a hidden state to carry information through the sequence.

GRU often has fewer parameters and can train faster than LSTM, although the best-performing architecture depends on the task and dataset.

## Output Files

- `models/` — saved LSTM and GRU models.
- `plots/` — accuracy, training-time, and parameter comparison charts.
- `model_comparison.csv` — experimental comparison results.
- `evaluation_report.txt` — evaluation summary.

## Note

Actual performance depends on the training run, hardware, and model configuration. The results in the generated CSV file should be used when documenting the experiment.

**Day:** 62  
**Topic:** Advanced Deep Learning — GRU Fundamentals
