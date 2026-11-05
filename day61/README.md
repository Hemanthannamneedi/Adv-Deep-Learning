# Day 61 — Advanced Deep Learning: LSTM vs RNN

## Overview

This project explores Long Short-Term Memory (LSTM) networks and compares their performance with a basic Recurrent Neural Network (SimpleRNN) on a numerical sequence prediction task.

The experiment also evaluates the impact of different LSTM hidden-unit sizes: 32, 64, and 128.

## Objectives

- Understand the limitations of vanilla RNNs.
- Learn the LSTM architecture and its three gates.
- Build an LSTM model using TensorFlow/Keras.
- Compare SimpleRNN and LSTM models.
- Evaluate different LSTM unit configurations.
- Compare prediction errors and training time.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow / Keras

## Models

1. SimpleRNN — 32 units
2. LSTM — 32 units
3. LSTM — 64 units
4. LSTM — 128 units

## Project Structure

```text
day61/
├── day61.py
├── requirements.txt
├── assignment.txt
├── README.md
├── model_comparison.csv
├── sample_prediction.csv
├── evaluation_report.txt
├── models/
└── plots/
```

## Installation

```cmd
pip install -r requirements.txt
```

## Run the Project

```cmd
python day61.py
```

## Evaluation

Models are compared using:

- Test MAE
- Test MSE
- Test RMSE
- Training time
- Epochs completed

The results are saved in `model_comparison.csv`.

## Output Files

- `models/` — saved trained models.
- `plots/` — training loss, test MAE, and training time comparisons.
- `model_comparison.csv` — model performance results.
- `sample_prediction.csv` — sample sequence prediction.
- `evaluation_report.txt` — experiment summary.

## Key Learning

LSTM networks use a cell state and three gates — forget, input, and output — to control information flow and help preserve useful information across longer sequences.

The number of LSTM units affects model capacity, computational cost, and the risk of overfitting.

## Note

This project uses generated numerical sequences for educational purposes. The best model and performance metrics depend on the actual training run.

**Day:** 61  
**Topic:** Advanced Deep Learning — LSTM Fundamentals
