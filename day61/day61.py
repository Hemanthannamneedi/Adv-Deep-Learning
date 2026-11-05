
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Configuration
TRAIN_SAMPLES = 4000
TEST_SAMPLES = 1000
SEQUENCE_LENGTH = 10
EPOCHS = 50
BATCH_SIZE = 32


def generate_sequences(samples):
    """Generate numerical sequences and their next-value targets."""
    X = []
    y = []

    for _ in range(samples):
        start = np.random.uniform(0, 10)
        step = np.random.uniform(0.1, 1.0)

        sequence = start + np.arange(
            SEQUENCE_LENGTH + 1
        ) * step

        X.append(sequence[:SEQUENCE_LENGTH])
        y.append(sequence[SEQUENCE_LENGTH])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # RNN and LSTM expect: samples, time steps, features
    X = X.reshape(samples, SEQUENCE_LENGTH, 1)

    return X, y


# Generate the same dataset for all models
X_train, y_train = generate_sequences(TRAIN_SAMPLES)
X_test, y_test = generate_sequences(TEST_SAMPLES)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


def build_rnn_model(units=32):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1)),
        SimpleRNN(units, activation="tanh"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_lstm_model(units):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1)),
        LSTM(units, activation="tanh"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


results = []
histories = {}
trained_models = {}

# Compare Vanilla RNN with LSTM configurations
experiments = [
    ("SimpleRNN_32", "RNN", 32),
    ("LSTM_32", "LSTM", 32),
    ("LSTM_64", "LSTM", 64),
    ("LSTM_128", "LSTM", 128)
]

for model_name, model_type, units in experiments:
    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    if model_type == "RNN":
        model = build_rnn_model(units)
    else:
        model = build_lstm_model(units)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    training_time = time.time() - start_time

    test_loss, test_mae = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    predictions = model.predict(X_test, verbose=0).flatten()

    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))

    results.append({
        "Model": model_name,
        "Architecture": model_type,
        "Units": units,
        "Test_MSE": float(test_loss),
        "Test_MAE": float(test_mae),
        "Test_RMSE": rmse,
        "Training_Time_Seconds": training_time,
        "Epochs_Trained": len(history.history["loss"])
    })

    histories[model_name] = history
    trained_models[model_name] = model

    model.save(f"models/{model_name.lower()}.keras")

    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Training time: {training_time:.2f} seconds")


# Save comparison results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Test_MAE",
    ascending=True
)

results_df.to_csv("model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Identify the model with the lowest test MAE
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest model based on test MAE:", best_model_name)

# Plot training and validation loss
plt.figure(figsize=(10, 6))

for model_name, history in histories.items():
    plt.plot(
        history.history["loss"],
        label=f"{model_name} Training"
    )

    plt.plot(
        history.history["val_loss"],
        linestyle="--",
        label=f"{model_name} Validation"
    )

plt.title("RNN and LSTM Training/Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/training_loss_comparison.png")
plt.close()

# Compare model MAE
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Test_MAE"])
plt.title("RNN vs LSTM - Test MAE")
plt.xlabel("Model")
plt.ylabel("Test Mean Absolute Error")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("plots/test_mae_comparison.png")
plt.close()

# Compare training time
plt.figure(figsize=(9, 5))
plt.bar(
    results_df["Model"],
    results_df["Training_Time_Seconds"]
)
plt.title("Training Time Comparison")
plt.xlabel("Model")
plt.ylabel("Training Time (seconds)")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("plots/training_time_comparison.png")
plt.close()

# Demonstrate prediction using the best model
sample_input = np.array(
    [[[10], [11], [12], [13], [14],
      [15], [16], [17], [18], [19]]],
    dtype=np.float32
)

actual_next_value = 20.0
predicted_next_value = float(
    best_model.predict(sample_input, verbose=0)[0][0]
)

print("\nSAMPLE SEQUENCE PREDICTION")
print("Input sequence:", sample_input.flatten())
print("Actual next value:", actual_next_value)
print("Predicted next value:", round(predicted_next_value, 4))

# Save sample prediction
pd.DataFrame([{
    "Input_Sequence": "10,11,12,13,14,15,16,17,18,19",
    "Actual_Next_Value": actual_next_value,
    "Predicted_Next_Value": predicted_next_value,
    "Selected_Model": best_model_name
}]).to_csv("sample_prediction.csv", index=False)

# Save a text report
with open("evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write("DAY 61 - RNN VS LSTM EVALUATION\n")
    file.write("=" * 45 + "\n\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nBest model based on test MAE: ")
    file.write(best_model_name)
    file.write("\n\nSample prediction:\n")
    file.write(f"Actual value: {actual_next_value}\n")
    file.write(f"Predicted value: {predicted_next_value:.4f}\n")

print("\nAll experiments completed.")
print("Models saved in the models folder.")
print("Plots saved in the plots folder.")
print("Comparison saved in model_comparison.csv")
print("Report saved in evaluation_report.txt")
