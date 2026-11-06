
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# 1. Reproducibility and folders
# --------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

MAX_FEATURES = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 64
UNITS = 64
EPOCHS = 5
BATCH_SIZE = 128

# --------------------------------------------------
# 3. Load IMDB sentiment dataset
# --------------------------------------------------

print("Loading IMDB movie reviews dataset...")

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=MAX_FEATURES,
    seed=SEED
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Pad sequences to a common length
X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

y_train = np.asarray(y_train, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)

print("Padded training shape:", X_train.shape)
print("Padded testing shape:", X_test.shape)

# --------------------------------------------------
# 4. Build LSTM model
# --------------------------------------------------

def build_lstm_model():
    model = Sequential([
        Input(shape=(MAX_LENGTH,)),
        Embedding(
            input_dim=MAX_FEATURES,
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        ),
        LSTM(UNITS),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# 5. Build GRU model
# --------------------------------------------------

def build_gru_model():
    model = Sequential([
        Input(shape=(MAX_LENGTH,)),
        Embedding(
            input_dim=MAX_FEATURES,
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        ),
        GRU(UNITS),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# 6. Train and evaluate both models
# --------------------------------------------------

experiments = [
    ("LSTM", build_lstm_model),
    ("GRU", build_gru_model)
]

results = []
histories = {}
trained_models = {}

for model_name, build_function in experiments:

    print("\n" + "=" * 55)
    print("Training", model_name, "model")
    print("=" * 55)

    model = build_function()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
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

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    parameter_count = model.count_params()

    results.append({
        "Model": model_name,
        "Parameters": parameter_count,
        "Training_Time_Seconds": training_time,
        "Test_Loss": float(test_loss),
        "Test_Accuracy": float(test_accuracy),
        "Best_Validation_Accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "Epochs_Trained": len(history.history["loss"])
    })

    histories[model_name] = history
    trained_models[model_name] = model

    model.save(f"models/{model_name.lower()}_sentiment.keras")

    print("\nModel:", model_name)
    print("Parameters:", parameter_count)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# 7. Save comparison results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test_Accuracy",
    ascending=False
)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nLSTM VS GRU COMPARISON")
print(results_df.to_string(index=False))

# Select model with highest test accuracy
best_model_name = results_df.iloc[0]["Model"]

print("\nBest model based on test accuracy:", best_model_name)


# --------------------------------------------------
# 8. Plot training and validation accuracy
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model_name, history in histories.items():

    plt.plot(
        history.history["accuracy"],
        label=f"{model_name} Training"
    )

    plt.plot(
        history.history["val_accuracy"],
        linestyle="--",
        label=f"{model_name} Validation"
    )

plt.title("LSTM vs GRU - Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/accuracy_comparison.png")
plt.close()


# --------------------------------------------------
# 9. Plot test accuracy
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Test_Accuracy"]
)

plt.title("LSTM vs GRU - Test Accuracy")
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("plots/test_accuracy_comparison.png")
plt.close()


# --------------------------------------------------
# 10. Plot training time
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Training_Time_Seconds"]
)

plt.title("LSTM vs GRU - Training Time")
plt.xlabel("Model")
plt.ylabel("Training Time (seconds)")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("plots/training_time_comparison.png")
plt.close()


# --------------------------------------------------
# 11. Plot parameter count
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Parameters"]
)

plt.title("LSTM vs GRU - Parameter Count")
plt.xlabel("Model")
plt.ylabel("Number of Parameters")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("plots/parameter_comparison.png")
plt.close()


# --------------------------------------------------
# 12. Save evaluation report
# --------------------------------------------------

with open(
    "evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("DAY 62 - LSTM VS GRU SENTIMENT ANALYSIS\n")
    file.write("=" * 50 + "\n\n")

    file.write(results_df.to_string(index=False))

    file.write("\n\nBest model based on test accuracy: ")
    file.write(best_model_name)

    file.write(
        "\n\nDataset: IMDB Movie Reviews\n"
        "Task: Binary Sentiment Classification\n"
    )

print("\nAll experiments completed successfully.")
print("Trained models saved in models/")
print("Comparison saved in model_comparison.csv")
print("Plots saved in plots/")
print("Report saved in evaluation_report.txt")
