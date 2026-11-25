
import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocess import (
    decode_imdb_reviews,
    create_vectorizer,
    MAX_LENGTH
)

from model import build_model


# ==========================================
# Configuration
# ==========================================

tf.keras.utils.set_random_seed(42)

BATCH_SIZE = 64
TUNING_EPOCHS = 2
FINAL_EPOCHS = 5

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================
# Load Real IMDB Dataset
# ==========================================

print("Loading IMDB dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.imdb.load_data(
        num_words=10000
    )
)

print("Training reviews:", len(x_train))
print("Testing reviews:", len(x_test))


# ==========================================
# Convert Integer Sequences to Text
# ==========================================

print("\nPreprocessing reviews...")

train_texts = decode_imdb_reviews(x_train)
test_texts = decode_imdb_reviews(x_test)

train_texts = np.array(train_texts)
test_texts = np.array(test_texts)

y_train = np.array(y_train)
y_test = np.array(y_test)


# ==========================================
# Train and Validation Split
# ==========================================

X_train, X_val, y_train, y_val = (
    train_test_split(
        train_texts,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(test_texts))


# ==========================================
# Text Vectorization
# ==========================================

vectorizer = create_vectorizer(X_train)

print(
    "Vocabulary size:",
    vectorizer.vocabulary_size()
)


def make_dataset(texts, labels, shuffle=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(texts.tolist(), dtype=tf.string),
            tf.constant(labels, dtype=tf.int32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            10000,
            seed=42
        )

    return dataset.batch(
        BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(
    X_train,
    y_train,
    shuffle=True
)

val_ds = make_dataset(
    X_val,
    y_val
)

test_ds = make_dataset(
    test_texts,
    y_test
)


# ==========================================
# Hyperparameter Tuning
# ==========================================

configurations = [
    {
        "num_heads": 2,
        "dropout": 0.2,
        "learning_rate": 0.001
    },
    {
        "num_heads": 4,
        "dropout": 0.3,
        "learning_rate": 0.0005
    }
]

tuning_results = []

best_accuracy = -1
best_config = None

for index, config in enumerate(configurations):

    print("\n" + "=" * 50)
    print("Testing configuration:", index + 1)
    print(config)
    print("=" * 50)

    tf.keras.utils.set_random_seed(42)

    candidate = build_model(
        vectorizer=vectorizer,
        max_length=MAX_LENGTH,
        **config
    )

    start_time = time.perf_counter()

    history = candidate.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TUNING_EPOCHS,
        verbose=1
    )

    elapsed = time.perf_counter() - start_time

    validation_accuracy = max(
        history.history["val_accuracy"]
    )

    tuning_results.append({
        **config,
        "validation_accuracy": validation_accuracy,
        "training_time_seconds": elapsed
    })

    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy
        best_config = config.copy()

    del candidate


pd.DataFrame(tuning_results).to_csv(
    "outputs/tuning_results.csv",
    index=False
)

print("\nBest configuration:", best_config)


# ==========================================
# Final Model Training
# ==========================================

tf.keras.utils.set_random_seed(42)

model = build_model(
    vectorizer=vectorizer,
    max_length=MAX_LENGTH,
    **best_config
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/transformer_classifier.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

start_time = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINAL_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

training_time = time.perf_counter() - start_time


# Save model before evaluation and plotting
model.save(
    "models/transformer_classifier.keras"
)

with open(
    "models/config.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump({
        "max_length": MAX_LENGTH,
        "threshold": 0.5,
        "best_configuration": best_config
    }, file, indent=4)

print("\nModel saved successfully!")


# ==========================================
# Model Evaluation
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

start_time = time.perf_counter()

probabilities = model.predict(
    test_ds,
    verbose=1
).reshape(-1)

inference_time = time.perf_counter() - start_time

predictions = (
    probabilities >= 0.5
).astype(int)

metrics = {
    "accuracy": float(
        accuracy_score(y_test, predictions)
    ),
    "precision": float(
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    ),
    "test_loss": float(test_loss),
    "training_time_seconds": training_time,
    "test_inference_time_seconds": inference_time,
    "average_inference_ms_per_review": (
        inference_time / len(y_test)
    ) * 1000,
    "trainable_parameters": int(
        sum(
            np.prod(weight.shape)
            for weight in model.trainable_weights
        )
    ),
    "best_configuration": best_config
}

print("\n========== FINAL METRICS ==========")

for key, value in metrics.items():
    print(f"{key}: {value}")

with open(
    "outputs/metrics.json",
    "w"
) as file:

    json.dump(metrics, file, indent=4)


# ==========================================
# Classification Report
# ==========================================

report = classification_report(
    y_test,
    predictions,
    target_names=["Negative", "Positive"],
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

with open(
    "outputs/classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================
# Accuracy Visualization
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transformer Training Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/training_accuracy.png"
)

plt.close()


# ==========================================
# Loss Visualization
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/training_loss.png"
)

plt.close()


# ==========================================
# Confusion Matrix
# ==========================================

matrix = confusion_matrix(
    y_test,
    predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Negative", "Positive"]
)

display.plot(cmap="Blues")

plt.title("Transformer Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

print("\nAll evaluation results saved!")
print("Day 70 Transformer Classifier Training Completed!")
