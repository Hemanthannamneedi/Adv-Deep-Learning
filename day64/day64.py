
import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "corpus.txt")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCE_LENGTH = 80
STEP = 3
EMBEDDING_DIM = 64
LSTM_UNITS = 128
EPOCHS = 30
BATCH_SIZE = 64

np.random.seed(42)
tf.random.set_seed(42)


# --------------------------------------------------
# 2. Load and prepare text corpus
# --------------------------------------------------

def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            "corpus.txt was not found. Add your training text to corpus.txt."
        )

    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    text = text.lower()
    text = " ".join(text.split())

    if len(text) <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Corpus must contain more than {SEQUENCE_LENGTH} characters."
        )

    return text


text = load_corpus()

# Character-to-integer and integer-to-character mappings
characters = sorted(set(text))
char_to_index = {char: index for index, char in enumerate(characters)}
index_to_char = {index: char for char, index in char_to_index.items()}

vocab_size = len(characters)

print("Corpus characters:", len(text))
print("Unique characters:", vocab_size)
print("Vocabulary:", "".join(characters))


# --------------------------------------------------
# 3. Create input sequences and next-character targets
# --------------------------------------------------

X = []
y = []

for start in range(0, len(text) - SEQUENCE_LENGTH, STEP):
    sequence = text[start:start + SEQUENCE_LENGTH]
    target = text[start + SEQUENCE_LENGTH]

    X.append([char_to_index[char] for char in sequence])
    y.append(char_to_index[target])

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

print("Input shape:", X.shape)
print("Target shape:", y.shape)

# Save vocabulary mappings
mapping_path = os.path.join(MODEL_DIR, "character_mapping.json")

with open(mapping_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "char_to_index": char_to_index,
            "index_to_char": {
                str(index): char for index, char in index_to_char.items()
            },
            "sequence_length": SEQUENCE_LENGTH
        },
        file,
        ensure_ascii=False,
        indent=4
    )


# --------------------------------------------------
# 4. Build the character-level LSTM
# --------------------------------------------------

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=SEQUENCE_LENGTH
    ),
    LSTM(LSTM_UNITS, return_sequences=True),
    Dropout(0.2),
    LSTM(LSTM_UNITS),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

start_time = time.time()

history = model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

training_time = time.time() - start_time

model_path = os.path.join(MODEL_DIR, "character_lstm.keras")
model.save(model_path)

print(f"\nTraining completed in {training_time:.2f} seconds.")
print("Model saved:", model_path)


# --------------------------------------------------
# 6. Character sampling with temperature and top-k
# --------------------------------------------------

def sample_next_character(probabilities, temperature=0.7, top_k=8):
    probabilities = np.asarray(probabilities).astype("float64")

    # Avoid invalid probabilities
    probabilities = np.maximum(probabilities, 1e-10)

    # Temperature controls randomness
    logits = np.log(probabilities) / temperature
    logits = logits - np.max(logits)

    adjusted_probabilities = np.exp(logits)

    # Keep only the top-k candidate characters
    if top_k is not None and top_k < len(adjusted_probabilities):
        top_indices = np.argsort(adjusted_probabilities)[-top_k:]
        filtered = np.zeros_like(adjusted_probabilities)
        filtered[top_indices] = adjusted_probabilities[top_indices]
        adjusted_probabilities = filtered

    adjusted_probabilities /= np.sum(adjusted_probabilities)

    next_index = np.random.choice(
        len(adjusted_probabilities),
        p=adjusted_probabilities
    )

    return next_index


# --------------------------------------------------
# 7. Generate new text
# --------------------------------------------------

def generate_text(seed_text, temperature=0.7, length=300):
    seed_text = seed_text.lower()
    seed_text = " ".join(seed_text.split())

    # Keep only characters available in the training vocabulary
    seed_text = "".join(
        char for char in seed_text if char in char_to_index
    )

    if len(seed_text) < SEQUENCE_LENGTH:
        seed_text = text[:SEQUENCE_LENGTH]

    generated = seed_text

    for _ in range(length):
        input_text = generated[-SEQUENCE_LENGTH:]

        input_sequence = np.array(
            [[char_to_index[char] for char in input_text]],
            dtype=np.int32
        )

        probabilities = model.predict(
            input_sequence,
            verbose=0
        )[0]

        next_index = sample_next_character(
            probabilities,
            temperature=temperature,
            top_k=8
        )

        next_character = index_to_char[next_index]
        generated += next_character

    return generated


# --------------------------------------------------
# 8. Experiment with different temperatures
# --------------------------------------------------

temperatures = [0.2, 0.5, 0.8, 1.2]

seed_text = text[:SEQUENCE_LENGTH]

results = {}

for temperature in temperatures:
    print(f"\nGenerating text with temperature = {temperature}")

    generated_text = generate_text(
        seed_text=seed_text,
        temperature=temperature,
        length=300
    )

    results[str(temperature)] = generated_text

    print("-" * 60)
    print(generated_text)
    print("-" * 60)


# Save generated text
output_path = os.path.join(OUTPUT_DIR, "temperature_experiments.txt")

with open(output_path, "w", encoding="utf-8") as file:
    for temperature, generated_text in results.items():
        file.write(f"\n\nTEMPERATURE: {temperature}\n")
        file.write("=" * 60 + "\n")
        file.write(generated_text)
        file.write("\n")


# --------------------------------------------------
# 9. Save training history and plot loss
# --------------------------------------------------

history_path = os.path.join(OUTPUT_DIR, "training_history.json")

with open(history_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        file,
        indent=4
    )

plt.figure(figsize=(9, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Character-Level LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "training_loss.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 10. Save experiment summary
# --------------------------------------------------

report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")

with open(report_path, "w", encoding="utf-8") as file:
    file.write("DAY 64 - CHARACTER LEVEL TEXT GENERATION\n")
    file.write("=" * 50 + "\n")
    file.write(f"Corpus characters: {len(text)}\n")
    file.write(f"Vocabulary size: {vocab_size}\n")
    file.write(f"Training sequences: {len(X)}\n")
    file.write(f"Sequence length: {SEQUENCE_LENGTH}\n")
    file.write(f"LSTM units: {LSTM_UNITS}\n")
    file.write(f"Training time: {training_time:.2f} seconds\n")
    file.write(
        f"Final training loss: {history.history['loss'][-1]:.4f}\n"
    )
    file.write(
        f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n"
    )
    file.write("\nTemperature experiments: 0.2, 0.5, 0.8, 1.2\n")
    file.write("Generated samples: temperature_experiments.txt\n")

print("\nDay 64 completed successfully.")
print("Generated text:", output_path)
print("Training report:", report_path)
