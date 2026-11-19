
import os
import csv
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------
# 1. Configuration
# -----------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
BATCH_SIZE = 128
EPOCHS = 3

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# -----------------------------------------
# 2. Sinusoidal Positional Encoding
# -----------------------------------------

@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEncoding(layers.Layer):

    def __init__(self, max_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model

        positions = np.arange(max_length)[:, np.newaxis]

        dimensions = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(
            10000,
            (2 * (dimensions // 2)) / np.float32(d_model)
        )

        angle_radians = positions * angle_rates

        positional_encoding = np.zeros(
            (max_length, d_model)
        )

        positional_encoding[:, 0::2] = np.sin(
            angle_radians[:, 0::2]
        )

        positional_encoding[:, 1::2] = np.cos(
            angle_radians[:, 1::2]
        )

        self.pos_encoding = tf.constant(
            positional_encoding[np.newaxis, ...],
            dtype=tf.float32
        )

        self.supports_masking = True

    def call(self, inputs):
        sequence_length = tf.shape(inputs)[1]

        return inputs + self.pos_encoding[:, :sequence_length, :]

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "d_model": self.d_model
        })
        return config


# -----------------------------------------
# 3. Transformer Encoder Block
# -----------------------------------------

@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerEncoderBlock(layers.Layer):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

        self.supports_masking = True

    def call(self, inputs, training=None, mask=None):

        attention_mask = None

        if mask is not None:
            attention_mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        # Residual connection + Layer Normalization
        out1 = self.norm1(inputs + attention_output)

        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        # Residual connection + Layer Normalization
        return self.norm2(out1 + ffn_output)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()

        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })

        return config


# -----------------------------------------
# 4. Build Transformer Encoder Classifier
# -----------------------------------------

def build_transformer_model():

    inputs = layers.Input(
        shape=(MAX_LENGTH,),
        dtype="int32"
    )

    # Token embeddings
    x = layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )(inputs)

    # Positional encoding
    x = PositionalEncoding(
        MAX_LENGTH,
        EMBEDDING_DIM
    )(x)

    # Transformer encoder
    x = TransformerEncoderBlock(
        d_model=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=0.1
    )(x)

    # Classification layers
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------------------
# 5. Build LSTM Classifier
# -----------------------------------------

def build_lstm_model():

    inputs = layers.Input(
        shape=(MAX_LENGTH,),
        dtype="int32"
    )

    x = layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )(inputs)

    x = layers.LSTM(64)(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------------------
# 6. Load and Prepare IMDB Dataset
# -----------------------------------------

print("\nLoading IMDB dataset...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

x_train = pad_sequences(
    x_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

x_test = pad_sequences(
    x_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# -----------------------------------------
# 7. Train and Evaluate Transformer
# -----------------------------------------

print("\nTraining Transformer Encoder...")

transformer_model = build_transformer_model()

transformer_model.summary()

start_time = time.perf_counter()

transformer_history = transformer_model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

transformer_time = time.perf_counter() - start_time

transformer_loss, transformer_accuracy = (
    transformer_model.evaluate(
        x_test,
        y_test,
        verbose=0
    )
)

print("\nTransformer Test Accuracy:", transformer_accuracy)
print("Transformer Test Loss:", transformer_loss)
print("Transformer Training Time:", transformer_time)


# -----------------------------------------
# 8. Train and Evaluate LSTM
# -----------------------------------------

print("\nTraining LSTM Model...")

lstm_model = build_lstm_model()

lstm_model.summary()

start_time = time.perf_counter()

lstm_history = lstm_model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

lstm_time = time.perf_counter() - start_time

lstm_loss, lstm_accuracy = lstm_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nLSTM Test Accuracy:", lstm_accuracy)
print("LSTM Test Loss:", lstm_loss)
print("LSTM Training Time:", lstm_time)


# -----------------------------------------
# 9. Compare Results
# -----------------------------------------

print("\n========== MODEL COMPARISON ==========")

print(f"Transformer Accuracy: {transformer_accuracy:.4f}")
print(f"LSTM Accuracy:        {lstm_accuracy:.4f}")

print(f"Transformer Time: {transformer_time:.2f} seconds")
print(f"LSTM Time:        {lstm_time:.2f} seconds")

# Save comparison results
with open(
    "outputs/model_comparison.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Test Accuracy",
        "Test Loss",
        "Training Time Seconds"
    ])

    writer.writerow([
        "Transformer",
        transformer_accuracy,
        transformer_loss,
        transformer_time
    ])

    writer.writerow([
        "LSTM",
        lstm_accuracy,
        lstm_loss,
        lstm_time
    ])


# -----------------------------------------
# 10. Plot Accuracy Comparison
# -----------------------------------------

models = ["Transformer", "LSTM"]

accuracies = [
    transformer_accuracy,
    lstm_accuracy
]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.ylabel("Test Accuracy")
plt.title("Transformer vs LSTM - Test Accuracy")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "outputs/accuracy_comparison.png"
)

plt.close()


# -----------------------------------------
# 11. Plot Training Time Comparison
# -----------------------------------------

training_times = [
    transformer_time,
    lstm_time
]

plt.figure(figsize=(8, 5))

plt.bar(models, training_times)

plt.ylabel("Training Time (seconds)")
plt.title("Transformer vs LSTM - Training Time")

plt.tight_layout()

plt.savefig(
    "outputs/training_time_comparison.png"
)

plt.close()


# -----------------------------------------
# 12. Plot Training Accuracy
# -----------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    transformer_history.history["accuracy"],
    label="Transformer Training"
)

plt.plot(
    transformer_history.history["val_accuracy"],
    label="Transformer Validation"
)

plt.plot(
    lstm_history.history["accuracy"],
    label="LSTM Training"
)

plt.plot(
    lstm_history.history["val_accuracy"],
    label="LSTM Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/training_accuracy.png"
)

plt.close()


# -----------------------------------------
# 13. Save Models
# -----------------------------------------

transformer_model.save(
    "models/transformer_model.keras"
)

lstm_model.save(
    "models/lstm_model.keras"
)

print("\nBoth models saved successfully!")

print("\nGenerated output files:")
print("- outputs/model_comparison.csv")
print("- outputs/accuracy_comparison.png")
print("- outputs/training_time_comparison.png")
print("- outputs/training_accuracy.png")

print("\nDay 68 Transformer Encoder Project Completed!")
