
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Sample input sentence representation
tokens = ["The", "cat", "sat", "on", "the", "mat"]

# Model configuration
batch_size = 1
sequence_length = len(tokens)
embedding_dim = 64

# Random input embeddings
input_data = np.random.rand(
    batch_size,
    sequence_length,
    embedding_dim
).astype(np.float32)

print("Input Shape:", input_data.shape)

# Test different attention head counts
head_counts = [2, 4, 8]

results = []

for num_heads in head_counts:

    print("\n" + "=" * 45)
    print(f"Testing Multi-Head Attention: {num_heads} Heads")
    print("=" * 45)

    # Build Multi-Head Attention layer
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim // num_heads,
        dropout=0.0
    )

    # Warm-up run
    _ = attention_layer(
        input_data,
        input_data,
        training=False
    )

    # Measure execution time
    start_time = time.perf_counter()

    output = attention_layer(
        input_data,
        input_data,
        training=False
    )

    execution_time = time.perf_counter() - start_time

    # Get attention scores
    output_with_scores, attention_scores = attention_layer(
        input_data,
        input_data,
        return_attention_scores=True,
        training=False
    )

    # Model parameters
    total_params = attention_layer.count_params()

    print("Number of Heads:", num_heads)
    print("Output Shape:", output.shape)
    print("Attention Scores Shape:", attention_scores.shape)
    print("Trainable Parameters:", total_params)
    print("Execution Time:", execution_time)

    results.append({
        "heads": num_heads,
        "time": execution_time,
        "parameters": total_params
    })

    # Visualize attention scores for the first sample
    plt.figure(figsize=(8, 5))

    for head in range(num_heads):
        plt.subplot(
            (num_heads + 3) // 4,
            min(num_heads, 4),
            head + 1
        )

        plt.imshow(
            attention_scores[0, head].numpy(),
            cmap="Blues",
            vmin=0,
            vmax=1
        )

        plt.title(f"Head {head + 1}")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")

        plt.xticks(
            range(sequence_length),
            tokens,
            rotation=45
        )

        plt.yticks(range(sequence_length), tokens)

    plt.suptitle(f"Attention Weights - {num_heads} Heads")
    plt.tight_layout()

    plt.savefig(
        f"outputs/attention_{num_heads}_heads.png"
    )

    plt.close()

# Display experiment summary
print("\n\nEXPERIMENT COMPARISON")
print("-" * 65)
print(f"{'Heads':<12}{'Time (sec)':<20}{'Parameters':<15}")

for result in results:
    print(
        f"{result['heads']:<12}"
        f"{result['time']:<20.6f}"
        f"{result['parameters']:<15}"
    )

# Plot execution time comparison
plt.figure(figsize=(8, 5))

plt.bar(
    [str(r["heads"]) for r in results],
    [r["time"] for r in results]
)

plt.xlabel("Number of Attention Heads")
plt.ylabel("Execution Time (seconds)")
plt.title("Multi-Head Attention Execution Time Comparison")

plt.tight_layout()
plt.savefig("outputs/execution_time_comparison.png")
plt.show()

# Plot parameter comparison
plt.figure(figsize=(8, 5))

plt.bar(
    [str(r["heads"]) for r in results],
    [r["parameters"] for r in results]
)

plt.xlabel("Number of Attention Heads")
plt.ylabel("Trainable Parameters")
plt.title("Parameter Count Comparison")

plt.tight_layout()
plt.savefig("outputs/parameter_comparison.png")
plt.show()

print("\nAll experiments completed successfully!")
