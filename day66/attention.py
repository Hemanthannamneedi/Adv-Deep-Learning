
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Input tokens
tokens = ["The", "cat", "sat", "on", "mat"]

# Input embedding matrix
X = np.random.rand(5, 4)

print("Input Embeddings:")
print(X)

# Initialize weight matrices
d_model = 4
d_k = 4

Wq = np.random.rand(d_model, d_k)
Wk = np.random.rand(d_model, d_k)
Wv = np.random.rand(d_model, d_k)

# Step 1: Calculate Query, Key, Value
Q = X @ Wq
K = X @ Wk
V = X @ Wv

print("\nQuery Matrix (Q):")
print(Q)

print("\nKey Matrix (K):")
print(K)

print("\nValue Matrix (V):")
print(V)

# Step 2: Calculate attention scores
scores = Q @ K.T

print("\nRaw Attention Scores:")
print(scores)

# Step 3: Scale scores
scaled_scores = scores / np.sqrt(d_k)

# Step 4: Apply Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)

print("\nAttention Weights:")
print(attention_weights)

# Step 5: Calculate attention output
attention_output = attention_weights @ V

print("\nAttention Output:")
print(attention_output)

# Verify attention weights
print("\nRow sums:")
print(np.sum(attention_weights, axis=1))

# Visualize attention weights
plt.figure(figsize=(8, 6))

plt.imshow(attention_weights, cmap="Blues")

plt.xticks(range(len(tokens)), tokens)
plt.yticks(range(len(tokens)), tokens)

plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.title("Scaled Dot-Product Attention Weights")

plt.colorbar(label="Attention Weight")
plt.tight_layout()

plt.savefig("outputs/attention_weights.png")
plt.show()
