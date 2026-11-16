# Day 66 – Attention Mechanism Fundamentals

## Overview

Implemented the Scaled Dot-Product Attention mechanism from scratch using NumPy. This project demonstrates how Query, Key, and Value matrices work together to calculate attention scores and generate contextual representations.

## Concepts Covered

- Attention Mechanism Fundamentals
- Query (Q), Key (K), Value (V)
- Dot Product Similarity
- Scaled Attention Scores
- Softmax Function
- Attention Weights
- Attention Output
- Attention Weight Visualization

## Technologies Used

- Python
- NumPy
- Matplotlib

## Implementation

1. Created input embedding representations.
2. Generated Query, Key, and Value matrices.
3. Calculated similarity scores using matrix multiplication.
4. Scaled the scores using the square root of the key dimension.
5. Applied Softmax to obtain attention weights.
6. Calculated the final attention output.
7. Visualized attention weights using a heatmap.

## How to Run

Install dependencies:

```bash
pip install numpy matplotlib
```

Run the implementation:

```bash
python attention.py
```

## Output

The attention weight visualization is saved in:

`outputs/attention_weights.png`

## Learning Outcome

Understood the mathematical foundation of the Attention Mechanism and how Query, Key, and Value matrices help neural networks focus on relevant information in a sequence.
