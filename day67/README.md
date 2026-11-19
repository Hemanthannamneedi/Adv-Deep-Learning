# Day 67 – Multi-Head Attention

## Overview

Implemented Multi-Head Attention using TensorFlow and Keras. The project demonstrates how multiple attention heads process input representations in parallel and combine their outputs through concatenation and linear projection.

## Concepts Covered

- Self-Attention
- Multi-Head Attention
- Parallel Attention Computation
- Attention Heads
- Concatenation
- Linear Projection
- Attention Weight Visualization

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Experiments

Tested three configurations:

- 2 Attention Heads
- 4 Attention Heads
- 8 Attention Heads

Compared:

- Output shapes
- Attention score shapes
- Execution time
- Trainable parameter counts

## How to Run

Install dependencies:

```bash
pip install tensorflow numpy matplotlib
```

Run:

```bash
python multi_head_attention.py
```

## Output Files

- `outputs/attention_2_heads.png`
- `outputs/attention_4_heads.png`
- `outputs/attention_8_heads.png`
- `outputs/execution_time_comparison.png`
- `outputs/parameter_comparison.png`

## Learning Outcome

Understood how multiple attention heads capture different relationships within input sequences and how their outputs are combined to generate contextual representations.
