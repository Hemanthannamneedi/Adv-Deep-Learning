# Day 69 Assignment – English-to-French Machine Translation

## Objective

Implement a complete Encoder-Decoder Transformer for English-to-French machine translation using TensorFlow and Keras.

## Architecture

### Transformer Encoder

- Token Embedding
- Sinusoidal Positional Encoding
- Multi-Head Self-Attention
- Residual Connections
- Layer Normalization
- Feed-Forward Network

### Transformer Decoder

- Token Embedding
- Sinusoidal Positional Encoding
- Masked Multi-Head Self-Attention
- Encoder-Decoder Cross-Attention
- Residual Connections
- Layer Normalization
- Feed-Forward Network
- Dense Softmax Output Layer

## Dataset

A custom educational dataset containing 60 English-French sentence pairs.

## Training Configuration

- Embedding Dimension: 64
- Attention Heads: 4
- Feed-Forward Dimension: 128
- Maximum Sequence Length: 12
- Batch Size: 16
- Epochs: 100
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

## Masked Self-Attention

Causal masking prevents the decoder from attending to future tokens during training and inference.

## Cross-Attention

Queries are generated from the decoder, while keys and values are obtained from the encoder output.

## Translation Process

During training, teacher forcing is used with shifted target sequences. During inference, the decoder generates French tokens autoregressively until the end token is produced or the maximum sequence length is reached.

## Results

Translation predictions are recorded in `outputs/translation_results.json`.

Training accuracy, loss, and computational time are recorded in `outputs/training_summary.json`.

## Limitations

The model uses a small educational dataset. Training accuracy and translations of previously seen sentences do not establish generalization to unseen sentences. A larger dataset and independent test set are required for reliable translation evaluation.

## Conclusion

This project demonstrates the complete Transformer architecture, including masked self-attention, cross-attention, teacher forcing, and autoregressive sequence generation.
