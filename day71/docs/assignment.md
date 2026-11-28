# Day 71 – Introduction to Large Language Models

## Objective

Understand GPT-style Large Language Models and explore text generation using a pre-trained GPT-2 model from Hugging Face.

## Model Used

- Model: GPT-2
- Framework: Hugging Face Transformers
- Deep Learning Backend: PyTorch
- Task: Autoregressive Text Generation

## GPT-2 Architecture

GPT-2 is based on the Transformer Decoder architecture. It uses masked self-attention and feed-forward neural network layers to predict the next token based on previous tokens.

## Implementation

The project loads the pre-trained GPT-2 tokenizer and language model using Hugging Face Transformers.

Text is generated using the model's `generate()` method with sampling parameters.

## Prompt Engineering Experiments

Five prompt types were tested:

1. Simple Prompt – Provides a short text input.
2. Detailed Prompt – Includes context and a clear task.
3. Creative Prompt – Requests a short creative story.
4. Role-based Prompt – Assigns a role to guide the response.
5. Few-shot Prompt – Provides examples to demonstrate an expected output pattern.

## Generation Settings

- Maximum New Tokens: 60
- Temperature: 0.8
- Top-p: 0.9
- Repetition Penalty: 1.15

## Results

Generated text and execution times are stored in:

- `outputs/generation_results.json`
- `outputs/prompt_comparison.csv`
- `outputs/experiment_report.txt`

## Observations

Prompt wording and context influence the generated continuation.

Detailed prompts provide more context than short prompts. Role-based prompts establish a requested perspective. Few-shot prompts demonstrate an expected pattern through examples.

However, GPT-2 is a base language model, so it may continue the prompt rather than follow instructions reliably.

## Conclusion

This project demonstrates how to load a pre-trained GPT-2 model, generate text using Hugging Face Transformers, and compare different prompting techniques.

It provides practical experience with autoregressive text generation and introductory prompt engineering.
