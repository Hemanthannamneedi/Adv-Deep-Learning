# LLM Experiments – GPT-2 Text Generation

Day 71 – Advanced Deep Learning

## Overview

A hands-on project exploring Large Language Models using the pre-trained GPT-2 model and Hugging Face Transformers.

The project demonstrates autoregressive text generation and experiments with different prompt engineering techniques.

## Features

- Load a pre-trained GPT-2 model
- Use Hugging Face Transformers
- Generate text from different prompts
- Explore simple and detailed prompting
- Experiment with role-based prompting
- Demonstrate few-shot prompting
- Compare generated outputs
- Save results in JSON, CSV, and TXT formats

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Pandas

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python llm_experiments.py
```

The first execution downloads the GPT-2 model weights from Hugging Face.

## Prompt Experiments

1. Simple Prompt
2. Detailed Prompt
3. Creative Prompt
4. Role-based Prompt
5. Few-shot Prompt

## Generation Parameters

- Model: GPT-2
- Maximum New Tokens: 60
- Temperature: 0.8
- Top-p: 0.9
- Repetition Penalty: 1.15

## Generated Files

- `outputs/generation_results.json`
- `outputs/prompt_comparison.csv`
- `outputs/experiment_report.txt`

## Important Note

GPT-2 is a base language model trained for next-token prediction. It is not specifically fine-tuned to behave like an instruction-following conversational assistant.

Generated outputs may vary and are not guaranteed to be factually correct.

## Learning Outcomes

- Understood GPT-style Transformer architecture.
- Loaded a pre-trained language model.
- Generated text using Hugging Face.
- Explored autoregressive generation.
- Compared different prompt engineering approaches.
