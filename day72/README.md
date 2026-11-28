# Prompt Engineering Library – Day 72

A Python-based Prompt Engineering project using Hugging Face FLAN-T5-Small.

## Features

- Centralized prompt template library
- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought style prompting
- Role-based prompting
- LLM response generation
- Automatic experiment result saving

## Supported Tasks

- Text Classification
- Text Summarization
- Question Answering
- Code Generation
- Creative Writing

## Technologies

- Python
- Hugging Face Transformers
- PyTorch
- FLAN-T5-Small

## Installation

```bash
py -m pip install -r requirements.txt
```

## Run

```bash
py main.py
```

The first execution downloads the pre-trained model.

## Project Structure

```text
day72/
├── prompt_library.py
├── main.py
├── requirements.txt
├── outputs/
│   └── prompt_results.txt
└── docs/
    └── assignment.md
```

## Learning Outcomes

- Understanding prompt engineering
- Building reusable prompt templates
- Comparing prompting techniques
- Integrating Hugging Face models
- Generating LLM responses programmatically
