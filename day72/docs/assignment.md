# Day 72 – Prompt Engineering Techniques

## 1. Objective

The objective of this assignment is to understand different prompt engineering techniques and develop a reusable Python Prompt Template Library.

## 2. Prompt Engineering

Prompt engineering is the process of designing clear and structured instructions to obtain useful responses from Large Language Models.

A well-designed prompt contains:

- Instruction
- Context
- Input Data
- Output Indicator

## 3. Prompting Techniques

### Zero-Shot Prompting

Provides an instruction without any examples.

### Few-Shot Prompting

Provides sample inputs and outputs to guide the model.

### Chain-of-Thought Style Prompting

Requests a brief explanation of the main steps along with the final answer.

### Role-Based Prompting

Assigns a specific role to the model to guide its response style.

## 4. Implementation

Developed a centralized prompt library using Python dictionaries and reusable functions.

Implemented templates for:

1. Classification
2. Summarization
3. Question Answering
4. Code Generation
5. Creative Writing

Created a `build_prompt()` function that accepts task type, input data, context, prompting technique, examples, and role.

Integrated the prompt library with the Hugging Face FLAN-T5-Small model to generate responses.

## 5. Tools and Technologies

- Python
- Hugging Face Transformers
- PyTorch
- FLAN-T5-Small

## 6. Expected Results

The project generates responses for different tasks using multiple prompting techniques.

The generated prompts and responses are saved in `outputs/prompt_results.txt`.

## 7. Conclusion

This project helped me understand how prompt structure and prompting techniques influence LLM outputs. I also learned how to organize reusable prompt templates and connect them to a pre-trained language model.
