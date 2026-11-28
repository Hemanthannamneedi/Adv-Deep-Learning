
import os
import json
import time
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)

# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 60

os.makedirs("outputs", exist_ok=True)

set_seed(42)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 55)
print("DAY 71 - LARGE LANGUAGE MODEL EXPERIMENTS")
print("=" * 55)

print("Selected device:", device)


# ==========================================
# Load Pre-trained GPT-2
# ==========================================

print("\nLoading GPT-2 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

print("Loading GPT-2 model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("\nGPT-2 loaded successfully!")

print("Model:", MODEL_NAME)
print("Parameters:", sum(
    parameter.numel()
    for parameter in model.parameters()
))


# ==========================================
# Prompt Experiments
# ==========================================

experiments = [
    {
        "name": "Simple Prompt",
        "prompt": "Artificial Intelligence is",
        "description": "A short and simple prompt."
    },
    {
        "name": "Detailed Prompt",
        "prompt": (
            "Write a short educational paragraph "
            "explaining Artificial Intelligence "
            "in simple English for beginners:"
        ),
        "description": "A prompt with clear context."
    },
    {
        "name": "Creative Prompt",
        "prompt": (
            "Write a short science fiction story "
            "about a robot that discovers emotions:"
        ),
        "description": "A creative writing prompt."
    },
    {
        "name": "Role-based Prompt",
        "prompt": (
            "You are a Python programming teacher. "
            "Explain Python functions in simple "
            "English with one example:"
        ),
        "description": "A prompt that provides a role."
    },
    {
        "name": "Few-shot Prompt",
        "prompt": (
            "Classify the sentiment of each review.\n"
            "Review: This movie was wonderful.\n"
            "Sentiment: Positive\n"
            "Review: The movie was boring.\n"
            "Sentiment: Negative\n"
            "Review: I really enjoyed the film.\n"
            "Sentiment:"
        ),
        "description": "A prompt with two examples."
    }
]


# ==========================================
# Text Generation Function
# ==========================================

def generate_text(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    start_time = time.perf_counter()

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    elapsed = time.perf_counter() - start_time

    # Decode only newly generated tokens
    new_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return generated_text.strip(), elapsed


# ==========================================
# Run All Experiments
# ==========================================

results = []

print("\nStarting prompt experiments...\n")

for index, experiment in enumerate(experiments, start=1):

    print("-" * 55)

    print(
        f"Experiment {index}: "
        f"{experiment['name']}"
    )

    print("Prompt:", experiment["prompt"])

    generated_text, elapsed = generate_text(
        experiment["prompt"]
    )

    print("\nGenerated Output:")
    print(generated_text)

    print(
        "\nGeneration Time:",
        round(elapsed, 3),
        "seconds"
    )

    results.append({
        "experiment": experiment["name"],
        "description": experiment["description"],
        "prompt": experiment["prompt"],
        "generated_text": generated_text,
        "generation_time_seconds": round(
            elapsed, 3
        )
    })


# ==========================================
# Save JSON Results
# ==========================================

with open(
    "outputs/generation_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# Save CSV Comparison
# ==========================================

dataframe = pd.DataFrame(results)

dataframe.to_csv(
    "outputs/prompt_comparison.csv",
    index=False,
    encoding="utf-8"
)


# ==========================================
# Generate Experiment Report
# ==========================================

with open(
    "outputs/experiment_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("DAY 71 - LLM PROMPT EXPERIMENT REPORT\n")
    file.write("=" * 50 + "\n\n")

    for result in results:

        file.write(
            f"Experiment: {result['experiment']}\n"
        )

        file.write(
            f"Description: {result['description']}\n"
        )

        file.write(
            f"Prompt: {result['prompt']}\n"
        )

        file.write(
            f"Generated Text: {result['generated_text']}\n"
        )

        file.write(
            f"Generation Time: "
            f"{result['generation_time_seconds']} seconds\n"
        )

        file.write("-" * 50 + "\n\n")


print("\n" + "=" * 55)
print("ALL EXPERIMENTS COMPLETED!")
print("=" * 55)

print("\nGenerated files:")

print("1. outputs/generation_results.json")
print("2. outputs/prompt_comparison.csv")
print("3. outputs/experiment_report.txt")

print("\nDay 71 LLM Experiments Completed!")
