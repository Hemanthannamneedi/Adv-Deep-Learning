
# main.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from prompt_library import build_prompt


MODEL_NAME = "google/flan-t5-small"

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading FLAN-T5 model...")
print("Device:", device)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("Model loaded successfully!")


def generate_response(
    task_type,
    input_text,
    context="",
    technique="zero_shot",
    examples=None,
    role=None
):
    """
    Construct a prompt and generate an LLM response.
    """

    prompt = build_prompt(
        task_type=task_type,
        input_text=input_text,
        context=context,
        technique=technique,
        examples=examples,
        role=role
    )

    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate response
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "task": task_type,
        "technique": technique,
        "prompt": prompt,
        "response": response
    }


def run_experiments():

    experiments = [

        {
            "task_type": "classification",
            "input_text": (
                "The product quality is excellent. "
                "I love it!"
            ),
            "technique": "zero_shot"
        },

        {
            "task_type": "summarization",
            "input_text": (
                "Artificial Intelligence is a technology "
                "that allows machines to perform tasks "
                "that normally require human intelligence. "
                "It is used in healthcare, education, "
                "business, and transportation."
            ),
            "technique": "zero_shot"
        },

        {
            "task_type": "question_answering",
            "input_text": "What is Python used for?",
            "context": (
                "Python is a programming language used "
                "for web development, data science, "
                "automation, and artificial intelligence."
            ),
            "technique": "few_shot",
            "examples": [
                {
                    "input": "What is HTML used for?",
                    "output": "HTML is used to structure web pages."
                },
                {
                    "input": "What is CSS used for?",
                    "output": "CSS is used to style web pages."
                }
            ]
        },

        {
            "task_type": "code_generation",
            "input_text": (
                "Write a Python function to find "
                "the largest number in a list."
            ),
            "technique": "chain_of_thought",
            "role": "You are a senior Python developer."
        },

        {
            "task_type": "creative_writing",
            "input_text": (
                "Write a short story about a robot "
                "that learns the meaning of friendship."
            ),
            "technique": "zero_shot"
        }
    ]

    results = []

    for index, experiment in enumerate(experiments, 1):

        print("\n" + "=" * 60)
        print(f"EXPERIMENT {index}")
        print("=" * 60)

        result = generate_response(**experiment)

        print("\nTask:", result["task"])
        print("Technique:", result["technique"])
        print("\nGenerated Response:")
        print(result["response"])

        results.append(result)

    # Save experiment results
    with open(
        "outputs/prompt_results.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for index, result in enumerate(results, 1):

            file.write(f"\nEXPERIMENT {index}\n")
            file.write("=" * 50 + "\n")
            file.write(f"Task: {result['task']}\n")
            file.write(
                f"Technique: {result['technique']}\n"
            )

            file.write("\nPROMPT:\n")
            file.write(result["prompt"])

            file.write("\n\nRESPONSE:\n")
            file.write(result["response"])
            file.write("\n\n")

    print("\nAll experiments completed!")
    print(
        "Results saved in outputs/prompt_results.txt"
    )


if __name__ == "__main__":
    run_experiments()
