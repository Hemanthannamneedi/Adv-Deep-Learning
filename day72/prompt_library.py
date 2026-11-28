
# prompt_library.py

"""
Centralized Prompt Template Library
Day 72 - Prompt Engineering Techniques
"""

PROMPT_TEMPLATES = {

    "classification": """
Instruction:
Classify the given text into one of these categories:
Positive, Negative, or Neutral.

Context:
You are an accurate sentiment classification assistant.

Input Data:
{input}

Output Indicator:
Return only the sentiment category.
""",

    "summarization": """
Instruction:
Summarize the following text.

Context:
You are a professional summarization assistant.
Keep the summary simple and clear.

Input Data:
{input}

Output Indicator:
Provide a short summary in 3 to 5 sentences.
""",

    "question_answering": """
Instruction:
Answer the question using the given context.

Context:
{context}

Question:
{input}

Output Indicator:
Provide a clear and accurate answer.
If the answer is not available in the context, say so.
""",

    "code_generation": """
Instruction:
Generate Python code for the following requirement.

Context:
You are an experienced Python developer.
Write clean and readable code.

Input Data:
{input}

Output Indicator:
Return the Python code with a short explanation.
""",

    "creative_writing": """
Instruction:
Create an engaging creative story.

Context:
You are a creative writer.

Input Data:
{input}

Output Indicator:
Write an interesting short story.
"""
}


def build_prompt(
    task_type,
    input_text,
    context="",
    technique="zero_shot",
    examples=None,
    role=None
):
    """
    Construct an optimized prompt based on task type
    and prompting technique.
    """

    task_type = task_type.lower()
    technique = technique.lower()

    if task_type not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unsupported task type: {task_type}"
        )

    template = PROMPT_TEMPLATES[task_type]

    prompt = template.format(
        input=input_text,
        context=context
    )

    # Role-based prompting
    if role:
        prompt = f"Role: {role}\n\n" + prompt

    # Few-shot prompting
    if technique == "few_shot":
        if not examples:
            raise ValueError(
                "Few-shot prompting requires examples."
            )

        example_text = "\n".join(
            [
                f"Example {i + 1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
                for i, ex in enumerate(examples)
            ]
        )

        prompt = (
            "Learn the pattern from these examples:\n\n"
            + example_text
            + "\n\nNow solve the following task:\n\n"
            + prompt
        )

    # Chain-of-thought style prompting
    elif technique == "chain_of_thought":
        prompt += (
            "\n\nSolve the task carefully. "
            "Provide the final answer with a brief explanation "
            "of the main steps."
        )

    elif technique != "zero_shot":
        raise ValueError(
            f"Unsupported technique: {technique}"
        )

    return prompt
