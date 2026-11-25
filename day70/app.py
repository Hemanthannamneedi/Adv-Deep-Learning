
import os
import json
import time
import threading
import tensorflow as tf

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Register custom Keras layers before loading
from model import PositionalEncoding, TransformerBlock
from preprocess import clean_text


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "transformer_classifier.keras"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "config.json"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "metrics.json"
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "Model not found. Run python train.py first."
    )

print("Loading Transformer Classifier...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

THRESHOLD = config.get("threshold", 0.5)

prediction_lock = threading.Lock()

app = FastAPI(
    title="Transformer Sentiment Classifier API",
    description=(
        "Text sentiment classification using "
        "a custom Transformer Encoder."
    ),
    version="1.0.0"
)


class TextRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000
    )


class BatchRequest(BaseModel):

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100
    )


def predict_sentiments(texts):

    cleaned_texts = []

    for text in texts:

        if len(text) > 10000:

            raise HTTPException(
                status_code=422,
                detail="Text exceeds 10000 characters."
            )

        cleaned = clean_text(text)

        if not cleaned:

            raise HTTPException(
                status_code=422,
                detail="Text cannot be empty."
            )

        cleaned_texts.append(cleaned)

    input_tensor = tf.constant(
        cleaned_texts,
        dtype=tf.string
    )

    start_time = time.perf_counter()

    with prediction_lock:

        probabilities = model(
            input_tensor,
            training=False
        ).numpy().reshape(-1)

    elapsed = time.perf_counter() - start_time

    results = []

    for text, probability in zip(
        texts,
        probabilities
    ):

        probability = float(probability)

        if probability >= THRESHOLD:

            sentiment = "Positive"
            confidence = probability

        else:

            sentiment = "Negative"
            confidence = 1 - probability

        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "positive_probability": round(
                probability,
                4
            )
        })

    return results, elapsed


@app.get("/")
def home():

    return {
        "message": "Transformer Classifier API",
        "status": "Running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/model-info")
def model_info():

    metrics = {}

    if os.path.exists(METRICS_PATH):

        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

    return {
        "model": "Transformer Encoder",
        "task": "Sentiment Classification",
        "dataset": "IMDB Movie Reviews",
        "classes": ["Negative", "Positive"],
        "configuration": config,
        "metrics": metrics
    }


@app.post("/predict")
def predict(request: TextRequest):

    results, elapsed = predict_sentiments(
        [request.text]
    )

    result = results[0]

    result["inference_time_ms"] = round(
        elapsed * 1000,
        2
    )

    return result


@app.post("/predict-batch")
def predict_batch(request: BatchRequest):

    results, elapsed = predict_sentiments(
        request.texts
    )

    return {
        "total_predictions": len(results),
        "results": results,
        "total_inference_time_ms": round(
            elapsed * 1000,
            2
        ),
        "average_inference_time_ms": round(
            elapsed * 1000 / len(results),
            2
        )
    }
