
import re
import html
import tensorflow as tf

VOCAB_SIZE = 10000
MAX_LENGTH = 150


def clean_text(text):
    text = html.unescape(text)
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def create_vectorizer(texts):

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LENGTH,
        standardize=None,
        name="text_vectorizer"
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        texts
    ).batch(256)

    vectorizer.adapt(dataset)

    return vectorizer


def decode_imdb_reviews(sequences):

    word_index = tf.keras.datasets.imdb.get_word_index()

    reverse_index = {
        value + 3: key
        for key, value in word_index.items()
    }

    reviews = []

    for sequence in sequences:

        words = [
            reverse_index.get(token, "unknown")
            for token in sequence
            if token > 2
        ]

        review = " ".join(words)

        reviews.append(clean_text(review))

    return reviews
