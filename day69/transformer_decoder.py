
import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model

# ==========================================
# 1. Configuration
# ==========================================

tf.keras.utils.set_random_seed(42)

MAX_LENGTH = 12
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
BATCH_SIZE = 16
EPOCHS = 100

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==========================================
# 2. English-French Translation Dataset
# ==========================================

sentence_pairs = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good evening", "bonsoir"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("thank you very much", "merci beaucoup"),
    ("please", "s il vous plait"),
    ("yes", "oui"),
    ("no", "non"),
    ("goodbye", "au revoir"),
    ("see you soon", "a bientot"),
    ("how are you", "comment allez vous"),
    ("i am fine", "je vais bien"),
    ("what is your name", "comment vous appelez vous"),
    ("my name is john", "je m appelle john"),
    ("i am happy", "je suis heureux"),
    ("i am sad", "je suis triste"),
    ("i am tired", "je suis fatigue"),
    ("i am hungry", "j ai faim"),
    ("i am thirsty", "j ai soif"),
    ("i love you", "je t aime"),
    ("i love music", "j aime la musique"),
    ("i love football", "j aime le football"),
    ("i like apples", "j aime les pommes"),
    ("i like books", "j aime les livres"),
    ("i like coffee", "j aime le cafe"),
    ("i like tea", "j aime le the"),
    ("i drink water", "je bois de l eau"),
    ("i eat bread", "je mange du pain"),
    ("i eat rice", "je mange du riz"),
    ("i read books", "je lis des livres"),
    ("i speak english", "je parle anglais"),
    ("i speak french", "je parle francais"),
    ("where are you", "ou es tu"),
    ("where is the station", "ou est la gare"),
    ("where is the hotel", "ou est l hotel"),
    ("this is my book", "c est mon livre"),
    ("this is my house", "c est ma maison"),
    ("this is my car", "c est ma voiture"),
    ("the sky is blue", "le ciel est bleu"),
    ("the sun is hot", "le soleil est chaud"),
    ("the water is cold", "l eau est froide"),
    ("he is happy", "il est heureux"),
    ("she is happy", "elle est heureuse"),
    ("he is a student", "il est etudiant"),
    ("she is a teacher", "elle est enseignante"),
    ("we are friends", "nous sommes amis"),
    ("we love music", "nous aimons la musique"),
    ("open the door", "ouvre la porte"),
    ("close the door", "ferme la porte"),
    ("come here", "viens ici"),
    ("sit down", "assieds toi"),
    ("help me", "aide moi"),
    ("i need help", "j ai besoin d aide"),
    ("i understand", "je comprends"),
    ("i do not understand", "je ne comprends pas"),
    ("have a nice day", "bonne journee"),
    ("welcome", "bienvenue"),
    ("excuse me", "excusez moi"),
    ("i am sorry", "je suis desole")
]

print("Total sentence pairs:", len(sentence_pairs))


# ==========================================
# 3. Vocabulary and Tokenization
# ==========================================

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<start>",
    "<end>"
]

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


def build_vocab(sentences):

    words = set()

    for sentence in sentences:
        words.update(sentence.lower().split())

    vocabulary = SPECIAL_TOKENS + sorted(words)

    return {
        word: index
        for index, word in enumerate(vocabulary)
    }


english_vocab = build_vocab(
    [pair[0] for pair in sentence_pairs]
)

french_vocab = build_vocab(
    [pair[1] for pair in sentence_pairs]
)

reverse_french_vocab = {
    index: word
    for word, index in french_vocab.items()
}

print("English vocabulary:", len(english_vocab))
print("French vocabulary:", len(french_vocab))


def encode_sentence(sentence, vocabulary):

    tokens = sentence.lower().split()

    return [
        vocabulary.get(token, UNK_ID)
        for token in tokens
    ]


def pad_sequence(sequence):

    sequence = sequence[:MAX_LENGTH]

    return sequence + [PAD_ID] * (
        MAX_LENGTH - len(sequence)
    )


# ==========================================
# 4. Prepare Encoder and Decoder Inputs
# ==========================================

encoder_inputs = []
decoder_inputs = []
decoder_targets = []

for english, french in sentence_pairs:

    english_ids = encode_sentence(
        english,
        english_vocab
    )

    french_ids = encode_sentence(
        french,
        french_vocab
    )

    # Teacher forcing:
    # Input:  <start> je suis heureux
    # Target: je suis heureux <end>

    decoder_input = [START_ID] + french_ids

    decoder_target = french_ids + [END_ID]

    encoder_inputs.append(
        pad_sequence(english_ids)
    )

    decoder_inputs.append(
        pad_sequence(decoder_input)
    )

    decoder_targets.append(
        pad_sequence(decoder_target)
    )


encoder_inputs = np.array(
    encoder_inputs,
    dtype=np.int32
)

decoder_inputs = np.array(
    decoder_inputs,
    dtype=np.int32
)

decoder_targets = np.array(
    decoder_targets,
    dtype=np.int32
)

sample_weights = (
    decoder_targets != PAD_ID
).astype(np.float32)

print("\nEncoder input shape:", encoder_inputs.shape)
print("Decoder input shape:", decoder_inputs.shape)
print("Decoder target shape:", decoder_targets.shape)


# ==========================================
# 5. Sinusoidal Positional Encoding
# ==========================================

class PositionalEncoding(layers.Layer):

    def __init__(self, max_length, d_model):
        super().__init__()

        positions = np.arange(max_length)[:, None]
        dimensions = np.arange(d_model)[None, :]

        angles = positions / np.power(
            10000,
            (2 * (dimensions // 2)) / d_model
        )

        encoding = np.zeros(
            (max_length, d_model)
        )

        encoding[:, 0::2] = np.sin(
            angles[:, 0::2]
        )

        encoding[:, 1::2] = np.cos(
            angles[:, 1::2]
        )

        self.encoding = tf.constant(
            encoding[None, :, :],
            dtype=tf.float32
        )

    def call(self, inputs):

        length = tf.shape(inputs)[1]

        return inputs + self.encoding[:, :length, :]


# ==========================================
# 6. Transformer Encoder Block
# ==========================================

class TransformerEncoder(layers.Layer):

    def __init__(self):
        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=D_MODEL // NUM_HEADS
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(FF_DIM, activation="relu"),
            layers.Dense(D_MODEL)
        ])

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, training=None):

        x, token_ids = inputs

        padding_mask = tf.not_equal(
            token_ids,
            PAD_ID
        )

        attention_mask = padding_mask[:, None, :]

        attention_output = self.attention(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        x = self.norm1(x + attention_output)

        ffn_output = self.ffn(
            x,
            training=training
        )

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.norm2(x + ffn_output)


# ==========================================
# 7. Transformer Decoder Block
# ==========================================

class TransformerDecoder(layers.Layer):

    def __init__(self):
        super().__init__()

        # Masked self-attention
        self.masked_attention = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=D_MODEL // NUM_HEADS
        )

        # Encoder-decoder cross-attention
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=D_MODEL // NUM_HEADS
        )

        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(FF_DIM, activation="relu"),
            layers.Dense(D_MODEL)
        ])

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm3 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        self.dropout3 = layers.Dropout(0.1)

    def call(self, inputs, training=None):

        x, encoder_output, source_ids, target_ids = inputs

        # Ignore decoder padding tokens
        target_padding_mask = tf.not_equal(
            target_ids,
            PAD_ID
        )[:, None, :]

        # Step 1: Masked Self-Attention
        self_attention_output = self.masked_attention(
            query=x,
            key=x,
            value=x,
            attention_mask=target_padding_mask,
            use_causal_mask=True,
            training=training
        )

        self_attention_output = self.dropout1(
            self_attention_output,
            training=training
        )

        x = self.norm1(
            x + self_attention_output
        )

        # Ignore encoder padding tokens
        source_padding_mask = tf.not_equal(
            source_ids,
            PAD_ID
        )[:, None, :]

        # Step 2: Cross-Attention
        cross_attention_output = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attention_mask=source_padding_mask,
            training=training
        )

        cross_attention_output = self.dropout2(
            cross_attention_output,
            training=training
        )

        x = self.norm2(
            x + cross_attention_output
        )

        # Step 3: Feed-Forward Network
        ffn_output = self.ffn(
            x,
            training=training
        )

        ffn_output = self.dropout3(
            ffn_output,
            training=training
        )

        return self.norm3(x + ffn_output)


# ==========================================
# 8. Build Encoder Model
# ==========================================

source_input = layers.Input(
    shape=(MAX_LENGTH,),
    dtype="int32",
    name="encoder_input"
)

encoder_embedding = layers.Embedding(
    input_dim=len(english_vocab),
    output_dim=D_MODEL
)(source_input)

encoder_embedding = PositionalEncoding(
    MAX_LENGTH,
    D_MODEL
)(encoder_embedding)

encoder_output = TransformerEncoder()([
    encoder_embedding,
    source_input
])

encoder_model = Model(
    source_input,
    encoder_output,
    name="encoder"
)


# ==========================================
# 9. Build Decoder Model
# ==========================================

target_input = layers.Input(
    shape=(MAX_LENGTH,),
    dtype="int32",
    name="decoder_input"
)

encoded_source = layers.Input(
    shape=(MAX_LENGTH, D_MODEL),
    name="encoder_output"
)

decoder_source_ids = layers.Input(
    shape=(MAX_LENGTH,),
    dtype="int32",
    name="source_ids"
)

decoder_embedding = layers.Embedding(
    input_dim=len(french_vocab),
    output_dim=D_MODEL
)(target_input)

decoder_embedding = PositionalEncoding(
    MAX_LENGTH,
    D_MODEL
)(decoder_embedding)

decoder_output = TransformerDecoder()([
    decoder_embedding,
    encoded_source,
    decoder_source_ids,
    target_input
])

predictions = layers.Dense(
    len(french_vocab),
    activation="softmax"
)(decoder_output)

decoder_model = Model(
    [
        target_input,
        encoded_source,
        decoder_source_ids
    ],
    predictions,
    name="decoder"
)


# ==========================================
# 10. Complete Encoder-Decoder Transformer
# ==========================================

translation_output = decoder_model([
    target_input,
    encoder_model(source_input),
    source_input
])

model = Model(
    [source_input, target_input],
    translation_output,
    name="translation_transformer"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    weighted_metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="token_accuracy"
        )
    ]
)

model.summary()


# ==========================================
# 11. Train Transformer Model
# ==========================================

print("\nTraining Transformer...")

start_time = time.perf_counter()

history = model.fit(
    [encoder_inputs, decoder_inputs],
    decoder_targets,
    sample_weight=sample_weights,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

training_time = time.perf_counter() - start_time

print(
    f"\nTraining Time: {training_time:.2f} seconds"
)


# ==========================================
# 12. Autoregressive Translation
# ==========================================


def translate(sentence):

    source_sequence = encode_sentence(
        sentence,
        english_vocab
    )

    source_sequence = tf.convert_to_tensor(
        [pad_sequence(source_sequence)],
        dtype=tf.int32
    )

    # Encode source sentence once
    encoded = encoder_model(
        source_sequence,
        training=False
    )

    generated_tokens = [START_ID]

    for _ in range(MAX_LENGTH - 1):

        decoder_sequence = tf.convert_to_tensor(
            [pad_sequence(generated_tokens)],
            dtype=tf.int32
        )

        probabilities = decoder_model(
            [
                decoder_sequence,
                encoded,
                source_sequence
            ],
            training=False
        )

        current_position = len(generated_tokens) - 1

        next_token_probs = probabilities[
            0,
            current_position
        ].numpy().copy()

        next_token_probs[PAD_ID] = -1
        next_token_probs[START_ID] = -1

        next_token = int(
            np.argmax(next_token_probs)
        )

        if next_token == END_ID:
            break

        generated_tokens.append(next_token)

    translated_words = [
        reverse_french_vocab.get(token, "<unk>")
        for token in generated_tokens[1:]
    ]

    return " ".join(translated_words)



# ==========================================
# 13. Test Translation
# ==========================================

test_sentences = [
    "hello",
    "thank you",
    "i love you",
    "good morning",
    "i am happy",
    "i speak french",
    "where is the station",
    "i like music",
    "we like coffee"
]

print("\n========== TRANSLATION RESULTS ==========")

translation_results = []

for sentence in test_sentences:

    translation = translate(sentence)

    print(f"\nEnglish: {sentence}")
    print(f"French:  {translation}")

    translation_results.append({
        "english": sentence,
        "predicted_french": translation
    })

with open(
    "outputs/translation_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        translation_results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# 14. Visualize Causal Attention Mask
# ==========================================

mask = np.tril(
    np.ones((MAX_LENGTH, MAX_LENGTH))
)

plt.figure(figsize=(8, 6))

plt.imshow(
    mask,
    cmap="Blues",
    vmin=0,
    vmax=1
)

plt.xlabel("Key Token Position")
plt.ylabel("Query Token Position")
plt.title("Decoder Causal Attention Mask")

plt.colorbar(
    ticks=[0, 1],
    label="0 = Blocked, 1 = Allowed"
)

plt.tight_layout()

plt.savefig(
    "outputs/causal_attention_mask.png"
)

plt.close()


# ==========================================
# 15. Plot Training Accuracy
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["token_accuracy"],
    label="Training Token Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transformer Training Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/training_accuracy.png"
)

plt.close()


# ==========================================
# 16. Plot Training Loss
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/training_loss.png"
)

plt.close()


# ==========================================
# 17. Save Model Weights and Vocabulary
# ==========================================

model.save_weights(
    "models/transformer.weights.h5"
)

with open(
    "models/vocabulary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump({
        "english_vocab": english_vocab,
        "french_vocab": french_vocab
    }, file, indent=4)

with open(
    "outputs/training_summary.json",
    "w"
) as file:

    json.dump({
        "epochs": EPOCHS,
        "sentence_pairs": len(sentence_pairs),
        "training_time_seconds": training_time,
        "final_training_accuracy": float(
            history.history["token_accuracy"][-1]
        ),
        "final_training_loss": float(
            history.history["loss"][-1]
        )
    }, file, indent=4)

print("\nModel weights saved successfully!")
print("Translation results saved successfully!")
print("Training visualizations generated!")

print("\nDay 69 Transformer Decoder Completed!")
