
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model


@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEncoding(layers.Layer):

    def __init__(self, max_length, d_model, **kwargs):

        super().__init__(**kwargs)

        self.max_length = max_length
        self.d_model = d_model

        positions = np.arange(max_length)[:, None]
        dimensions = np.arange(d_model)[None, :]

        angles = positions / np.power(
            10000,
            (2 * (dimensions // 2)) / d_model
        )

        encoding = np.zeros(
            (max_length, d_model),
            dtype=np.float32
        )

        encoding[:, 0::2] = np.sin(
            angles[:, 0::2]
        )

        encoding[:, 1::2] = np.cos(
            angles[:, 1::2]
        )

        self.encoding = tf.constant(
            encoding[None, :, :]
        )

        self.supports_masking = True

    def call(self, inputs):

        length = tf.shape(inputs)[1]

        return inputs + self.encoding[:, :length, :]

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):

        config = super().get_config()

        config.update({
            "max_length": self.max_length,
            "d_model": self.d_model
        })

        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout=0.1,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

        self.supports_masking = True

    def build(self, input_shape):

        self.attention.build(
            input_shape,
            input_shape
        )

        self.ffn.build(input_shape)
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):

        attention_mask = None

        if mask is not None:
            attention_mask = mask[:, None, :]

        attention_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        x = self.norm1(
            inputs + attention_output
        )

        ffn_output = self.ffn(
            x,
            training=training
        )

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.norm2(
            x + ffn_output
        )

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):

        config = super().get_config()

        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })

        return config


def build_model(
    vectorizer,
    max_length=150,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    dropout=0.2,
    learning_rate=0.001
):

    inputs = layers.Input(
        shape=(),
        dtype=tf.string,
        name="review"
    )

    x = vectorizer(inputs)

    x = layers.Embedding(
        input_dim=vectorizer.vocabulary_size(),
        output_dim=d_model,
        mask_zero=True
    )(x)

    x = PositionalEncoding(
        max_length,
        d_model
    )(x)

    x = TransformerBlock(
        d_model,
        num_heads,
        ff_dim,
        dropout
    )(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(dropout)(x)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="sentiment"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="transformer_classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
