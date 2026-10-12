"""CNN architecture for real vs AI-generated image classification."""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import config


def conv_block(x, filters: int, dropout: float = 0.25):
    """Two 3x3 convs + BatchNorm + ReLU, then max-pool and dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_model(input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
                learning_rate: float = config.LEARNING_RATE) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    x = conv_block(inputs, 32)    # 256 -> 128
    x = conv_block(x, 64)         # 128 -> 64
    x = conv_block(x, 128)        # 64  -> 32
    x = conv_block(x, 256)        # 32  -> 16

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_real")(x)

    model = models.Model(inputs, outputs, name="DeepGuardCNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


if __name__ == "__main__":
    build_model().summary()
