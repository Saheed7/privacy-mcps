"""
Hybrid 1D-CNN-BiGRU-Attention Model.

Implements the complete architecture described in Section III-A:
  1D-CNN (Eq. 1-4) → BiGRU (Eq. 5-10) → Multi-Head Attention (Eq. 11-18) → Classification (Eq. 19-22)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, Bidirectional, GRU,
    Dropout, Dense, GlobalAveragePooling1D, Reshape
)
from src.models.attention import MultiHeadSelfAttention


def build_model(
    n_features: int,
    n_classes: int,
    task: str = "binary",
    conv1_filters: int = 128,
    conv2_filters: int = 64,
    kernel_size: int = 3,
    pool_size: int = 2,
    gru_units: int = 64,
    n_heads: int = 4,
    d_model: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> Model:
    """
    Build and compile the 1D-CNN-BiGRU-Attention model.

    Args:
        n_features: Number of input features (F).
        n_classes: Number of output classes (C).
        task: 'binary' or 'multiclass'.
        conv1_filters: Number of filters in first Conv1D layer (K1).
        conv2_filters: Number of filters in second Conv1D layer (K2).
        kernel_size: Convolution kernel width (h).
        pool_size: Max-pooling window size (s).
        gru_units: Hidden units per GRU direction (d_h).
        n_heads: Number of attention heads (N_heads).
        d_model: Attention model dimensionality (must equal 2*gru_units).
        dense_units: Dense layer units before output.
        dropout_rate: Dropout probability (p_drop).
        learning_rate: Adam optimizer learning rate (eta).

    Returns:
        Compiled Keras Model.
    """
    assert d_model == 2 * gru_units, f"d_model ({d_model}) must equal 2*gru_units ({2*gru_units})"

    # Input layer
    inputs = Input(shape=(n_features,), name="input_features")

    # Reshape for Conv1D: (batch, features, 1)
    x = Reshape((n_features, 1), name="reshape_to_2d")(inputs)

    # === 1D-CNN Block 1 (Eq. 1-4) ===
    x = Conv1D(
        filters=conv1_filters, kernel_size=kernel_size,
        activation="relu", padding="valid", name="conv1d_1"
    )(x)
    x = BatchNormalization(name="bn_1")(x)
    x = MaxPooling1D(pool_size=pool_size, name="maxpool_1")(x)

    # === 1D-CNN Block 2 ===
    x = Conv1D(
        filters=conv2_filters, kernel_size=kernel_size,
        activation="relu", padding="valid", name="conv1d_2"
    )(x)
    x = BatchNormalization(name="bn_2")(x)
    x = MaxPooling1D(pool_size=pool_size, name="maxpool_2")(x)

    # === BiGRU (Eq. 5-10) ===
    x = Bidirectional(
        GRU(gru_units, return_sequences=True, name="gru_forward"),
        backward_layer=GRU(gru_units, return_sequences=True, go_backwards=True, name="gru_backward"),
        name="bigru"
    )(x)
    x = Dropout(dropout_rate, name="dropout_bigru")(x)

    # === Multi-Head Self-Attention (Eq. 11-18) ===
    x = MultiHeadSelfAttention(
        d_model=d_model, n_heads=n_heads, name="multi_head_attention"
    )(x)

    # === Global Average Pooling (Eq. 19-20) ===
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)

    # === Dense Classification Head ===
    x = Dense(dense_units, activation="relu", name="dense_1")(x)
    x = Dropout(dropout_rate, name="dropout_dense")(x)

    # === Output Layer ===
    if task == "binary":
        outputs = Dense(1, activation="sigmoid", name="output_binary")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        outputs = Dense(n_classes, activation="softmax", name="output_multi")(x)
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]

    model = Model(inputs=inputs, outputs=outputs, name="CNN_BiGRU_Attention")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def get_model_summary(model: Model) -> str:
    """Return model summary as string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
