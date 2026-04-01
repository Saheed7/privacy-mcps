"""
Multi-Head Self-Attention Mechanism.

Implements scaled dot-product attention with multiple heads
as described in Section III-A(3) of the paper.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout


class MultiHeadSelfAttention(Layer):
    """Multi-Head Self-Attention mechanism (Equations 11-18)."""

    def __init__(self, d_model=128, n_heads=4, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Key dimension per head
        self.d_v = d_model // n_heads  # Value dimension per head
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Query, Key, Value projection matrices (Eq. 11-13)
        self.W_Q = Dense(self.d_model, use_bias=False, name="query_proj")
        self.W_K = Dense(self.d_model, use_bias=False, name="key_proj")
        self.W_V = Dense(self.d_model, use_bias=False, name="value_proj")
        # Output projection (Eq. 17)
        self.W_O = Dense(self.d_model, use_bias=False, name="output_proj")
        self.dropout = Dropout(self.dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def _split_heads(self, x, batch_size):
        """Split the last dimension into (n_heads, d_k)."""
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _scaled_dot_product_attention(self, Q, K, V):
        """Scaled dot-product attention (Eq. 14-15)."""
        # Compute attention scores: Q * K^T / sqrt(d_k)
        d_k = tf.cast(self.d_k, tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)

        # Softmax normalization (Eq. 15)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = tf.matmul(attention_weights, V)
        return output, attention_weights

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        # Project to Q, K, V (Eq. 11-13)
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        V = self.W_V(inputs)

        # Split into multiple heads (Eq. 16)
        Q = self._split_heads(Q, batch_size)
        K = self._split_heads(K, batch_size)
        V = self._split_heads(V, batch_size)

        # Scaled dot-product attention per head
        attn_output, attn_weights = self._scaled_dot_product_attention(Q, K, V)

        # Concatenate heads (Eq. 17)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        concat_attn = tf.reshape(attn_output, (batch_size, -1, self.d_model))

        # Output projection
        output = self.W_O(concat_attn)

        # Residual connection + Layer normalization (Eq. 18)
        output = self.layer_norm(inputs + output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config
