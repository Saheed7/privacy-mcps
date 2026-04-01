"""Tests for the 1D-CNN-BiGRU-Attention model."""

import numpy as np
import pytest
import tensorflow as tf
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_bigru_attention import build_model
from src.models.attention import MultiHeadSelfAttention


class TestModel:
    """Test suite for the hybrid deep learning model."""

    def test_binary_model_builds(self):
        model = build_model(n_features=46, n_classes=2, task="binary")
        assert model is not None
        assert model.output_shape == (None, 1)

    def test_multiclass_model_builds(self):
        model = build_model(n_features=61, n_classes=15, task="multiclass")
        assert model is not None
        assert model.output_shape == (None, 15)

    def test_forward_pass_binary(self):
        model = build_model(n_features=46, n_classes=2, task="binary")
        X = np.random.randn(8, 46).astype(np.float32)
        y_pred = model.predict(X, verbose=0)
        assert y_pred.shape == (8, 1)
        assert np.all(y_pred >= 0) and np.all(y_pred <= 1)

    def test_forward_pass_multiclass(self):
        model = build_model(n_features=61, n_classes=15, task="multiclass")
        X = np.random.randn(8, 61).astype(np.float32)
        y_pred = model.predict(X, verbose=0)
        assert y_pred.shape == (8, 15)
        assert np.allclose(y_pred.sum(axis=1), 1.0, atol=1e-5)

    def test_model_trainable_params(self):
        model = build_model(n_features=46, n_classes=2, task="binary")
        total = model.count_params()
        assert total > 100000  # Should have ~150K+ params

    def test_d_model_assertion(self):
        with pytest.raises(AssertionError):
            build_model(n_features=46, n_classes=2, task="binary", gru_units=64, d_model=64)


class TestAttention:
    """Test suite for Multi-Head Self-Attention layer."""

    def test_attention_output_shape(self):
        attn = MultiHeadSelfAttention(d_model=128, n_heads=4)
        x = tf.random.normal((2, 10, 128))
        out = attn(x)
        assert out.shape == (2, 10, 128)

    def test_attention_preserves_residual(self):
        attn = MultiHeadSelfAttention(d_model=128, n_heads=4)
        x = tf.zeros((1, 5, 128))
        out = attn(x)
        # With zero input and layernorm, output should be close to zero
        assert tf.reduce_max(tf.abs(out)).numpy() < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
