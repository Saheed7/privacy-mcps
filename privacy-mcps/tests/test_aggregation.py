"""Tests for secure aggregation of encrypted model parameters."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.privacy.paillier_enc import PaillierPHE
from src.privacy.secure_aggregation import SecureAggregator


class TestSecureAggregation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.phe = PaillierPHE(key_length=512)
        self.phe.generate_keys()
        self.aggregator = SecureAggregator(self.phe)

    def test_aggregate_two_servers(self):
        """Test aggregation from 2 edge servers produces correct average."""
        params_1 = [np.array([1.0, 2.0, 3.0])]
        params_2 = [np.array([5.0, 6.0, 7.0])]
        shapes = [(3,)]

        enc_1 = self.phe.encrypt_model_params(params_1)
        enc_2 = self.phe.encrypt_model_params(params_2)

        result = self.aggregator.full_aggregation_pipeline([enc_1, enc_2], shapes)

        expected = [(np.array([1.0, 2.0, 3.0]) + np.array([5.0, 6.0, 7.0])) / 2]
        np.testing.assert_allclose(result[0], expected[0], atol=1e-5)

    def test_aggregate_preserves_accuracy(self):
        """Theorem 3: Zero accuracy degradation — encrypted avg == plaintext avg."""
        M = 4
        np.random.seed(42)
        all_params = [np.random.randn(10).tolist() for _ in range(M)]

        # Plaintext average
        plaintext_avg = np.mean(all_params, axis=0)

        # Encrypted pipeline
        all_enc = []
        for p in all_params:
            enc = self.phe.encrypt_model_params([np.array(p)])
            all_enc.append(enc)

        shapes = [(10,)]
        encrypted_avg = self.aggregator.full_aggregation_pipeline(all_enc, shapes)

        np.testing.assert_allclose(encrypted_avg[0], plaintext_avg, atol=1e-5)

    def test_aggregate_multiple_layers(self):
        """Test aggregation with multiple parameter arrays (simulating model layers)."""
        M = 3
        np.random.seed(123)
        layer_shapes = [(4,), (6,)]

        all_enc = []
        all_plain = []
        for _ in range(M):
            p = [np.random.randn(s[0]) for s in layer_shapes]
            all_plain.append(p)
            all_enc.append(self.phe.encrypt_model_params(p))

        result = self.aggregator.full_aggregation_pipeline(all_enc, layer_shapes)

        for layer_idx in range(len(layer_shapes)):
            expected = np.mean([all_plain[es][layer_idx] for es in range(M)], axis=0)
            np.testing.assert_allclose(result[layer_idx], expected, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
