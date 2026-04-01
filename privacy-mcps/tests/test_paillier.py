"""Tests for Paillier PHE encryption, decryption, and homomorphic operations."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.privacy.paillier_enc import PaillierPHE
from src.privacy.encoding import FixedPointEncoder


class TestPaillierPHE:
    """Test suite for Paillier encryption system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.phe = PaillierPHE(key_length=512)  # Small key for fast testing
        self.phe.generate_keys()

    def test_key_generation(self):
        assert self.phe.public_key is not None
        assert self.phe.private_key is not None

    def test_encrypt_decrypt_single(self):
        original = 3.14159
        ct = self.phe.encrypt_value(original)
        decrypted = self.phe.decrypt_value(ct)
        assert abs(decrypted - original) < 1e-6

    def test_encrypt_decrypt_negative(self):
        original = -0.5678
        ct = self.phe.encrypt_value(original)
        decrypted = self.phe.decrypt_value(ct)
        assert abs(decrypted - original) < 1e-6

    def test_additive_homomorphism(self):
        """Property 1 (Eq. 28): D(E(m1) * E(m2)) = m1 + m2."""
        a, b = 42.5, 17.3
        ct_a = self.phe.encrypt_value(a)
        ct_b = self.phe.encrypt_value(b)
        ct_sum = ct_a + ct_b  # Homomorphic addition
        result = self.phe.decrypt_value(ct_sum)
        assert abs(result - (a + b)) < 1e-6

    def test_scalar_multiplication(self):
        """Property 2 (Eq. 30): D(E(m)^k) = k * m."""
        m, k = 7.5, 3.0
        ct = self.phe.encrypt_value(m)
        ct_scaled = ct * k  # Homomorphic scalar mult
        result = self.phe.decrypt_value(ct_scaled)
        assert abs(result - (m * k)) < 1e-5

    def test_encrypt_decrypt_array(self):
        arr = np.array([1.1, -2.2, 3.3, -4.4, 5.5])
        encrypted = self.phe.encrypt_array(arr)
        assert len(encrypted) == 5
        decrypted = self.phe.decrypt_array(encrypted)
        np.testing.assert_allclose(decrypted, arr, atol=1e-6)

    def test_homomorphic_averaging(self):
        """Corollary 1 (Eq. 32): Encrypted averaging of M values."""
        values = [10.0, 20.0, 30.0, 40.0]
        encrypted = [self.phe.encrypt_value(v) for v in values]
        # Sum
        ct_sum = encrypted[0]
        for ct in encrypted[1:]:
            ct_sum = ct_sum + ct
        # Average (scalar mult by 1/M)
        ct_avg = ct_sum * (1.0 / len(values))
        result = self.phe.decrypt_value(ct_avg)
        expected = np.mean(values)
        assert abs(result - expected) < 1e-5


class TestFixedPointEncoder:
    def test_encode_decode(self):
        enc = FixedPointEncoder(precision=8)
        original = 0.123456789
        encoded = enc.encode(original)
        decoded = enc.decode(encoded)
        assert abs(decoded - original) < 1e-7

    def test_max_error_bound(self):
        enc = FixedPointEncoder(precision=8)
        err = enc.max_encoding_error(n_servers=6)
        assert err == 6e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
