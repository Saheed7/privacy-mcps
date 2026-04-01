"""
Fixed-Point Encoding for Paillier Cryptosystem.

Converts real-valued neural network parameters to integers (Eq. 34-37).
"""

import numpy as np


class FixedPointEncoder:
    """Encode/decode floats to/from integers for Paillier encryption."""

    def __init__(self, precision: int = 8):
        """
        Args:
            precision: Number of decimal places to preserve (P in Eq. 34).
        """
        self.precision = precision
        self.scale = 10 ** precision

    def encode(self, value: float) -> int:
        """Encode float to integer (Eq. 34): Encode(w) = floor(w * 10^P)."""
        return int(np.floor(value * self.scale))

    def decode(self, encoded: int, n: int = None) -> float:
        """Decode integer back to float (Eq. 35-36)."""
        return encoded / self.scale

    def encode_array(self, arr: np.ndarray) -> np.ndarray:
        """Encode numpy array element-wise."""
        return np.floor(arr * self.scale).astype(np.int64)

    def decode_array(self, arr: np.ndarray) -> np.ndarray:
        """Decode numpy array element-wise."""
        return arr.astype(np.float64) / self.scale

    def max_encoding_error(self, n_servers: int) -> float:
        """Compute maximum encoding error bound (Eq. 58): M * 10^{-P}."""
        return n_servers * (10 ** (-self.precision))
