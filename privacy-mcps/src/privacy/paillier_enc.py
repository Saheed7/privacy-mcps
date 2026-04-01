"""
Paillier Partially Homomorphic Encryption.

Implements key generation, encryption, decryption, and homomorphic operations
as described in Section III-B (Equations 23-33).
"""

import phe
from phe import paillier
import numpy as np
from typing import Tuple, List, Dict
import time
import logging

logger = logging.getLogger(__name__)


class PaillierPHE:
    """Paillier Partially Homomorphic Encryption system."""

    def __init__(self, key_length: int = 2048):
        """
        Initialize Paillier cryptosystem.

        Args:
            key_length: Bit length of the key (default 2048 for 112-bit security).
        """
        self.key_length = key_length
        self.public_key = None
        self.private_key = None

    def generate_keys(self) -> Tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
        """
        Generate public-private key pair (Section III-B, Step 1-4).

        Returns:
            Tuple of (public_key, private_key).
        """
        logger.info(f"Generating Paillier key pair ({self.key_length} bits)...")
        start = time.time()
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=self.key_length
        )
        elapsed = time.time() - start
        logger.info(f"Key generation completed in {elapsed:.2f}s")
        return self.public_key, self.private_key

    def encrypt_value(self, value: float) -> paillier.EncryptedNumber:
        """
        Encrypt a single value using Paillier encryption (Eq. 24).

        c = E(m, r) = g^m * r^n mod n^2

        Args:
            value: Plaintext value to encrypt.

        Returns:
            EncryptedNumber (ciphertext).
        """
        assert self.public_key is not None, "Keys not generated. Call generate_keys() first."
        return self.public_key.encrypt(value)

    def decrypt_value(self, ciphertext: paillier.EncryptedNumber) -> float:
        """
        Decrypt a ciphertext using private key (Eq. 26).

        m = D(c) = L(c^lambda mod n^2) * mu mod n

        Args:
            ciphertext: Encrypted value.

        Returns:
            Decrypted plaintext value.
        """
        assert self.private_key is not None, "Keys not generated. Call generate_keys() first."
        return self.private_key.decrypt(ciphertext)

    def encrypt_array(self, array: np.ndarray) -> List[paillier.EncryptedNumber]:
        """
        Encrypt a numpy array element-wise (Eq. 39).

        E(W_L^(i)) = {E(Encode(w)) : w in W_L^(i)}

        Args:
            array: 1D numpy array of plaintext values.

        Returns:
            List of EncryptedNumbers.
        """
        flat = array.flatten().tolist()
        encrypted = [self.public_key.encrypt(v) for v in flat]
        return encrypted

    def decrypt_array(self, encrypted_list: List[paillier.EncryptedNumber],
                      shape: tuple = None) -> np.ndarray:
        """
        Decrypt a list of encrypted values back to numpy array.

        Args:
            encrypted_list: List of EncryptedNumbers.
            shape: Optional shape to reshape the result.

        Returns:
            Numpy array of decrypted values.
        """
        decrypted = np.array([self.private_key.decrypt(c) for c in encrypted_list])
        if shape is not None:
            decrypted = decrypted.reshape(shape)
        return decrypted

    def encrypt_model_params(self, params: List[np.ndarray]) -> List[List[paillier.EncryptedNumber]]:
        """
        Encrypt all model parameters (weight matrices and bias vectors).

        Args:
            params: List of numpy arrays (one per layer).

        Returns:
            List of lists of EncryptedNumbers.
        """
        logger.info(f"Encrypting {sum(p.size for p in params)} model parameters...")
        start = time.time()
        encrypted_params = []
        for i, param in enumerate(params):
            encrypted = self.encrypt_array(param)
            encrypted_params.append(encrypted)
            logger.debug(f"  Layer {i}: {param.size} params encrypted")
        elapsed = time.time() - start
        logger.info(f"Encryption completed in {elapsed:.2f}s")
        return encrypted_params

    def decrypt_model_params(self, encrypted_params: List[List[paillier.EncryptedNumber]],
                             shapes: List[tuple]) -> List[np.ndarray]:
        """
        Decrypt all encrypted model parameters.

        Args:
            encrypted_params: List of lists of EncryptedNumbers.
            shapes: Original shapes for each parameter array.

        Returns:
            List of numpy arrays (decrypted parameters).
        """
        logger.info("Decrypting aggregated model parameters...")
        start = time.time()
        decrypted_params = []
        for enc_layer, shape in zip(encrypted_params, shapes):
            decrypted = self.decrypt_array(enc_layer, shape)
            decrypted_params.append(decrypted)
        elapsed = time.time() - start
        logger.info(f"Decryption completed in {elapsed:.2f}s")
        return decrypted_params

    @staticmethod
    def homomorphic_add(ct_a: paillier.EncryptedNumber,
                        ct_b: paillier.EncryptedNumber) -> paillier.EncryptedNumber:
        """
        Homomorphic addition via ciphertext multiplication (Property 1, Eq. 28).

        D(E(m1) * E(m2) mod n^2) = (m1 + m2) mod n

        Args:
            ct_a: First ciphertext.
            ct_b: Second ciphertext.

        Returns:
            Encrypted sum.
        """
        return ct_a + ct_b  # phe library overloads + for homomorphic add

    @staticmethod
    def homomorphic_scalar_mult(ct: paillier.EncryptedNumber,
                                scalar: float) -> paillier.EncryptedNumber:
        """
        Homomorphic scalar multiplication (Property 2, Eq. 30).

        D(E(m)^k mod n^2) = k * m mod n

        Args:
            ct: Ciphertext.
            scalar: Plaintext scalar.

        Returns:
            Encrypted product.
        """
        return ct * scalar  # phe library overloads * for scalar mult
