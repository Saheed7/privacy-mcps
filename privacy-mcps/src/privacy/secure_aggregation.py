"""
Secure Aggregation Protocol using Paillier PHE.

Implements the homomorphic aggregation of encrypted model parameters
from multiple private edge servers (Equations 40-43).
"""

import numpy as np
from typing import List, Dict
import logging
from src.privacy.paillier_enc import PaillierPHE

logger = logging.getLogger(__name__)


class SecureAggregator:
    """Implements DAMS functionality for encrypted parameter aggregation."""

    def __init__(self, paillier_system: PaillierPHE):
        self.phe = paillier_system

    def aggregate_encrypted_params(
        self,
        all_encrypted_params: List[List[list]],
    ) -> List[list]:
        """
        Homomorphic aggregation of encrypted parameters from M edge servers (Eq. 40-41).

        E(W_I) = prod_i E(W_L^*(i)) mod n^2

        Args:
            all_encrypted_params: List of M encrypted parameter sets,
                each being a list of lists (one per layer).

        Returns:
            Aggregated encrypted parameters (sum).
        """
        M = len(all_encrypted_params)
        n_layers = len(all_encrypted_params[0])
        logger.info(f"Aggregating encrypted params from {M} edge servers, {n_layers} layers")

        aggregated = []
        for layer_idx in range(n_layers):
            layer_size = len(all_encrypted_params[0][layer_idx])
            agg_layer = list(all_encrypted_params[0][layer_idx])  # Start with first ES

            for es_idx in range(1, M):
                es_layer = all_encrypted_params[es_idx][layer_idx]
                for param_idx in range(layer_size):
                    # Homomorphic addition: ciphertext multiplication (Eq. 28)
                    agg_layer[param_idx] = agg_layer[param_idx] + es_layer[param_idx]

            aggregated.append(agg_layer)
            logger.debug(f"  Layer {layer_idx}: {layer_size} params aggregated")

        return aggregated

    def compute_encrypted_average(
        self,
        aggregated_encrypted: List[list],
        n_servers: int,
    ) -> List[list]:
        """
        Compute encrypted average using scalar multiplication (Eq. 42).

        E(W_bar) = E(W_F)^{M^{-1}} mod n^2

        In the phe library, this is: encrypted_sum * (1/M)

        Args:
            aggregated_encrypted: Aggregated encrypted parameters.
            n_servers: Number of servers M.

        Returns:
            Encrypted averaged parameters.
        """
        scale = 1.0 / n_servers
        logger.info(f"Computing encrypted average (1/{n_servers} scaling)")

        averaged = []
        for layer_enc in aggregated_encrypted:
            avg_layer = [ct * scale for ct in layer_enc]
            averaged.append(avg_layer)

        return averaged

    def full_aggregation_pipeline(
        self,
        all_encrypted_params: List[List[list]],
        shapes: List[tuple],
    ) -> List[np.ndarray]:
        """
        Complete aggregation pipeline: aggregate → average → decrypt (Eq. 40-43).

        Args:
            all_encrypted_params: Encrypted params from M edge servers.
            shapes: Original shapes of each parameter array.

        Returns:
            Decrypted averaged model parameters.
        """
        M = len(all_encrypted_params)

        # Step 1: Homomorphic aggregation (Eq. 40-41)
        aggregated = self.aggregate_encrypted_params(all_encrypted_params)

        # Step 2: Encrypted averaging (Eq. 42)
        averaged = self.compute_encrypted_average(aggregated, M)

        # Step 3: Decryption (Eq. 43)
        decrypted = self.phe.decrypt_model_params(averaged, shapes)

        logger.info("Full aggregation pipeline completed successfully")
        return decrypted
