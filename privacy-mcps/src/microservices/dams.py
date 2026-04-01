"""Data Aggregation Microservice (DAMS) — Layer 4.
Wraps SecureAggregator for microservice deployment."""

from src.privacy.secure_aggregation import SecureAggregator
from src.privacy.paillier_enc import PaillierPHE

class DAMS:
    def __init__(self, phe_system: PaillierPHE):
        self.aggregator = SecureAggregator(phe_system)
    def aggregate(self, all_encrypted, shapes):
        return self.aggregator.full_aggregation_pipeline(all_encrypted, shapes)
