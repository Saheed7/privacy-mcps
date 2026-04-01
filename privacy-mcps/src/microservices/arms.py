"""Activity Receiver Microservice (ARMS) — Layer 1.
Handles data ingestion, validation, protocol translation, and routing."""

import logging
logger = logging.getLogger(__name__)

class ARMS:
    """Activity Receiver Microservice."""
    def __init__(self, config: dict):
        self.config = config
        self.sources = []

    def register_source(self, source_id: str, protocol: str):
        self.sources.append({"id": source_id, "protocol": protocol})
        logger.info(f"Registered source: {source_id} ({protocol})")

    def validate(self, data_record: dict) -> bool:
        """Validate incoming data record schema and integrity."""
        required = ["timestamp", "features", "source_id"]
        return all(k in data_record for k in required)

    def receive_and_route(self, data_stream: list) -> list:
        """Filter invalid records and route to DPMS (Eq. 38)."""
        validated = [d for d in data_stream if self.validate(d)]
        logger.info(f"ARMS: {len(validated)}/{len(data_stream)} records validated")
        return validated
