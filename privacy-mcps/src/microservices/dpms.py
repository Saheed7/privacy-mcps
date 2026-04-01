"""Data Preprocessing Microservice (DPMS) — Layer 2.
Wraps the DataPipeline for microservice deployment."""

from src.preprocessing.data_pipeline import DataPipeline

class DPMS:
    def __init__(self, config):
        self.pipeline = DataPipeline(config)
    def process(self, dataset_name, task):
        return self.pipeline.full_pipeline(dataset_name, task)
