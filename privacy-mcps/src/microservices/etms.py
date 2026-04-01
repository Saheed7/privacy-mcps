"""Encrypted Training Microservice (ETMS) — Layer 3.
Performs local training + Paillier encryption on a data partition."""

import numpy as np
import tensorflow as tf
import logging
from src.models.cnn_bigru_attention import build_model
from src.privacy.paillier_enc import PaillierPHE

logger = logging.getLogger(__name__)

class ETMS:
    def __init__(self, partition_id, model_cfg, train_cfg, phe_system: PaillierPHE):
        self.partition_id = partition_id
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.phe = phe_system

    def train_and_encrypt(self, X, y, n_features, n_classes, task):
        """Algorithm 1: Train local model and encrypt parameters."""
        logger.info(f"[ETMS-{self.partition_id}] Training on {len(X)} samples")
        model = build_model(
            n_features=n_features, n_classes=n_classes, task=task,
            **{k: self.model_cfg[k] for k in [
                "conv1_filters", "conv2_filters", "kernel_size", "pool_size",
                "gru_units", "n_heads", "d_model", "dense_units", "dropout_rate"]},
            learning_rate=self.train_cfg["learning_rate"],
        )
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.train_cfg["early_stopping_patience"],
            restore_best_weights=True)]
        model.fit(X, y, validation_split=self.train_cfg["validation_split"],
                  epochs=self.train_cfg["max_epochs"], batch_size=self.train_cfg["batch_size"],
                  callbacks=callbacks, verbose=0)

        # Extract and encrypt parameters
        params = [w.numpy().flatten() for layer in model.layers for w in layer.trainable_weights]
        shapes = [w.numpy().shape for layer in model.layers for w in layer.trainable_weights]
        encrypted = self.phe.encrypt_model_params(params)
        logger.info(f"[ETMS-{self.partition_id}] Encrypted {sum(p.size for p in params)} params")
        return encrypted, shapes
