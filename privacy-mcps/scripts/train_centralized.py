#!/usr/bin/env python3
"""
Centralized Training Script (Baseline).

Trains the 1D-CNN-BiGRU-Attention model without privacy or distribution.
Usage: python scripts/train_centralized.py --dataset cic-iomt --task binary
"""

import argparse
import os
import sys
import yaml
import numpy as np
import tensorflow as tf
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_pipeline import DataPipeline
from src.models.cnn_bigru_attention import build_model, get_model_summary
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualization import plot_training_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_MAP = {"cic-iomt": "cic_iomt", "edge-iiot": "edge_iiot"}


def main():
    parser = argparse.ArgumentParser(description="Centralized Training (Baseline)")
    parser.add_argument("--dataset", choices=["cic-iomt", "edge-iiot"], required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    # Load configs
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)["model"]

    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)

    ds_key = DATASET_MAP[args.dataset]

    # ---- Data Pipeline ----
    logger.info(f"Dataset: {args.dataset}, Task: {args.task}")
    pipeline = DataPipeline(config)
    data = pipeline.full_pipeline(ds_key, args.task)

    logger.info(f"Features: {data['n_features']}, Classes: {data['n_classes']}")
    logger.info(f"Train samples: {data['X_train'].shape[0]}, Test samples: {data['X_test'].shape[0]}")

    # ---- Build Model ----
    model = build_model(
        n_features=data["n_features"],
        n_classes=data["n_classes"],
        task=args.task,
        conv1_filters=model_cfg["conv1_filters"],
        conv2_filters=model_cfg["conv2_filters"],
        kernel_size=model_cfg["kernel_size"],
        pool_size=model_cfg["pool_size"],
        gru_units=model_cfg["gru_units"],
        n_heads=model_cfg["n_heads"],
        d_model=model_cfg["d_model"],
        dense_units=model_cfg["dense_units"],
        dropout_rate=model_cfg["dropout_rate"],
        learning_rate=config["training"]["learning_rate"],
    )
    logger.info("\n" + get_model_summary(model))

    # ---- Training ----
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    logger.info("Starting centralized training...")
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_split=config["training"]["validation_split"],
        epochs=config["training"]["max_epochs"],
        batch_size=config["training"]["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Save Model ----
    model_path = f"{args.output_dir}/models/centralized_{args.dataset}_{args.task}.h5"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # ---- Evaluation ----
    y_prob = model.predict(data["X_test"], batch_size=256)

    if args.task == "binary":
        y_pred = (y_prob.flatten() >= 0.5).astype(int)
        y_prob_flat = y_prob.flatten()
        metrics = MetricsCalculator.compute_all_metrics(data["y_test"], y_pred, y_prob_flat, "binary")
    else:
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(data["y_test"], axis=1) if data["y_test"].ndim > 1 else data["y_test"]
        metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred, y_prob, "multiclass")

    MetricsCalculator.print_metrics(metrics, args.task)

    # ---- Plots ----
    plot_training_history(
        history,
        f"{data['dataset_name']} ({args.task})",
        f"{args.output_dir}/figures/centralized_{args.dataset}_{args.task}_history.png"
    )

    logger.info("Centralized training completed successfully.")


if __name__ == "__main__":
    main()
