#!/usr/bin/env python3
"""
Distributed Training with Paillier PHE.

Simulates the six-layer microservice architecture:
  ARMS → DPMS → ETMS (×M) → DAMS → BMS/SELMS → MLaMS

Usage: python scripts/train_distributed.py --dataset cic-iomt --partitions 6 --key-length 2048
"""

import argparse
import os
import sys
import yaml
import numpy as np
import tensorflow as tf
import time
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_pipeline import DataPipeline
from src.models.cnn_bigru_attention import build_model
from src.privacy.paillier_enc import PaillierPHE
from src.privacy.secure_aggregation import SecureAggregator
from src.evaluation.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_MAP = {"cic-iomt": "cic_iomt", "edge-iiot": "edge_iiot"}


def train_local_model(X_part, y_part, n_features, n_classes, task, model_cfg, train_cfg):
    """ETMS: Train a local model on one data partition (Layer 3)."""
    model = build_model(
        n_features=n_features, n_classes=n_classes, task=task,
        conv1_filters=model_cfg["conv1_filters"],
        conv2_filters=model_cfg["conv2_filters"],
        kernel_size=model_cfg["kernel_size"],
        pool_size=model_cfg["pool_size"],
        gru_units=model_cfg["gru_units"],
        n_heads=model_cfg["n_heads"],
        d_model=model_cfg["d_model"],
        dense_units=model_cfg["dense_units"],
        dropout_rate=model_cfg["dropout_rate"],
        learning_rate=train_cfg["learning_rate"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
    ]

    model.fit(
        X_part, y_part,
        validation_split=train_cfg["validation_split"],
        epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )

    return model


def extract_weights(model):
    """Extract all trainable weights as flat numpy arrays."""
    params = []
    shapes = []
    for layer in model.layers:
        for w in layer.trainable_weights:
            arr = w.numpy()
            shapes.append(arr.shape)
            params.append(arr.flatten())
    return params, shapes


def set_weights_from_flat(model, flat_params, shapes):
    """Set model weights from flat arrays with given shapes."""
    idx = 0
    for layer in model.layers:
        new_weights = []
        for w in layer.trainable_weights:
            shape = shapes[idx]
            size = int(np.prod(shape))
            new_weights.append(flat_params[idx].reshape(shape))
            idx += 1
        if new_weights:
            layer.set_weights(new_weights)


def main():
    parser = argparse.ArgumentParser(description="Distributed Training with Paillier PHE")
    parser.add_argument("--dataset", choices=["cic-iomt", "edge-iiot"], required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--partitions", type=int, default=6, help="Number of private edge servers (M)")
    parser.add_argument("--key-length", type=int, default=2048, help="Paillier key length in bits")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip-encryption", action="store_true",
                        help="Skip Paillier encryption (for speed testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)["model"]

    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)
    os.makedirs(f"{args.output_dir}/tables", exist_ok=True)

    ds_key = DATASET_MAP[args.dataset]
    M = args.partitions
    train_cfg = config["training"]

    # ========== LAYER 1-2: ARMS + DPMS ==========
    logger.info(f"[ARMS/DPMS] Loading and preprocessing {args.dataset} ({args.task})")
    pipeline = DataPipeline(config)
    data = pipeline.full_pipeline(ds_key, args.task)

    # Partition data for distributed training
    logger.info(f"[DPMS] Partitioning into {M} segments")
    partitions = pipeline.partition_data(data["X_train"], data["y_train"], M)

    # ========== LAYER 3: PRIVACY PRESERVATION (ETMS × M) ==========
    logger.info(f"\n{'='*60}")
    logger.info(f"  LAYER 3: Training {M} local models (ETMS)")
    logger.info(f"{'='*60}")

    local_models = []
    all_params = []
    all_shapes = None
    total_train_time = 0

    for i, (X_part, y_part) in enumerate(partitions):
        logger.info(f"\n[ETMS-{i+1}] Training on partition {i+1}/{M} ({len(X_part)} samples)")
        t_start = time.time()

        model_i = train_local_model(
            X_part, y_part, data["n_features"], data["n_classes"],
            args.task, model_cfg, train_cfg
        )

        t_train = time.time() - t_start
        total_train_time += t_train
        logger.info(f"[ETMS-{i+1}] Training completed in {t_train:.1f}s")

        # Extract weights
        params_i, shapes_i = extract_weights(model_i)
        if all_shapes is None:
            all_shapes = shapes_i
        all_params.append(params_i)
        local_models.append(model_i)

    total_params = sum(p.size for p in all_params[0])
    logger.info(f"\nTotal trainable parameters per model: {total_params:,}")
    logger.info(f"Total local training time: {total_train_time:.1f}s")

    # ========== PAILLIER ENCRYPTION ==========
    if not args.skip_encryption:
        logger.info(f"\n{'='*60}")
        logger.info(f"  PAILLIER PHE ENCRYPTION ({args.key_length}-bit)")
        logger.info(f"{'='*60}")

        phe_system = PaillierPHE(key_length=args.key_length)
        phe_system.generate_keys()

        # Encrypt all local parameters
        all_encrypted = []
        total_enc_time = 0
        for i, params_i in enumerate(all_params):
            logger.info(f"[ETMS-{i+1}] Encrypting {sum(p.size for p in params_i)} parameters...")
            t_enc = time.time()
            encrypted_i = phe_system.encrypt_model_params(params_i)
            t_enc = time.time() - t_enc
            total_enc_time += t_enc
            all_encrypted.append(encrypted_i)
            logger.info(f"[ETMS-{i+1}] Encryption completed in {t_enc:.1f}s")

        logger.info(f"Total encryption time: {total_enc_time:.1f}s")

        # ========== LAYER 4: KNOWLEDGE AGGREGATION (DAMS) ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"  LAYER 4: Secure Aggregation (DAMS)")
        logger.info(f"{'='*60}")

        aggregator = SecureAggregator(phe_system)
        t_agg = time.time()
        global_params = aggregator.full_aggregation_pipeline(all_encrypted, all_shapes)
        t_agg = time.time() - t_agg
        logger.info(f"Aggregation + Decryption completed in {t_agg:.1f}s")

    else:
        # Plaintext averaging (skip encryption for speed)
        logger.info("\n[SKIP] Paillier encryption skipped (plaintext averaging)")
        global_params = []
        for layer_idx in range(len(all_params[0])):
            stacked = np.stack([all_params[es][layer_idx] for es in range(M)])
            avg = np.mean(stacked, axis=0)
            global_params.append(avg)
        total_enc_time = 0
        t_agg = 0

    # ========== BUILD GLOBAL MODEL (BMS) ==========
    logger.info(f"\n{'='*60}")
    logger.info(f"  BMS: Building Global Model")
    logger.info(f"{'='*60}")

    global_model = build_model(
        n_features=data["n_features"], n_classes=data["n_classes"], task=args.task,
        conv1_filters=model_cfg["conv1_filters"],
        conv2_filters=model_cfg["conv2_filters"],
        kernel_size=model_cfg["kernel_size"],
        pool_size=model_cfg["pool_size"],
        gru_units=model_cfg["gru_units"],
        n_heads=model_cfg["n_heads"],
        d_model=model_cfg["d_model"],
        dense_units=model_cfg["dense_units"],
        dropout_rate=model_cfg["dropout_rate"],
        learning_rate=train_cfg["learning_rate"],
    )

    set_weights_from_flat(global_model, global_params, all_shapes)

    model_path = f"{args.output_dir}/models/distributed_{args.dataset}_{args.task}_M{M}.h5"
    global_model.save(model_path)
    logger.info(f"Global model saved to {model_path}")

    # ========== EVALUATION ==========
    logger.info(f"\n{'='*60}")
    logger.info(f"  EVALUATION")
    logger.info(f"{'='*60}")

    y_prob = global_model.predict(data["X_test"], batch_size=256)

    if args.task == "binary":
        y_pred = (y_prob.flatten() >= 0.5).astype(int)
        metrics = MetricsCalculator.compute_all_metrics(data["y_test"], y_pred, y_prob.flatten(), "binary")
    else:
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(data["y_test"], axis=1) if data["y_test"].ndim > 1 else data["y_test"]
        metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred, y_prob, "multiclass")

    MetricsCalculator.print_metrics(metrics, args.task)

    # ========== SAVE RESULTS ==========
    result_summary = {
        "dataset": args.dataset,
        "task": args.task,
        "partitions": M,
        "key_length": args.key_length,
        "skip_encryption": args.skip_encryption,
        "accuracy": float(metrics["accuracy"]),
        "total_train_time": total_train_time,
        "total_enc_time": total_enc_time,
        "aggregation_time": t_agg,
        "total_params": total_params,
    }

    result_path = f"{args.output_dir}/tables/distributed_{args.dataset}_{args.task}_M{M}.json"
    with open(result_path, "w") as f:
        json.dump(result_summary, f, indent=2)
    logger.info(f"Results saved to {result_path}")
    logger.info("\nDistributed training completed successfully.")


if __name__ == "__main__":
    main()
