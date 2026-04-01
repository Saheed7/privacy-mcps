#!/usr/bin/env python3
"""
Ablation Experiments (Table XVII).

Systematically removes components to measure individual contributions.
Usage: python scripts/run_ablation.py --dataset cic-iomt --task binary
"""

import argparse, os, sys, yaml, numpy as np, json, logging
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, Bidirectional, GRU,
    Dropout, Dense, GlobalAveragePooling1D, Reshape, Flatten
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_pipeline import DataPipeline
from src.models.attention import MultiHeadSelfAttention
from src.evaluation.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DATASET_MAP = {"cic-iomt": "cic_iomt", "edge-iiot": "edge_iiot"}


def build_ablation_model(n_features, n_classes, task, use_cnn=True, use_bigru=True, use_attention=True, lr=0.001):
    """Build model with optional component removal."""
    inputs = Input(shape=(n_features,))
    x = Reshape((n_features, 1))(inputs)

    if use_cnn:
        x = Conv1D(128, 3, activation="relu", padding="valid")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(64, 3, activation="relu", padding="valid")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)

    if use_bigru:
        x = Bidirectional(GRU(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
    
    if use_attention:
        d = x.shape[-1]
        if d and d != 128:
            x = Dense(128)(x)
        x = MultiHeadSelfAttention(d_model=128, n_heads=4)(x)

    x = GlobalAveragePooling1D()(x) if len(x.shape) == 3 else Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    if task == "binary":
        outputs = Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = Dense(n_classes, activation="softmax")(x)
        loss = "categorical_crossentropy"

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
    return model


def train_and_evaluate(model, X_train, y_train, X_test, y_test, task, cfg):
    """Train model and return accuracy."""
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"], restore_best_weights=True)]
    model.fit(X_train, y_train, validation_split=cfg["validation_split"],
              epochs=cfg["max_epochs"], batch_size=cfg["batch_size"], callbacks=callbacks, verbose=0)
    y_prob = model.predict(X_test, batch_size=256)
    if task == "binary":
        y_pred = (y_prob.flatten() >= 0.5).astype(int)
        return MetricsCalculator.compute_all_metrics(y_test, y_pred, y_prob.flatten(), "binary")["accuracy"]
    else:
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        return MetricsCalculator.compute_all_metrics(y_true, y_pred, y_prob, "multiclass")["accuracy"]


def main():
    parser = argparse.ArgumentParser(description="Ablation Experiments")
    parser.add_argument("--dataset", choices=["cic-iomt", "edge-iiot"], required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ds_key = DATASET_MAP[args.dataset]
    pipeline = DataPipeline(config)
    data = pipeline.full_pipeline(ds_key, args.task)
    cfg = config["training"]

    # Ablation configurations: (name, use_cnn, use_bigru, use_attention)
    ablations = [
        ("Full Model (CNN+BiGRU+Attn)", True, True, True),
        ("Without Attention", True, True, False),
        ("Without BiGRU (CNN+Attn)", True, False, True),
        ("Without CNN (BiGRU+Attn)", False, True, True),
        ("CNN only", True, False, False),
        ("BiGRU only", False, True, False),
        ("Attention only", False, False, True),
    ]

    results = []
    for name, cnn, bigru, attn in ablations:
        logger.info(f"\n{'='*50}")
        logger.info(f"  Ablation: {name}")
        logger.info(f"    CNN={cnn}, BiGRU={bigru}, Attention={attn}")
        logger.info(f"{'='*50}")

        model = build_ablation_model(data["n_features"], data["n_classes"], args.task, cnn, bigru, attn, cfg["learning_rate"])
        acc = train_and_evaluate(model, data["X_train"], data["y_train"], data["X_test"], data["y_test"], args.task, cfg)

        results.append({"config": name, "accuracy": acc})
        logger.info(f"  Result: {acc:.2f}%")
        tf.keras.backend.clear_session()

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY RESULTS — {args.dataset} ({args.task})")
    print(f"{'='*70}")
    baseline = results[0]["accuracy"]
    print(f"  {'Configuration':<40} {'Accuracy':>10} {'Delta':>10}")
    print(f"  {'-'*60}")
    for r in results:
        delta = r["accuracy"] - baseline
        delta_str = f"{delta:+.2f}%" if r["config"] != results[0]["config"] else "—"
        print(f"  {r['config']:<40} {r['accuracy']:>9.2f}% {delta_str:>10}")
    print(f"{'='*70}\n")

    # Save
    os.makedirs(f"{args.output_dir}/tables", exist_ok=True)
    with open(f"{args.output_dir}/tables/ablation_{args.dataset}_{args.task}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
