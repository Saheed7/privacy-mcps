#!/usr/bin/env python3
"""
Full Evaluation Pipeline.

Generates all figures, tables, and metrics from a trained model.
Usage: python scripts/evaluate.py --dataset cic-iomt --task binary --model-path results/models/best.h5
"""

import argparse, os, sys, yaml, numpy as np, json, logging
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_pipeline import DataPipeline
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_roc_curves, plot_training_history
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DATASET_MAP = {"cic-iomt": "cic_iomt", "edge-iiot": "edge_iiot"}


def main():
    parser = argparse.ArgumentParser(description="Full Evaluation Pipeline")
    parser.add_argument("--dataset", choices=["cic-iomt", "edge-iiot"], required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fig_dir = f"{args.output_dir}/figures"
    tbl_dir = f"{args.output_dir}/tables"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    ds_key = DATASET_MAP[args.dataset]
    pipeline = DataPipeline(config)
    data = pipeline.full_pipeline(ds_key, args.task)

    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy" if args.task == "binary" else "categorical_crossentropy", metrics=["accuracy"])

    # Predict
    y_prob = model.predict(data["X_test"], batch_size=256)

    if args.task == "binary":
        y_pred = (y_prob.flatten() >= 0.5).astype(int)
        y_true = data["y_test"]
        y_prob_for_metrics = y_prob.flatten()
        metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred, y_prob_for_metrics, "binary")
    else:
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(data["y_test"], axis=1) if data["y_test"].ndim > 1 else data["y_test"]
        metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred, y_prob, "multiclass")

    MetricsCalculator.print_metrics(metrics, args.task)

    # Confusion Matrix
    prefix = f"{args.dataset}_{args.task}"
    if args.task == "binary":
        labels = ["Benign", "Malicious"] if "cic" in args.dataset else ["Normal", "Attack"]
    else:
        labels = [str(i) for i in range(data["n_classes"])]

    plot_confusion_matrix(
        metrics["confusion_matrix"], labels,
        f"{data['dataset_name']} — {args.task.title()} Confusion Matrix",
        f"{fig_dir}/cm_{prefix}.png"
    )

    # ROC Curve (binary)
    if args.task == "binary":
        roc_data = {"PHE + Proposed": {"fpr": metrics["fpr"], "tpr": metrics["tpr"], "auc": metrics["auc"]}}
        plot_roc_curves(roc_data, f"{data['dataset_name']} — Binary ROC Curve", f"{fig_dir}/roc_{prefix}.png")

    # Save metrics JSON
    save_metrics = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray) and k not in ["fpr", "tpr", "roc_thresholds", "per_class_roc", "confusion_matrix"]}
    with open(f"{tbl_dir}/metrics_{prefix}.json", "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
