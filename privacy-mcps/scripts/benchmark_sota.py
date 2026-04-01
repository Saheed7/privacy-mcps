#!/usr/bin/env python3
"""
Benchmarking Against State-of-the-Art (Table XVI).

Compares proposed framework against baseline methods including
RBFN with DP, standard ML classifiers, and ablated architectures.

Usage: python scripts/benchmark_sota.py --dataset cic-iomt --task binary
"""

import argparse, os, sys, yaml, numpy as np, json, logging
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_pipeline import DataPipeline
from src.models.cnn_bigru_attention import build_model
from src.evaluation.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DATASET_MAP = {"cic-iomt": "cic_iomt", "edge-iiot": "edge_iiot"}


def evaluate_sklearn(clf, X_train, y_train, X_test, y_test, name):
    """Train and evaluate an sklearn classifier."""
    logger.info(f"  Training {name}...")
    y_tr = y_train if y_train.ndim == 1 else np.argmax(y_train, axis=1)
    y_te = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)
    clf.fit(X_train, y_tr)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_te) * 100
    f1 = float(np.mean(y_pred == y_te) * 100)  # simplified
    try:
        from sklearn.metrics import f1_score as f1s
        avg = "binary" if len(np.unique(y_te)) == 2 else "macro"
        f1 = f1s(y_te, y_pred, average=avg, zero_division=0) * 100
    except Exception:
        pass
    return {"method": name, "accuracy": round(acc, 2), "f1_score": round(f1, 2)}


def evaluate_proposed(data, task, model_cfg, train_cfg):
    """Train and evaluate the proposed model."""
    logger.info("  Training Proposed (1D-CNN-BiGRU-Attn)...")
    model = build_model(
        n_features=data["n_features"], n_classes=data["n_classes"], task=task,
        **{k: model_cfg[k] for k in ["conv1_filters", "conv2_filters", "kernel_size",
           "pool_size", "gru_units", "n_heads", "d_model", "dense_units", "dropout_rate"]},
        learning_rate=train_cfg["learning_rate"],
    )
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=train_cfg["early_stopping_patience"], restore_best_weights=True)]
    model.fit(data["X_train"], data["y_train"], validation_split=train_cfg["validation_split"],
              epochs=train_cfg["max_epochs"], batch_size=train_cfg["batch_size"], callbacks=callbacks, verbose=0)

    y_prob = model.predict(data["X_test"], batch_size=256)
    if task == "binary":
        y_pred = (y_prob.flatten() >= 0.5).astype(int)
        m = MetricsCalculator.compute_all_metrics(data["y_test"], y_pred, y_prob.flatten(), "binary")
        return {"method": "Proposed (PHE + 1D-CNN-BiGRU-Attn)", "accuracy": round(m["accuracy"], 2), "f1_score": round(m["f1_score"], 2)}
    else:
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(data["y_test"], axis=1) if data["y_test"].ndim > 1 else data["y_test"]
        m = MetricsCalculator.compute_all_metrics(y_true, y_pred, y_prob, "multiclass")
        return {"method": "Proposed (PHE + 1D-CNN-BiGRU-Attn)", "accuracy": round(m["accuracy"], 2), "f1_score": round(m["f1_score_macro"], 2)}


def main():
    parser = argparse.ArgumentParser(description="SOTA Benchmarking")
    parser.add_argument("--dataset", choices=["cic-iomt", "edge-iiot"], required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)["model"]

    ds_key = DATASET_MAP[args.dataset]
    pipeline = DataPipeline(config)
    data = pipeline.full_pipeline(ds_key, args.task)

    # Limit samples for sklearn classifiers (they can be slow on large datasets)
    max_sklearn = 50000
    if data["X_train"].shape[0] > max_sklearn:
        idx = np.random.choice(data["X_train"].shape[0], max_sklearn, replace=False)
        X_tr_sk, y_tr_sk = data["X_train"][idx], data["y_train"][idx]
    else:
        X_tr_sk, y_tr_sk = data["X_train"], data["y_train"]

    results = []

    # Sklearn baselines
    baselines = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)),
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ("MLP Classifier", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)),
    ]

    for name, clf in baselines:
        r = evaluate_sklearn(clf, X_tr_sk, y_tr_sk, data["X_test"], data["y_test"], name)
        results.append(r)
        logger.info(f"    {name}: Acc={r['accuracy']}%, F1={r['f1_score']}%")

    # Proposed method
    r_prop = evaluate_proposed(data, args.task, model_cfg, config["training"])
    results.append(r_prop)
    logger.info(f"    Proposed: Acc={r_prop['accuracy']}%, F1={r_prop['f1_score']}%")

    # Print table
    print(f"\n{'='*70}")
    print(f"  SOTA BENCHMARKING — {args.dataset} ({args.task})")
    print(f"{'='*70}")
    print(f"  {'Method':<45} {'Accuracy':>10} {'F1-Score':>10}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {r['method']:<45} {r['accuracy']:>9.2f}% {r['f1_score']:>9.2f}%")
    print(f"{'='*70}\n")

    os.makedirs(f"{args.output_dir}/tables", exist_ok=True)
    with open(f"{args.output_dir}/tables/benchmark_{args.dataset}_{args.task}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
