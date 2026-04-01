"""
Comprehensive Evaluation Metrics.

Computes accuracy, precision, recall, F1-score, AUC, FAR,
confusion matrices, and generates classification reports.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Compute all evaluation metrics for binary and multi-class tasks."""

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: np.ndarray, task: str = "binary") -> Dict:
        """
        Compute comprehensive metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels (argmax for multi-class).
            y_prob: Predicted probabilities.
            task: 'binary' or 'multiclass'.

        Returns:
            Dictionary of all metrics.
        """
        if task == "binary":
            return MetricsCalculator._binary_metrics(y_true, y_pred, y_prob)
        else:
            return MetricsCalculator._multiclass_metrics(y_true, y_pred, y_prob)

    @staticmethod
    def _binary_metrics(y_true, y_pred, y_prob) -> Dict:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, zero_division=0) * 100
        recall = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0

        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_score = 0.0

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc_score,
            "far": far,
            "confusion_matrix": cm,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fpr": fpr, "tpr": tpr, "roc_thresholds": thresholds,
        }

    @staticmethod
    def _multiclass_metrics(y_true, y_pred, y_prob) -> Dict:
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
        cm = confusion_matrix(y_true, y_pred)

        n_classes = y_prob.shape[1] if y_prob.ndim > 1 else len(np.unique(y_true))

        # FAR: macro-averaged false alarm rate
        far_per_class = []
        for c in range(n_classes):
            y_true_bin = (y_true == c).astype(int)
            y_pred_bin = (y_pred == c).astype(int)
            cm_c = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            tn_c, fp_c = cm_c[0, 0], cm_c[0, 1]
            far_c = fp_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0.0
            far_per_class.append(far_c)
        far_macro = np.mean(far_per_class) * 100

        # AUC: one-vs-rest macro
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            auc_macro = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        except Exception:
            auc_macro = 0.0

        # Per-class ROC curves
        per_class_roc = {}
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            for c in range(n_classes):
                fpr_c, tpr_c, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
                per_class_roc[c] = {"fpr": fpr_c, "tpr": tpr_c, "auc": auc(fpr_c, tpr_c)}
        except Exception:
            pass

        return {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_score_macro": f1,
            "auc_macro": auc_macro,
            "far_macro": far_macro,
            "confusion_matrix": cm,
            "per_class_roc": per_class_roc,
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }

    @staticmethod
    def print_metrics(metrics: Dict, task: str = "binary"):
        """Pretty-print metrics."""
        print("\n" + "=" * 60)
        print(f"  EVALUATION RESULTS ({task.upper()})")
        print("=" * 60)
        print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
        if task == "binary":
            print(f"  Precision: {metrics['precision']:.2f}%")
            print(f"  Recall:    {metrics['recall']:.2f}%")
            print(f"  F1-Score:  {metrics['f1_score']:.2f}%")
            print(f"  AUC:       {metrics['auc']:.4f}")
            print(f"  FAR:       {metrics['far']:.2f}%")
            print(f"\n  Confusion Matrix:")
            print(f"    TN={metrics['tn']}, FP={metrics['fp']}")
            print(f"    FN={metrics['fn']}, TP={metrics['tp']}")
        else:
            print(f"  Precision (macro): {metrics['precision_macro']:.2f}%")
            print(f"  Recall (macro):    {metrics['recall_macro']:.2f}%")
            print(f"  F1-Score (macro):  {metrics['f1_score_macro']:.2f}%")
            print(f"  AUC (macro):       {metrics['auc_macro']:.4f}")
            print(f"  FAR (macro):       {metrics['far_macro']:.2f}%")
        print("=" * 60 + "\n")
