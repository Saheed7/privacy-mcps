"""
Visualization Module.

Generates publication-quality figures: confusion matrices (Fig. 3),
ROC curves (Fig. 4), and training convergence plots (Fig. 5).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import Dict, List
import os, logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm, class_labels, title, save_path, figsize=(8, 7)):
    """Plot a single confusion matrix heatmap."""
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, linecolor="white", vmin=0, vmax=100,
                cbar_kws={"label": "Percentage (%)"})
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(results_dict: Dict, title: str, save_path: str):
    """
    Plot ROC curves for multiple methods on a single chart.

    Args:
        results_dict: {method_name: {"fpr": array, "tpr": array, "auc": float}}
        title: Plot title.
        save_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#1E8449", "#2E86C1", "#CB4335", "#8E44AD", "#D4AC0D"]
    styles = ["-", "--", "-.", ":", "-"]

    for idx, (name, data) in enumerate(results_dict.items()):
        ax.plot(data["fpr"], data["tpr"], color=colors[idx % len(colors)],
                linewidth=2, linestyle=styles[idx % len(styles)],
                label=f'{name} (AUC = {data["auc"]:.4f})')

    ax.plot([0, 1], [0, 1], ":", color="#ABB2B9", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")


def plot_training_history(history, title_prefix: str, save_path: str):
    """
    Plot training/validation accuracy and loss curves.

    Args:
        history: Keras History object or dict with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        title_prefix: Dataset name for title.
        save_path: Output file path.
    """
    h = history if isinstance(history, dict) else history.history
    epochs = range(1, len(h["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, [a * 100 for a in h["accuracy"]], color="#1E8449", lw=2, label="Training")
    ax1.plot(epochs, [a * 100 for a in h["val_accuracy"]], color="#CB4335", lw=2, ls="--", label="Validation")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title(f"{title_prefix} — Accuracy", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, ls="--")

    # Loss
    ax2.plot(epochs, h["loss"], color="#1E8449", lw=2, label="Training")
    ax2.plot(epochs, h["val_loss"], color="#CB4335", lw=2, ls="--", label="Validation")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax2.set_title(f"{title_prefix} — Loss", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, ls="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Training history saved to {save_path}")
