"""Evaluate the trained model on data/test with Scikit-learn metrics.

Usage:
    python evaluate.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config
from preprocess import OpenCVDataGenerator, list_images


def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"No model at {config.MODEL_PATH}. Run train.py first.")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model = tf.keras.models.load_model(config.MODEL_PATH)
    paths, y_true = list_images(config.TEST_DIR)
    test_gen = OpenCVDataGenerator(paths, y_true, shuffle=False)

    probs = model.predict(test_gen, verbose=1).ravel()
    y_pred = (probs >= config.THRESHOLD).astype(int)
    y_true = y_true.astype(int)

    labels = [config.DISPLAY_NAMES[c] for c in config.CLASS_NAMES]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        # F1 is reported as macro-average over both classes
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, probs),
        "num_test_images": int(len(y_true)),
    }

    print("\n========== DeepGuard AI — Test Results ==========")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
    print("\n" + classification_report(y_true, y_pred, target_names=labels, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels, os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    with open(os.path.join(config.OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"Saved metrics and confusion matrix to {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
