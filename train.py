"""Train the DeepGuard CNN.

Usage:
    python train.py                 # defaults from config.py
    python train.py --epochs 30 --batch-size 64
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
from model import build_model
from preprocess import OpenCVDataGenerator, list_images


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_history(history, out_path):
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h["accuracy"], label="train")
    axes[0].plot(h["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(h["loss"], label="train")
    axes[1].plot(h["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train DeepGuard AI CNN")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    set_seed(config.SEED)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ---- Data ----
    paths, labels = list_images(config.TRAIN_DIR)
    tr_p, va_p, tr_y, va_y = train_test_split(
        paths, labels, test_size=config.VAL_SPLIT,
        stratify=labels, random_state=config.SEED,
    )
    print(f"Train: {len(tr_p)} images | Val: {len(va_p)} images")
    for i, c in enumerate(config.CLASS_NAMES):
        print(f"  {c}: train={int((tr_y == i).sum())}, val={int((va_y == i).sum())}")

    train_gen = OpenCVDataGenerator(tr_p, tr_y, args.batch_size,
                                    shuffle=True, use_augmentation=True)
    val_gen = OpenCVDataGenerator(va_p, va_y, args.batch_size, shuffle=False)

    # ---- Model ----
    model = build_model(learning_rate=args.lr)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(config.OUTPUT_DIR, "training_log.csv")),
    ]

    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=args.epochs, callbacks=callbacks)

    # ModelCheckpoint already saved the best model; keep the restored best weights too.
    model.save(config.MODEL_PATH)
    plot_history(history, os.path.join(config.OUTPUT_DIR, "training_curves.png"))
    with open(os.path.join(config.OUTPUT_DIR, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()},
                  f, indent=2)

    print(f"\nModel saved to {config.MODEL_PATH}")
    print("Run `python evaluate.py` to compute test accuracy and F1-score.")


if __name__ == "__main__":
    main()
