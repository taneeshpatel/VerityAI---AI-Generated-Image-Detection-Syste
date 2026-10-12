"""Predict whether one or more images are real or AI-generated.

Usage:
    python predict.py path/to/image.jpg [more images...]
"""
import argparse
import os

import tensorflow as tf

import config
from preprocess import load_image, prepare_for_model


def predict_image(model, path):
    p_real = float(model.predict(prepare_for_model(load_image(path)), verbose=0)[0][0])
    if p_real >= config.THRESHOLD:
        return config.DISPLAY_NAMES["REAL"], p_real
    return config.DISPLAY_NAMES["FAKE"], 1.0 - p_real


def main():
    parser = argparse.ArgumentParser(description="DeepGuard AI single-image prediction")
    parser.add_argument("images", nargs="+", help="Image file(s) to classify")
    args = parser.parse_args()

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"No model at {config.MODEL_PATH}. Run train.py first.")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    for path in args.images:
        label, conf = predict_image(model, path)
        print(f"{path}: {label} (confidence {conf * 100:.2f}%)")


if __name__ == "__main__":
    main()
