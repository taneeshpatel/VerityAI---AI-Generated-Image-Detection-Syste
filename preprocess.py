"""Image preprocessing with OpenCV + a Keras data generator.

Pipeline: read (OpenCV, BGR->RGB) -> resize to 256x256 -> normalize to [0, 1]
-> float32 tensor of shape (N, 256, 256, 3) for the CNN.
"""
import os

import cv2
import numpy as np
import tensorflow as tf

import config


# ---------------------------------------------------------------------------
# Single-image helpers
# ---------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """Read an image from disk as an RGB uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (e.g. from an upload) into an RGB uint8 array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(img: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Resize to size x size and normalize pixel values to [0, 1]."""
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def to_tensor(img: np.ndarray) -> np.ndarray:
    """Add the batch dimension: (H, W, C) -> (1, H, W, C)."""
    return np.expand_dims(img, axis=0)


def prepare_for_model(img: np.ndarray) -> np.ndarray:
    """Full pipeline for inference on one RGB image."""
    return to_tensor(preprocess_image(img))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def list_images(root: str):
    """Return (paths, labels) for root/FAKE and root/REAL."""
    paths, labels = [], []
    for idx, cls in enumerate(config.CLASS_NAMES):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Expected folder '{cls_dir}'. See README.md for the dataset layout."
            )
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(config.VALID_EXTS):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return np.array(paths), np.array(labels, dtype=np.float32)


def random_augment(img: np.ndarray) -> np.ndarray:
    """Light augmentation. Kept mild so generation artifacts are preserved."""
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)  # horizontal flip
    return img


class OpenCVDataGenerator(tf.keras.utils.Sequence):
    """Loads batches from disk with OpenCV so the full dataset never sits in RAM."""

    def __init__(self, paths, labels, batch_size=config.BATCH_SIZE,
                 shuffle=True, use_augmentation=False, **kwargs):
        super().__init__(**kwargs)
        self.paths = np.asarray(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_augmentation = use_augmentation
        self.indices = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, i):
        idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        X = np.empty((len(idx), config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
                     dtype=np.float32)
        for j, k in enumerate(idx):
            img = preprocess_image(load_image(self.paths[k]))
            if self.use_augmentation:
                img = random_augment(img)
            X[j] = img
        return X, self.labels[idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
