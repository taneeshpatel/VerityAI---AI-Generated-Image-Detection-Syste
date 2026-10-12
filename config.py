"""Central configuration for DeepGuard AI."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Paths ----
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepguard_cnn.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ---- Image / training settings ----
IMG_SIZE = 256            # images are resized to 256x256
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
SEED = 42

# ---- Labels ----
# Folder names inside data/train and data/test.
# Index 0 = AI-generated (FAKE), index 1 = real photo (REAL).
# The model's sigmoid output is P(REAL).
CLASS_NAMES = ["FAKE", "REAL"]
DISPLAY_NAMES = {"FAKE": "AI-Generated", "REAL": "Real"}
THRESHOLD = 0.5

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
