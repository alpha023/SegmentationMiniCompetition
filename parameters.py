# parameters.py

import torch

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
IMAGE_SIZE = 300
RESIZE_SIZE = 320
CROP_SIZE = 300

BATCH_SIZE = 16
NUM_WORKERS = 2

# ---------------- TRAINING ----------------
EPOCHS = 100
LEARNING_RATE = 3e-4
PATIENCE = 25

# ---------------- MODEL ----------------
NUM_CLASSES = 21

# ---------------- LOSS ----------------
IGNORE_INDEX = 255

# ---------------- PATHS ----------------
IMAGE_DIR = "dataset/VOC2012_train_val/JPEGImages"
MASK_DIR = "dataset/VOC2012_train_val/SegmentationClass"

# ---------------- SAVE ----------------
MODEL_PATH = "model.pth"
SCORE_FILE = "score.txt"

# ---------------- LOGGING ----------------
LOG_FILE = "training_log.txt"

# ---------------- SEED (important for reproducibility) ----------------
SEED = 42