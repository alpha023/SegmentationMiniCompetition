# parameters.py

# ---------------- DATA ----------------
IMAGE_SIZE = 300
BATCH_SIZE = 8
NUM_WORKERS = 2

# ---------------- TRAINING ----------------
EPOCHS = 10
LEARNING_RATE = 3e-4

# ---------------- MODEL ----------------
NUM_CLASSES = 21

# ---------------- AUGMENTATION ----------------
RESIZE_SIZE = 320
CROP_SIZE = 300

# ---------------- LOSS ----------------
IGNORE_INDEX = 255

# ---------------- PATHS ----------------
IMAGE_DIR = "dataset/VOC2012_train_val/JPEGImages"
MASK_DIR = "dataset/VOC2012_train_val/SegmentationClass"

# ---------------- SAVE ----------------
MODEL_PATH = "model.pth"
SCORE_FILE = "score.txt"

PATIENCE=10