import os
import torch

# Dataset Config
DATA_DIR = "dataset_patches"
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 1 # 1 for binary segmentation

# Training Config
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
SAVE_PATH = "checkpoints/best_model.pth"

# Model Config
BACKBONE_NAME = "resnet50"
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 8
TRANSFORMER_DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Device Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
