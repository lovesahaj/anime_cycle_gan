import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = 'cpu'
LAMBDA = 10
LRN_RATE = 2e-4

LAMBDA_GP = 10
IMG_CHANNELS = 3

LOAD_MODEL = True
MODEL_DIR = "../trained_models"
SAVE_MODEL = True

NUM_EPOCHS = 50
DEPTH_OF_CRITIC = 4
NUM_FEATURES = 64
BATCH_SIZE = 10
IMG_SIZE = 128
