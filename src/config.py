import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
ASSETS_DIR = BASE_DIR / 'assets'

# Ensure assets directory exists
ASSETS_DIR.mkdir(exist_ok=True)

# File paths
DATA_FILE = DATA_DIR / 'water_potability.csv'

# Model Hyperparameters
INPUT_SIZE = 9
HIDDEN_1 = 64
HIDDEN_2 = 32
HIDDEN_3 = 16
OUTPUT_SIZE = 1

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
TEST_SPLIT = 0.2
RANDOM_STATE = 42
