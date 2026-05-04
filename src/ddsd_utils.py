"""
DDSD (Device-Directed Speech Detection) utilities
Adapted from kws_utils.py for binary classification task
"""

import os

# Import base utilities from kws_utils
from kws_util import *

# Override/add DDSD-specific parameters
NUM_CLASSES = 2
CLASS_NAMES = ["DD (Device-Directed)", "NDD (Not Device-Directed)"]

# Data organization
DD_SUBFOLDERS = ["DD_NA", "DD_A"]      # Device-directed
NDD_SUBFOLDERS = ["NDD_S"]              # Not device-directed
DATA_DIR = "data"

# Model paths
PRETRAINED_MODEL_PATH = "trained_models/kws_ref_model"
RETRAINED_MODELS_DIR = "retrained_models"
SAVE_DIR = "retrained_models"
LOGS_DIR = "logs"

# Noise augmentation paths
NOISE_DIR = "noise"
BACKGROUND_NOISE_DIR_NAME = "noise"

# Experiment logging
EXPERIMENT_LOG_CSV = "experiments_log.csv"

# Keep all audio processing params from kws_utils
# (sample_rate, clip_duration_ms, window_size_ms, etc.)
