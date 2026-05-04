"""
DDSD (Device-Directed Speech Detection) utilities
Adapted from kws_utils.py for binary classification task
"""

# Import base utilities from kws_utils
from kws_utils import *

# Override/add DDSD-specific parameters
NUM_CLASSES = 2
CLASS_NAMES = ["DD (Device-Directed)", "NDD (Not Device-Directed)"]

# Data organization
DD_SUBFOLDERS = ["DD_NA", "DD_A"]      # Device-directed
NDD_SUBFOLDERS = ["NDD_S"]              # Not device-directed
DATA_DIR = "data"

# Model paths
PRETRAINED_MODEL_PATH = "trained_models/kws_ref_model"
SAVE_DIR = "retrained_models"
LOGS_DIR = "logs"

# Keep all audio processing params from kws_utils
# (sample_rate, clip_duration_ms, window_size_ms, etc.)