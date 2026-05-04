"""
Configuration and argument parsing for DDSD training
"""
import argparse
from ddsd_utils import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer learning: KWS → Device-Directed Speech Detection"
    )
    
    # Training mode
    parser.add_argument("--train_mode", type=str, default="layers",
                       choices=["head", "scratch", "layers", "baseline"],
                       help="Which layers to train")
    parser.add_argument("--unfreeze_layer_indices", nargs="+", 
                       default=["-7", "-6", "-5", "-4", "-3", "-2", "-1"])
    
    # Model architecture
    parser.add_argument("--model_size", type=str, default="all",
                       choices=["all", "layer", "block"])
    parser.add_argument("--remove_layers", type=int, nargs="*", default=[])
    parser.add_argument("--remove_blocks", type=int, nargs="*", default=[])
    
    # Data selection
    parser.add_argument("--dd_subfolders", type=str, nargs="+", 
                       default=DD_SUBFOLDERS)
    parser.add_argument("--ndd_subfolders", type=str, nargs="+", 
                       default=NDD_SUBFOLDERS)
    parser.add_argument("--dataset_scale", type=float, default=1.0)
    parser.add_argument("--dataset_size", type=int, default=-1)
    
    # Augmentation
    parser.add_argument("--noise_type", type=str, default="none",
                       choices=["none", "white", "pink", "brown"])
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--noise_frac", type=float, default=0.0)
    parser.add_argument("--mixup_prob", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    
    # Data split
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Early stopping
    parser.add_argument("--disable_early_stopping", action="store_true")
    parser.add_argument("--es_monitor", type=str, default="val_auc")
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--es_mode", type=str, default="max")
    
    return parser.parse_args()

def get_config():
    """Parse args and compute derived constants"""
    args = parse_args()
    
    # Derived constants
    DESIRED_SAMPLES = int(args.sample_rate * args.clip_duration_ms / 1000)
    WINDOW_SIZE_SAMPLES = int(args.sample_rate * args.window_size_ms / 1000)
    WINDOW_STRIDE_SAMPLES = int(args.sample_rate * args.window_stride_ms / 1000)
    SPECTROGRAM_LENGTH = 1 + (DESIRED_SAMPLES - WINDOW_SIZE_SAMPLES) // WINDOW_STRIDE_SAMPLES
    
    return args, {
        'DESIRED_SAMPLES': DESIRED_SAMPLES,
        'WINDOW_SIZE_SAMPLES': WINDOW_SIZE_SAMPLES,
        'WINDOW_STRIDE_SAMPLES': WINDOW_STRIDE_SAMPLES,
        'SPECTROGRAM_LENGTH': SPECTROGRAM_LENGTH,
    }