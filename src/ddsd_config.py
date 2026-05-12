"""
Configuration and argument parsing for DDSD transfer learning training
"""
import argparse
import os
from ddsd_utils import DD_SUBFOLDERS, NDD_SUBFOLDERS

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer learning: KWS → Device-Directed Speech Detection"
    )
    
    # --- Architecture pruning ---
    parser.add_argument("--model_size", type=str, default="all",
                       choices=["all", "layer", "block"],
                       help="'all', 'layer', or 'block'")
    parser.add_argument("--remove_layers", type=int, nargs="*", default=[],
                       help="Layer indices to remove (e.g. 8 9 10)")
    parser.add_argument("--remove_blocks", type=int, nargs="*", default=[],
                       help="Block numbers to remove (e.g. 3 4)")
    
    # --- Training mode ---
    parser.add_argument("--train_mode", type=str, default="layers",
                       choices=["head", "scratch", "layers", "baseline"],
                       help="Which layers to train")
    parser.add_argument("--unfreeze_layer_indices", nargs="+",
                   default=["-7", "-6", "-5", "-4", "-3", "-2", "-1"],
                   help="Layers to unfreeze: 'all' or list of indices (e.g. -7 -6 -5 or all)")
    
    # --- Subfolder selection ---
    parser.add_argument("--dd_subfolders", type=str, nargs="+", 
                       default=DD_SUBFOLDERS)
    parser.add_argument("--ndd_subfolders", type=str, nargs="+", 
                       default=NDD_SUBFOLDERS)
    
    # --- Dataset scaling ---
    parser.add_argument("--dataset_scale", type=float, default=1.0,
                       help="Fraction of training data to use (0.0-1.0)")
    parser.add_argument("--dataset_size", type=int, default=-1,
                       help="Fixed number of samples per class (-1 = use all)")
    
    # --- Noise augmentation ---
    parser.add_argument("--noise_type", type=str, default="none",
                       choices=["none", "white", "pink", "brown", "babble", "factory"],
                       help="Type of noise to add during training")
    parser.add_argument("--noise_scale", type=float, default=0.0,
                       help="Noise mixing ratio (0.0-1.0)")
    parser.add_argument("--noise_frac", type=float, default=0.0,
                       help="Fraction of training samples to augment with noise (0.0-1.0)")
    
    # --- Mixup augmentation ---
    parser.add_argument("--mixup_prob", type=float, default=0.0,
                       help="Probability of applying mixup per batch (0.0-1.0)")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                       help="Beta distribution alpha for mixup lambda")
    
    # --- Data split ---
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    
    # --- Training hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # --- Early stopping ---
    parser.add_argument("--disable_early_stopping", action="store_true")
    parser.add_argument("--es_monitor", type=str, default="val_auc")
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--es_mode", type=str, default="max")
    
    # --- Audio & feature extraction ---
    parser.add_argument("--feature_type", type=str, default="mfcc",
                       choices=["mfcc", "lfbe"],
                       help="Type of audio features to extract")
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Audio sample rate (Hz)")
    parser.add_argument("--clip_duration_ms", type=int, default=1000,
                       help="Audio clip duration (milliseconds)")
    parser.add_argument("--window_size_ms", type=float, default=30.0,
                       help="STFT window size (milliseconds)")
    parser.add_argument("--window_stride_ms", type=float, default=20.0,
                       help="STFT window stride (milliseconds)")
    parser.add_argument("--dct_coefficient_count", type=int, default=10,
                       help="Number of MFCC/LFBE coefficients to keep")
    
    # --- MFCC-specific parameters ---
    parser.add_argument("--mel_lower_hz", type=float, default=20.0,
                       help="Lower mel frequency bound")
    parser.add_argument("--mel_upper_hz", type=float, default=4000.0,
                       help="Upper mel frequency bound")
    parser.add_argument("--mel_num_bins", type=int, default=40,
                       help="Number of mel bins")
    
    # --- LFBE-specific parameters ---
    parser.add_argument("--lfbe_preemphasis", type=float, default=(1 - 2**-5),
                       help="Preemphasis coefficient for LFBE")
    parser.add_argument("--lfbe_power_offset", type=int, default=52,
                       help="Power offset for LFBE")
    
    return parser.parse_args()

def get_config():
    """Parse args and compute derived constants"""
    args = parse_args()
    
    # Convert unfreeze_layer_indices: handle "all" or list of strings
    if len(args.unfreeze_layer_indices) == 1 and args.unfreeze_layer_indices[0].lower() == "all":
        args.unfreeze_layer_indices = "all"
    else:
        try:
            args.unfreeze_layer_indices = [int(idx) for idx in args.unfreeze_layer_indices]
        except ValueError:
            raise ValueError(f"unfreeze_layer_indices must be 'all' or list of integers, got {args.unfreeze_layer_indices}")
    
    # Validate splits
    assert abs(args.train_split + args.val_split + args.test_split - 1.0) < 1e-6, \
        "train_split + val_split + test_split must sum to 1.0"
    
    # Compute derived audio constants
    DESIRED_SAMPLES = int(args.sample_rate * args.clip_duration_ms / 1000)
    WINDOW_SIZE_SAMPLES = int(args.sample_rate * args.window_size_ms / 1000)
    WINDOW_STRIDE_SAMPLES = int(args.sample_rate * args.window_stride_ms / 1000)
    SPECTROGRAM_LENGTH = 1 + (DESIRED_SAMPLES - WINDOW_SIZE_SAMPLES) // WINDOW_STRIDE_SAMPLES
    
    derived = {
        'DESIRED_SAMPLES': DESIRED_SAMPLES,
        'WINDOW_SIZE_SAMPLES': WINDOW_SIZE_SAMPLES,
        'WINDOW_STRIDE_SAMPLES': WINDOW_STRIDE_SAMPLES,
        'SPECTROGRAM_LENGTH': SPECTROGRAM_LENGTH,
    }
    
    return args, derived