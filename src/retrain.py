"""
Main training script: Transfer learning from KWS to DDSD
"""
import tensorflow as tf
import os
import sys
from ddsd_config import get_config
from ddsd_utils import *
from ddsd_builder import *
from get_dataset import get_file_paths_and_labels, make_dataset
from evaluation import evaluate_model, save_results

def main():
    # Parse configuration
    args, derived = get_config()
    
    # Setup directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Load data
    print(f"Loading data from {DATA_DIR}...")
    filepaths, labels = get_file_paths_and_labels(
        DATA_DIR, args.dd_subfolders, args.ndd_subfolders
    )
    
    # Split data
    train_ds, val_ds, test_ds = split_dataset(filepaths, labels, args, derived)
    
    # Load pretrained model
    print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}...")
    pretrained_model = load_pretrained_kws_model(PRETRAINED_MODEL_PATH)
    
    # Build binary classification model
    model = build_binary_classification_model(
        pretrained_model,
        remove_layers=args.remove_layers,
        remove_blocks=args.remove_blocks
    )
    
    # Set trainable layers
    model = set_trainable_layers(
        model, args.train_mode, args.unfreeze_layer_indices
    )
    
    # Compile
    model = compile_model(model, learning_rate=args.learning_rate)
    print_model_info(model)
    
    # Train
    if args.train_mode != "baseline":
        print("Starting training...")
        history = train_model(model, train_ds, val_ds, args)
    else:
        history = None
    
    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluate_model(model, test_ds)
    
    # Save
    print("Saving results...")
    save_results(model, metrics, history, args, derived)
    
    print("✅ Done!")

if __name__ == '__main__':
    main()