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
from evaluation import evaluate_model, save_results, log_experiment_to_csv

def main():
    # Parse configuration
    args, derived = get_config()
    
    # Setup directories
    os.makedirs(RETRAINED_MODELS_DIR, exist_ok=True)
    
    # Load data
    print(f"Loading data from {DATA_DIR}...")
    filepaths, labels = get_file_paths_and_labels(
        DATA_DIR, args.dd_subfolders, args.ndd_subfolders
    )
    
    # Split data
    train_ds, val_ds, test_ds, n_train, n_val, n_test = split_dataset(
        filepaths, labels, args, derived
    )
    
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
    
    # Get model stats
    total_params = sum(tf.size(w).numpy() for w in model.weights)
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    
    # Train
    history = None
    if args.train_mode != "baseline":
        print("Starting training...")
        history = train_model(model, train_ds, val_ds, args)
    else:
        print("Baseline mode: skipping training")
    
    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluate_model(model, test_ds)
    
    # Save model, history, and results
    print("Saving results...")
    model_name, total_params, trainable_params = save_results(
        model, metrics, history, args, derived
    )
    
    # Log to CSV
    log_experiment_to_csv(
        model_name, metrics, history, args, derived, 
        n_train, n_val, n_test, trainable_params, total_params
    )
    
    print("✅ Training pipeline complete!")

if __name__ == '__main__':
    main()