"""
Main training script for DDSD transfer learning.

Flow:
1. Parse configuration
2. Load pre-trained KWS model
3. Modify model for binary classification (remove blocks/layers, replace head)
4. Set trainable layers based on training mode
5. Load and prepare data (with noise/mixup augmentation)
6. Train model (with early stopping)
7. Evaluate on test set (loss, accuracy, AUC, EER, UAR)
8. Save model, history, and log results to CSV
"""
import tensorflow as tf
import os
import sys
import numpy as np
from datetime import datetime

# Import custom modules
from ddsd_config import get_config
from ddsd_utils import (
    RETRAINED_MODELS_DIR, EXPERIMENT_LOG_CSV, 
    DD_SUBFOLDERS, NDD_SUBFOLDERS, PRETRAINED_MODEL_PATH
)
from ddsd_builder import (
    load_pretrained_kws_model, build_binary_classification_model,
    set_trainable_layers, compile_model, print_model_info
)
from ddsd_data_loader import (
    get_file_paths_and_labels, split_dataset, load_custom_noise
)
from evaluation import (
    evaluate_model, save_model, save_history, log_experiment_to_csv
)


def main():
    """Main training pipeline."""
    
    print(f"\n{'='*60}")
    print(f"DDSD TRANSFER LEARNING - FINETUNE BINARY CLASSIFIER")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # STEP 1: CONFIGURATION
    # =========================================================================
    print("STEP 1: Loading configuration...")
    args, derived = get_config()
    
    print(f"  Train mode: {args.train_mode}")
    print(f"  Feature type: {args.feature_type}")
    print(f"  Noise type: {args.noise_type}")
    print(f"  Mixup: prob={args.mixup_prob}, alpha={args.mixup_alpha}")
    print(f"  Dataset: scale={args.dataset_scale}, size={args.dataset_size}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    
    # Create output directory
    os.makedirs(RETRAINED_MODELS_DIR, exist_ok=True)
    
    # =========================================================================
    # STEP 2: LOAD DATA
    # =========================================================================
    print(f"\nSTEP 2: Loading data...")
    filepaths, labels = get_file_paths_and_labels(
        "data", args.dd_subfolders, args.ndd_subfolders
    )
    
    # Load custom noise if specified
    noise_data = None
    if args.noise_type in ["babble", "factory"]:
        print(f"\nLoading {args.noise_type} noise...")
        noise_data = load_custom_noise(args.noise_type, noise_dir="noise")
        if noise_data is None:
            print(f"⚠️  Warning: {args.noise_type} noise not found, will use white noise as fallback")
    
    # =========================================================================
    # STEP 3: SPLIT DATA
    # =========================================================================
    print(f"\nSTEP 3: Splitting data into train/val/test...")
    
    if args.train_mode != "baseline":
        ds_train, ds_val, ds_test, n_train, n_val, n_test = split_dataset(
            filepaths, labels, args, derived, noise_data=noise_data
        )
    else:
        # Baseline mode: only need test set
        _, _, ds_test, _, _, n_test = split_dataset(
            filepaths, labels, args, derived, noise_data=noise_data
        )
        ds_train = None
        ds_val = None
        n_train = 0
        n_val = 0
    
    # =========================================================================
    # STEP 4: BUILD MODEL
    # =========================================================================
    print(f"\nSTEP 4: Building model...")
    
    # Load pre-trained KWS model
    pretrained_model = load_pretrained_kws_model(PRETRAINED_MODEL_PATH)
    
    # Build binary classification model
    model = build_binary_classification_model(
        pretrained_model,
        remove_layers=args.remove_layers,
        remove_blocks=args.remove_blocks
    )
    
    # Set trainable layers based on training mode
    model = set_trainable_layers(
        model, args.train_mode, args.unfreeze_layer_indices
    )
    
    # Compile model
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # Print model info
    total_params, trainable_params = print_model_info(model)
    
    # =========================================================================
    # STEP 5: TRAINING
    # =========================================================================
    print(f"\nSTEP 5: Training model...")
    
    history = None
    if args.train_mode != "baseline":
        # Setup callbacks
        callbacks = []
        if not args.disable_early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=args.es_monitor,
                    mode=args.es_mode,
                    patience=args.es_patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )
            print(f"  Early stopping enabled (monitor={args.es_monitor}, patience={args.es_patience})")
        
        # Train
        print(f"  Starting training for {args.epochs} epochs...\n")
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )
        print()
    else:
        print(f"  Baseline mode: skipping training (evaluation only)")
    
    # =========================================================================
    # STEP 6: EVALUATION
    # =========================================================================
    print(f"\nSTEP 6: Evaluating on test set...")
    metrics = evaluate_model(model, ds_test)
    
    # =========================================================================
    # STEP 7: SAVING RESULTS
    # =========================================================================
    print(f"STEP 7: Saving results...")
    
    # Generate model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_name = f"DDSD_{args.train_mode}_{timestamp}"
    
    # Create model folder
    model_dir = os.path.join(RETRAINED_MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    save_model(model, model_path)
    
    # Save training history
    if history is not None:
        history_path = os.path.join(model_dir, f"{model_name}_history.npz")
        save_history(history, history_path)
    
    # Log to CSV
    log_experiment_to_csv(
        EXPERIMENT_LOG_CSV,
        model_name, metrics, history, args, derived,
        n_train, n_val, n_test, total_params, trainable_params
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"{'='*60}")
    print(f"✅ TRAINING PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"Model saved to: {model_path}")
    if history is not None:
        print(f"History saved to: {history_path}")
    print(f"Results logged to: {EXPERIMENT_LOG_CSV}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()