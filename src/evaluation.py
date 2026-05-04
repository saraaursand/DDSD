"""
Evaluation metrics, results logging, and visualization for DDSD transfer learning.
Computes:
- Test loss, accuracy, AUC
- Equal Error Rate (EER)
- Unweighted Average Recall (UAR)
- Logs all results to CSV
"""
import numpy as np
import csv
import os
import datetime
import tensorflow as tf
from sklearn.metrics import roc_curve


def compute_eer(model, dataset):
    """
    Compute Equal Error Rate (EER) from model predictions.
    
    Args:
        model: Trained Keras model
        dataset: TF Dataset with test data
    
    Returns:
        eer: Equal Error Rate value
        eer_threshold: Threshold at which EER occurs
        y_true: True labels (one-hot)
        y_pred: Predicted probabilities
    """
    # Concatenate all batches
    y_true = np.concatenate([y.numpy() for _, y in dataset], axis=0)
    y_pred = model.predict(dataset)
    
    # Use DD class (column 0) as positive class
    fpr, tpr, thresholds = roc_curve(y_true[:, 0], y_pred[:, 0])
    frr = 1 - tpr
    
    # Find threshold where FPR = FRR (EER)
    eer_index = np.nanargmin(np.abs(fpr - frr))
    eer = (fpr[eer_index] + frr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    return eer, eer_threshold, y_true, y_pred


def compute_uar(y_true, y_pred):
    """
    Compute Unweighted Average Recall (UAR) for binary classification.
    
    Args:
        y_true: True labels (one-hot encoded, shape (N, 2))
        y_pred: Predicted probabilities (shape (N, 2))
    
    Returns:
        uar: Unweighted Average Recall (average of recall per class)
    """
    # Convert from one-hot to class indices
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    
    recall_per_class = []
    for cls in [0, 1]:
        tp = np.sum((y_true_labels == cls) & (y_pred_labels == cls))
        fn = np.sum((y_true_labels == cls) & (y_pred_labels != cls))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_per_class.append(recall)
    
    uar = np.mean(recall_per_class)
    return uar


def evaluate_model(model, test_ds):
    """
    Run all evaluation metrics on test set.
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset
    
    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING ON TEST SET")
    print(f"{'='*60}")
    
    # Standard metrics
    test_scores = model.evaluate(test_ds, return_dict=True)
    
    # EER and UAR
    eer, eer_threshold, y_true, y_pred = compute_eer(model, test_ds)
    uar = compute_uar(y_true, y_pred)
    
    metrics = {
        'test_loss': float(test_scores['loss']),
        'test_acc': float(test_scores['acc']),
        'test_auc': float(test_scores['auc']),
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'uar': float(uar),
    }
    
    # Print results
    print(f"Test Loss:      {metrics['test_loss']:.6f}")
    print(f"Test Accuracy:  {metrics['test_acc']:.6f}")
    print(f"Test AUC:       {metrics['test_auc']:.6f}")
    print(f"EER:            {metrics['eer']:.6f} ({metrics['eer']*100:.2f}%)")
    print(f"EER Threshold:  {metrics['eer_threshold']:.6f}")
    print(f"UAR:            {metrics['uar']:.6f}")
    print(f"{'='*60}\n")
    
    return metrics


def save_history(history, save_path):
    """
    Save training history to .npz file.
    
    Args:
        history: Keras history object from model.fit()
        save_path: Path where to save (e.g., "model_history.npz")
    
    Returns:
        save_path: Path to saved file
    """
    if history is None:
        return None
    
    np.savez(
        save_path,
        loss=np.array(history.history['loss']),
        acc=np.array(history.history['acc']),
        auc=np.array(history.history['auc']),
        val_loss=np.array(history.history['val_loss']),
        val_acc=np.array(history.history['val_acc']),
        val_auc=np.array(history.history['val_auc']),
    )
    print(f"✅ Training history saved to {save_path}")
    return save_path


def save_model(model, model_path):
    """
    Save trained model as .h5 file.
    
    Args:
        model: Trained Keras model
        model_path: Path where to save
    """
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")


def log_experiment_to_csv(csv_path, model_name, metrics, history, args, derived,
                          n_train, n_val, n_test, total_params, trainable_params):
    """
    Log experiment results to CSV file.
    
    Args:
        csv_path: Path to CSV file (e.g., "experiments_log.csv")
        model_name: Name of the model
        metrics: Dictionary with evaluation metrics
        history: Keras history object (or None for baseline)
        args: Configuration arguments object
        derived: Dictionary with derived constants
        n_train, n_val, n_test: Number of samples in each split
        total_params: Total model parameters
        trainable_params: Number of trainable parameters
    """
    # Compute training statistics if model was trained
    if history is not None:
        best_val_loss = float(min(history.history['val_loss']))
        best_val_acc = float(max(history.history['val_acc']))
        best_val_auc = float(max(history.history['val_auc']))
        epochs_trained = len(history.history['loss'])
    else:
        best_val_loss = "-"
        best_val_acc = "-"
        best_val_auc = "-"
        epochs_trained = "-"
    
    # Format unfrozen layers for logging
    if args.train_mode == "layers":
        if isinstance(args.unfreeze_layer_indices, list):
            unfrozen_layers = ', '.join(str(i) for i in args.unfreeze_layer_indices)
        else:
            unfrozen_layers = "all"
    else:
        unfrozen_layers = "-"
    
    # Format removed layers/blocks
    removed_layers = ','.join(str(l) for l in args.remove_layers) if args.remove_layers else "-"
    removed_blocks = ','.join(str(b) for b in args.remove_blocks) if args.remove_blocks else "-"
    
    # Calculate model size in MB
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    # Build CSV row
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        
        # Architecture
        "model_size": args.model_size,
        "removed_layers": removed_layers,
        "removed_blocks": removed_blocks,
        "train_mode": args.train_mode,
        "unfrozen_layers": unfrozen_layers,
        
        # Data
        "dd_subfolders": ','.join(args.dd_subfolders),
        "ndd_subfolders": ','.join(args.ndd_subfolders),
        "dataset_scale": args.dataset_scale,
        "dataset_size": args.dataset_size if args.dataset_size != -1 else "-",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        
        # Augmentation
        "noise_type": args.noise_type,
        "noise_scale": args.noise_scale,
        "noise_frac": args.noise_frac,
        "mixup_prob": args.mixup_prob,
        "mixup_alpha": args.mixup_alpha,
        
        # Training hyperparameters
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "epochs_trained": epochs_trained,
        "es_monitor": args.es_monitor if not args.disable_early_stopping else "-",
        "es_patience": args.es_patience if not args.disable_early_stopping else "-",
        
        # Training metrics
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_auc": best_val_auc,
        
        # Test metrics
        "test_loss": round(metrics['test_loss'], 6),
        "test_acc": round(metrics['test_acc'], 6),
        "test_auc": round(metrics['test_auc'], 6),
        "test_eer": round(metrics['eer'], 6),
        "test_uar": round(metrics['uar'], 6),
        
        # Model size
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
    }
    
    # Write CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"✅ Results logged to {csv_path}\n")