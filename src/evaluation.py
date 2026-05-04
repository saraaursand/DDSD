"""
Evaluation metrics, results logging, and visualization
"""
import numpy as np
import csv
import os
import datetime
import tensorflow as tf
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from ddsd_utils import RETRAINED_MODELS_DIR, EXPERIMENT_LOG_CSV

def compute_eer(model, dataset):
    """Compute Equal Error Rate"""
    y_true = np.concatenate([y.numpy() for _, y in dataset], axis=0)
    y_pred = model.predict(dataset)
    
    fpr, tpr, thresholds = roc_curve(y_true[:, 0], y_pred[:, 0])
    frr = 1 - tpr
    
    eer_index = np.nanargmin(np.abs(fpr - frr))
    eer = (fpr[eer_index] + frr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    return eer, eer_threshold, y_true, y_pred

def compute_uar(y_true, y_pred):
    """Compute Unweighted Average Recall (UAR)"""
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    
    recall_per_class = []
    for cls in [0, 1]:
        tp = np.sum((y_true_labels == cls) & (y_pred_labels == cls))
        fn = np.sum((y_true_labels == cls) & (y_pred_labels != cls))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_per_class.append(recall)
    
    return np.mean(recall_per_class)

def evaluate_model(model, test_ds):
    """Run all evaluation metrics"""
    test_scores = model.evaluate(test_ds, return_dict=True)
    eer, eer_threshold, y_true, y_pred = compute_eer(model, test_ds)
    uar = compute_uar(y_true, y_pred)
    
    metrics = {
        'test_loss': test_scores['loss'],
        'test_acc': test_scores['acc'],
        'test_auc': test_scores['auc'],
        'eer': eer,
        'eer_threshold': eer_threshold,
        'uar': uar,
    }
    
    print(f"\n{'='*50}")
    print(f"TEST SET METRICS")
    print(f"{'='*50}")
    print(f"Test Loss: {metrics['test_loss']:.6f}")
    print(f"Test Acc:  {metrics['test_acc']:.6f}")
    print(f"Test AUC:  {metrics['test_auc']:.6f}")
    print(f"EER:       {metrics['eer']:.6f} ({metrics['eer']*100:.2f}%)")
    print(f"UAR:       {metrics['uar']:.6f}")
    print(f"{'='*50}\n")
    
    return metrics

def save_history(history, model_folder_path):
    """Save training history as .npz file"""
    if history is None:
        return None
    
    model_name = os.path.basename(model_folder_path)
    history_path = os.path.join(model_folder_path, f"{model_name}_history.npz")
    np.savez(
        history_path,
        loss=np.array(history.history['loss']),
        acc=np.array(history.history['acc']),
        auc=np.array(history.history['auc']),
        val_loss=np.array(history.history['val_loss']),
        val_acc=np.array(history.history['val_acc']),
        val_auc=np.array(history.history['val_auc']),
    )
    print(f"✅ Training history saved to {history_path}")
    return history_path

def plot_training_history(history_path):
    """Plot training/validation curves from .npz file"""
    data = np.load(history_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(data['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(data['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(data['acc'], label='Train Acc', linewidth=2)
    axes[1].plot(data['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(data['auc'], label='Train AUC', linewidth=2)
    axes[2].plot(data['val_auc'], label='Val AUC', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('AUC over Epochs')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = history_path.replace("_history.npz", "_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to {plot_path}")
    plt.close()

def log_experiment_to_csv(model_name, metrics, history, args, derived, n_train, n_val, n_test, trainable_params, total_params):
    """Log experiment results to CSV at root level"""
    csv_path = EXPERIMENT_LOG_CSV  # Now at root level
    
    # Compute best validation metrics if trained
    if history is not None:
        best_val_loss = min(history.history['val_loss'])
        best_val_acc = max(history.history['val_acc'])
        best_val_auc = max(history.history['val_auc'])
        epochs_trained = len(history.history['loss'])
    else:
        best_val_loss = best_val_acc = best_val_auc = "-"
        epochs_trained = "-"
    
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "train_mode": args.train_mode,
        "model_size": args.model_size,
        "removed_layers": ','.join(str(l) for l in args.remove_layers) if args.remove_layers else "-",
        "removed_blocks": ','.join(str(b) for b in args.remove_blocks) if args.remove_blocks else "-",
        "noise_type": args.noise_type,
        "noise_scale": args.noise_scale,
        "noise_frac": args.noise_frac,
        "mixup_prob": args.mixup_prob,
        "mixup_alpha": args.mixup_alpha,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "epochs_trained": epochs_trained,
        "learning_rate": args.learning_rate,
        "es_monitor": args.es_monitor if not args.disable_early_stopping else "-",
        "es_patience": args.es_patience if not args.disable_early_stopping else "-",
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_auc": best_val_auc,
        "test_loss": round(metrics['test_loss'], 6),
        "test_acc": round(metrics['test_acc'], 6),
        "test_auc": round(metrics['test_auc'], 6),
        "test_eer": round(metrics['eer'], 6),
        "test_uar": round(metrics['uar'], 6),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round((total_params * 4) / (1024 * 1024), 2),
    }
    
    # Write CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"✅ Results logged to {csv_path}")

def save_results(model, metrics, history, args, derived):
    """Orchestrate all saving: model in timestamped folder, history, plots, and CSV"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"DDSD_{args.train_mode}_{timestamp}"
    
    # Create experiment folder
    model_folder = os.path.join(RETRAINED_MODELS_DIR, model_name)
    os.makedirs(model_folder, exist_ok=True)
    
    model_path = os.path.join(model_folder, f"{model_name}.h5")
    
    # Save model
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save training history as .npz (inside same folder)
    if history is not None:
        history_path = save_history(history, model_folder)
        # Plot the history (inside same folder)
        plot_training_history(history_path)
    
    # Get model stats
    total_params = sum(tf.size(w).numpy() for w in model.weights)
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    
    print(f"\n{'='*50}")
    print(f"All results saved in: {model_folder}")
    print(f"Params:    {trainable_params:,} / {total_params:,}")
    print(f"{'='*50}\n")
    
    return model_name, total_params, trainable_params