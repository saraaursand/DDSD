"""
Evaluation metrics and results logging
"""
import numpy as np
import csv
import os
from sklearn.metrics import roc_curve
from ddsd_utils import SAVE_DIR, LOGS_DIR

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
    
    print(f"Test Loss: {metrics['test_loss']:.6f}")
    print(f"Test Acc:  {metrics['test_acc']:.6f}")
    print(f"Test AUC:  {metrics['test_auc']:.6f}")
    print(f"EER:       {metrics['eer']:.6f} ({metrics['eer']*100:.2f}%)")
    print(f"UAR:       {metrics['uar']:.6f}")
    
    return metrics

def save_results(model, metrics, history, args, derived):
    """Save model, metrics, and log to CSV"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"DDSD_{args.train_mode}_{timestamp}"
    model_path = os.path.join(SAVE_DIR, f"{model_name}.h5")
    
    # Save model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save history if trained
    if history is not None:
        history_path = model_path.replace(".h5", "_history.npz")
        np.savez(history_path, **history.history)
    
    # Log to CSV
    csv_path = os.path.join(LOGS_DIR, "experiment_log.csv")
    # ... (CSV logging logic)