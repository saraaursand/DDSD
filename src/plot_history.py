"""
Plot training history from saved .npz files
Specify model names at the top, then run the script
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

MODEL_NAMES = [
    "DDSD_layers_20260504_112530"
]

# For multiple models, use:
# MODEL_NAMES = [
#     "DDSD_head_20260504_112530",
#     "DDSD_layers_20260504_113000",
#     "DDSD_scratch_20260504_113500"
# ]

RETRAINED_MODELS_DIR = "retrained_models"

def get_history_path(model_name):
    """Construct path to history file"""
    folder = os.path.join(RETRAINED_MODELS_DIR, model_name)
    history_file = os.path.join(folder, f"{model_name}_history.npz")
    return history_file

def plot_single(model_name, data):
    """Plot single experiment"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(model_name, fontsize=14, fontweight='bold')
    
    # Loss
    axes[0].plot(data['loss'], label='Train', linewidth=2)
    axes[0].plot(data['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(data['acc'], label='Train', linewidth=2)
    axes[1].plot(data['val_acc'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(data['auc'], label='Train', linewidth=2)
    axes[2].plot(data['val_auc'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('AUC')
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_comparison(model_names, histories):
    """Compare multiple experiments"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"Comparison: {len(model_names)} Experiments", fontsize=14, fontweight='bold')
    
    for idx, (name, data) in enumerate(zip(model_names, histories)):
        color = colors[idx]
        
        # Train Loss
        axes[0, 0].plot(data['loss'], label=name, linewidth=2, color=color)
        # Train Accuracy
        axes[0, 1].plot(data['acc'], label=name, linewidth=2, color=color)
        # Train AUC
        axes[0, 2].plot(data['auc'], label=name, linewidth=2, color=color)
        # Val Loss
        axes[1, 0].plot(data['val_loss'], label=name, linewidth=2, color=color)
        # Val Accuracy
        axes[1, 1].plot(data['val_acc'], label=name, linewidth=2, color=color)
        # Val AUC
        axes[1, 2].plot(data['val_auc'], label=name, linewidth=2, color=color)
    
    # Configure axes
    titles = [['Train Loss', 'Train Accuracy', 'Train AUC'],
              ['Val Loss', 'Val Accuracy', 'Val AUC']]
    
    for i in range(2):
        for j in range(3):
            axes[i, j].set_title(titles[i][j])
            axes[i, j].set_xlabel('Epoch')
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].legend(fontsize=8)
            if j > 0:  # Accuracy and AUC
                axes[i, j].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load histories
    histories = []
    for model_name in MODEL_NAMES:
        history_path = get_history_path(model_name)
        
        if not os.path.exists(history_path):
            print(f"❌ File not found: {history_path}")
            exit(1)
        
        data = np.load(history_path)
        histories.append(data)
        print(f"✅ Loaded: {model_name}")
    
    # Plot
    if len(MODEL_NAMES) == 1:
        plot_single(MODEL_NAMES[0], histories[0])
    else:
        plot_comparison(MODEL_NAMES, histories)