"""
Plot training history from saved .npz files
Specify model names at the top, then run the script
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

MODEL_NAMES = {
    "DDSD_layers_20260511_195722": "AllLayers_BS16_LR1e-3_white",
    "DDSD_layers_20260511_195756": "AllLayers_BS16_LR1e-3_pink",
    "DDSD_layers_20260511_195830": "AllLayers_BS16_LR1e-3_factory",
}
#best head: DDSD_head_20260511_192013
#best layers: DDSD_layers_20260511_193949
#best alllayers:DDSD_layers_20260511_195722

# For multiple models, use:
# MODEL_NAMES = {
#     "DDSD_head_20260504_112530": "Head_v1",
#     "DDSD_layers_20260504_113000": "Layers_v2",
#     "DDSD_scratch_20260504_113500": "Scratch_v3"
# }

RETRAINED_MODELS_DIR = "retrained_models"

def get_history_path(model_name):
    """Construct path to history file"""
    folder = os.path.join(RETRAINED_MODELS_DIR, model_name)
    history_file = os.path.join(folder, f"{model_name}_history.npz")
    return history_file

def plot_single(model_name, display_name, data):
    """Plot single experiment"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(display_name, fontsize=14, fontweight='bold')
    
    # Loss
    axes[0].plot(data['loss'], label='Train', linewidth=2, color='blue', linestyle='--')
    axes[0].plot(data['val_loss'], label='Val', linewidth=2, color='blue', linestyle='-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(data['acc'], label='Train', linewidth=2, color='blue', linestyle='--')
    axes[1].plot(data['val_acc'], label='Val', linewidth=2, color='blue', linestyle='-')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(data['auc'], label='Train', linewidth=2, color='blue', linestyle='--')
    axes[2].plot(data['val_auc'], label='Val', linewidth=2, color='blue', linestyle='-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('AUC')
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_comparison(model_dict, histories):
    """Compare multiple experiments"""
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Plot of training history", fontsize=14, fontweight='bold')
    
    for idx, (display_name, data) in enumerate(zip(model_dict.values(), histories)):
        color = colors[idx % len(colors)]
        
        # Loss
        axes[0].plot(data['loss'], label=f'{display_name} (Train)', linewidth=2, color=color, linestyle='--')
        axes[0].plot(data['val_loss'], label=f'{display_name} (Val)', linewidth=2, color=color, linestyle='-')
        
        # Accuracy
        axes[1].plot(data['acc'], label=f'{display_name} (Train)', linewidth=2, color=color, linestyle='--')
        axes[1].plot(data['val_acc'], label=f'{display_name} (Val)', linewidth=2, color=color, linestyle='-')
        
        # AUC
        axes[2].plot(data['auc'], label=f'{display_name} (Train)', linewidth=2, color=color, linestyle='--')
        axes[2].plot(data['val_auc'], label=f'{display_name} (Val)', linewidth=2, color=color, linestyle='-')
    
    # Configure axes
    titles = ['Loss', 'Accuracy', 'AUC']
    
    for j in range(3):
        axes[j].set_title(titles[j], fontsize=12)
        axes[j].set_xlabel('Epoch')
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=8, loc='best')
        if j > 0:  # Accuracy and AUC
            axes[j].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load histories
    histories = []
    for model_name in MODEL_NAMES.keys():
        history_path = get_history_path(model_name)
        
        if not os.path.exists(history_path):
            print(f"❌ File not found: {history_path}")
            exit(1)
        
        data = np.load(history_path)
        histories.append(data)
        print(f"✅ Loaded: {model_name}")
    
    # Plot
    if len(MODEL_NAMES) == 1:
        model_name = list(MODEL_NAMES.keys())[0]
        display_name = MODEL_NAMES[model_name]
        plot_single(model_name, display_name, histories[0])
    else:
        plot_comparison(MODEL_NAMES, histories)