# Device-Directed Speech Detection (DDSD)

Transfer learning from Keyword Spotting (KWS) to Device-Directed Speech Detection (DDSD)

## Overview

This repository implements transfer learning to adapt a pre-trained Keyword Spotting (KWS) model from the [MLCommons Tiny](https://github.com/mlcommons/tiny) project to the task of **Device Directed Speech Detection (DDSD)** - binary classification to determine whether speech is directed towards a device or not.

The model learns to distinguish between:
- **DD (Device Directed)**
- **NDD (Non-Device Directed)**

### Key Features

- **Transfer Learning**: Re-trianing the pre-trained KWS model for DDSD task
- **Flexible Architecture**: Supports layer/block removal for model pruning
- **Training Modes**: 
  - `head`: Train only the binary classification head
  - `layers`: Train selected layers while keeping others frozen
  - `baseline`: Train all layers
  - `scratch`: Train from random initialization
- **Data Augmentation**: 
  - Noise augmentation (synthetic white/pink noise and real babble/factory noise)
  - Mixup augmentation (spatial concatenation with label smoothing)
- **Comprehensive Evaluation**: Metrics include accuracy, Area Under the ROC Curve (AUC), Equal Error Rate (EER), and Unweighted Average Recall (UAR)
- **Experiment Management**: YAML-based experiment configuration for batch training

## Project Structure

```
DDSD/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── experiments.yaml               # Experiment configurations
├── experiments_log.csv            # Results from all training runs
├── visualize_mfcc.py              # Utility for MFCC visualization
│
├── data/                          # Training/validation data (not in repo)
│   ├── DD/                        # Device-directed speech
│   │   ├── DD_A/                  # DD including the word "Alexa"
│   │   └── DD_NA/                 # DD without the word "Alexa"
│   └── NDD/                       # Non-device-directed speech
│       ├── NDD_S/                 # NDD spoken by different people S
│       └── NDD_J/                 # NDD spoken by the same person J
│
├── noise/                         # Background noise for augmentation
│   ├── babble/                    # Human speech background (from Noises repo)
│   └── factory/                   # Industrial noise (from Noises repo)
│
├── trained_models/                # Pre-trained KWS model (from Tiny repo)
│   └── kws_ref_model/             # Reference KWS model
│
├── retrained_models/              # Generated DDSD models from training runs
│   ├── DDSD_head_<timestamp>/
│   ├── DDSD_head_rm_blocks_<config>_<timestamp>/
│   └── ...
│
└── src/                           # Python source code
    ├── retrain.py                 # Main training pipeline
    ├── run_experiments.py         # Batch experiment runner
    ├── evaluation.py              # Evaluation metrics and logging
    ├── keras_model.py             # Keras model utilities
    ├── plot_history.py            # Training history visualization
    │
    ├── ddsd_builder.py            # Model architecture modifications
    ├── ddsd_config.py             # Argument parsing and configuration
    ├── ddsd_data_loader.py        # Data loading and preprocessing
    ├── ddsd_utils.py              # DDSD-specific utilities
    │
    ├── kws_data_loader.py         # KWS data utilities (from Tiny)
    ├── kws_util.py                # KWS utilities (from Tiny)
    └── __pycache__/               # Python cache
```


## Dependencies

**Requirements:**
- TensorFlow 2.12.0
- NumPy 1.22-1.23
- scikit-learn ≥ 1.3.0
- PyYAML ≥ 6.0
- Matplotlib ≥ 3.7.0


## Attribution and Licenses

This project builds upon:

### MLCommons Tiny
- **Source**: [MLCommons Tiny](https://github.com/mlcommons/tiny)
- **License**: Apache License 2.0
- **Usage**: Reference keyword spotting model and data loading utilities
- **Files**: 
  - `src/kws_util.py`
  - `src/kws_data_loader.py`
  - `trained_models/kws_ref_model/`

### Noises Repository
- **Source**: [Noises Repository](https://github.com/speechdnn/Noises)
- **License**: Research/Academic use only (see notice below)
- **Usage**: Babble and factory noise samples for augmentation
- **Files**:
  - `noise/babble/`
  - `noise/factory/`

White and pink noise used for augmentation are synthetically generated during training.

### License Notice for Noises Repository
The noise data from the Noises repository is provided under the restriction: "Just used for research or study, please be not for commercial." This project uses these samples for academic research purposes only.

## Author Notes

This project was developed for Masters thesis research on device-directed speech detection using transfer learning from keyword spotting models.
