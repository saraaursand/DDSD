"""
Model building and architecture utilities for transfer learning
"""
import tensorflow as tf

# Block-to-layer mapping for the pretrained model
DS_BLOCK_LAYERS = {
    1: [5, 6, 7, 8, 9, 10],
    2: [11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22],
    4: [23, 24, 25, 26, 27, 28]
}

def load_pretrained_kws_model(model_path):
    """Load the pre-trained KWS model"""
    print(f"Loading pre-trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully. Total layers: {len(model.layers)}")
    return model

def build_binary_classification_model(pretrained_model, remove_layers=None, remove_blocks=None):
    """
    Rebuild model with:
    - Removed layers/blocks (if specified)
    - Binary classification head (2 classes instead of 12)
    - Preserved pre-trained weights in body
    """
    if remove_layers is None:
        remove_layers = []
    if remove_blocks is None:
        remove_blocks = []
    
    # Build skip set
    skip = set(remove_layers)
    for block in remove_blocks:
        skip.update(DS_BLOCK_LAYERS.get(block, []))
    
    print(f"Building binary classification model...")
    if skip:
        print(f"  Skipping layers: {sorted(skip)}")
    
    # Rebuild: Input → Body (skip removed) → Binary Head
    x = pretrained_model.input
    for i, layer in enumerate(pretrained_model.layers[1:-1]):
        if (i + 1) not in skip:
            x = layer(x)
    
    # Add binary classification head
    output = tf.keras.layers.Dense(2, activation='softmax', name='binary_head')(x)
    model = tf.keras.Model(inputs=pretrained_model.input, outputs=output)
    
    print(f"✅ Binary model created. Output shape: (None, 2)")
    return model

def set_trainable_layers(model, train_mode, unfreeze_indices="all"):
    """
    Control which layers are trainable based on training mode
    - "head": Only train new binary head
    - "layers": Train specific layers (unfrozen_indices)
    - "scratch": Train everything (reinitialize weights)
    - "baseline": No training
    """
    print(f"Setting trainable layers for mode: {train_mode}")
    
    if train_mode == "head":
        for layer in model.layers[:-1]:
            layer.trainable = False
        print(f"  Frozen: {len(model.layers)-1} body layers")
        print(f"  Trainable: binary head (1 layer)")
    
    elif train_mode == "scratch":
        for layer in model.layers:
            layer.trainable = True
            # Reinitialize weights
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))
        print(f"  Trainable: ALL {len(model.layers)} layers (weights reinitialized)")
    
    elif train_mode == "layers":
        for layer in model.layers:
            layer.trainable = False
        if unfreeze_indices == "all":
            for layer in model.layers:
                layer.trainable = True
            print(f"  Trainable: ALL {len(model.layers)} layers")
        else:
            trainable_count = 0
            for idx in unfreeze_indices:
                if 0 <= idx < len(model.layers):
                    model.layers[idx].trainable = True
                    trainable_count += 1
            print(f"  Trainable: {trainable_count} layers at indices {unfreeze_indices}")
    
    elif train_mode == "baseline":
        for layer in model.layers:
            layer.trainable = False
        print(f"  Frozen: ALL {len(model.layers)} layers (evaluation only)")
    
    return model

def compile_model(model, learning_rate=1e-4):
    """Compile model with appropriate optimizer and loss"""
    print(f"Compiling model with learning_rate={learning_rate}...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='acc'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    print("✅ Model compiled")
    return model

def print_model_info(model):
    """Print model size and trainable parameter count"""
    total_params = sum(tf.size(w).numpy() for w in model.weights)
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    model_size_kb = (total_params * 4) / 1024  # float32 = 4 bytes
    
    print(f"\n{'='*60}")
    print(f"MODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Estimated size:   {model_size_kb:.1f} KB ({model_size_kb/1024:.2f} MB)")
    print(f"{'='*60}\n")
    
    return total_params, trainable_params