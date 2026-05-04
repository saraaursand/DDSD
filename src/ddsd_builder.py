"""
Model building and architecture utilities for transfer learning
"""
import tensorflow as tf
from keras_model import load_kws_model

# Block-to-layer mapping for the pretrained model
DS_BLOCK_LAYERS = {
    1: [5, 6, 7, 8, 9, 10],
    2: [11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22],
    4: [23, 24, 25, 26, 27, 28]
}

def load_pretrained_kws_model(model_path):
    """Load the pre-trained KWS model"""
    return tf.keras.models.load_model(model_path)

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
    
    # Rebuild: Input → Body (skip removed) → Binary Head
    x = pretrained_model.input
    for i, layer in enumerate(pretrained_model.layers[1:-1]):
        if (i + 1) not in skip:
            x = layer(x)
    
    # Add binary classification head
    output = tf.keras.layers.Dense(2, activation='softmax', name='binary_head')(x)
    model = tf.keras.Model(inputs=pretrained_model.input, outputs=output)
    
    return model

def set_trainable_layers(model, train_mode, unfreeze_indices="all"):
    """
    Control which layers are trainable based on training mode
    - "head": Only train new binary head
    - "layers": Train specific layers (unfrozen_indices)
    - "scratch": Train everything (reinitialize weights)
    - "baseline": No training
    """
    if train_mode == "head":
        for layer in model.layers[:-1]:
            layer.trainable = False
    
    elif train_mode == "scratch":
        for layer in model.layers:
            layer.trainable = True
            # Reinitialize weights
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))
    
    elif train_mode == "layers":
        for layer in model.layers:
            layer.trainable = False
        if unfreeze_indices == "all":
            for layer in model.layers:
                layer.trainable = True
        else:
            for idx in unfreeze_indices:
                model.layers[idx].trainable = True
    
    # "baseline": no changes
    
    return model

def compile_model(model, learning_rate=1e-4):
    """Compile model with appropriate optimizer and loss"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='acc'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    return model

def print_model_info(model):
    """Print model size and trainable parameter count"""
    total_params = sum(tf.size(w).numpy() for w in model.weights)
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    model_size_kb = (total_params * 4) / 1024  # float32 = 4 bytes
    
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Estimated size:   {model_size_kb:.1f} KB ({model_size_kb/1024:.2f} MB)")