"""
Data loading, preprocessing, and augmentation for DDSD transfer learning.
Handles:
- Loading DD/NDD audio files with binary labels
- Audio normalization and padding
- Feature extraction (MFCC/LFBE) 
- Noise augmentation (white, pink, brown, babble, factory)
- Mixup augmentation (temporal concatenation with label smoothing)
"""
import tensorflow as tf
import numpy as np
import os
import functools


def get_file_paths_and_labels(data_dir, dd_subfolders, ndd_subfolders):
    """
    Load all audio file paths and create binary one-hot labels.
    
    Args:
        data_dir: Path to data directory (e.g., "data")
        dd_subfolders: List of DD subfolder names (e.g., ["DD_NA", "DD_A"])
        ndd_subfolders: List of NDD subfolder names (e.g., ["NDD_S"])
    
    Returns:
        filepaths: numpy array of file paths
        labels: numpy array of one-hot labels shape (N, 2)
                - [1, 0] for device-directed (DD)
                - [0, 1] for not device-directed (NDD)
    """
    filepaths, labels = [], []
    
    def get_files(folder, label):
        """Get all .wav files from folder with given label."""
        if not os.path.exists(folder):
            print(f"⚠️  Warning: Folder not found: {folder}")
            return [], []
        
        files = sorted([
            os.path.join(folder, fname) 
            for fname in os.listdir(folder) 
            if fname.endswith('.wav')
        ])
        return files, [label] * len(files)
    
    # Load DD (device-directed) samples - label [1, 0]
    print(f"Loading DD (Device-Directed) samples...")
    dd_count = 0
    for subfolder in dd_subfolders:
        folder = os.path.join(data_dir, "DD", subfolder)
        files, lbls = get_files(folder, [1, 0])
        filepaths.extend(files)
        labels.extend(lbls)
        dd_count += len(files)
        print(f"  {subfolder}: {len(files)} files")
    
    # Load NDD (not device-directed) samples - label [0, 1]
    print(f"Loading NDD (Not Device-Directed) samples...")
    ndd_count = 0
    for subfolder in ndd_subfolders:
        folder = os.path.join(data_dir, "NDD", subfolder)
        files, lbls = get_files(folder, [0, 1])
        filepaths.extend(files)
        labels.extend(lbls)
        ndd_count += len(files)
        print(f"  {subfolder}: {len(files)} files")
    
    print(f"\n✅ Total loaded: {dd_count} DD + {ndd_count} NDD = {len(filepaths)} samples\n")
    
    return np.array(filepaths), np.array(labels)


def load_and_preprocess(filepath, label, args, derived, is_training=False, noise_data=None):
    """
    Load audio file, preprocess, extract features, and optionally add noise/mixup.
    
    Args:
        filepath: Path to .wav file
        label: One-hot encoded label [batch_size, 2]
        args: Configuration object with all parameters
        derived: Dictionary with DESIRED_SAMPLES, WINDOW_SIZE_SAMPLES, etc.
        is_training: Whether to apply augmentation (noise, mixup)
        noise_data: Pre-loaded noise array for custom noise (babble/factory)
    
    Returns:
        features: MFCC or LFBE spectrogram [SPECTROGRAM_LENGTH, DCT_COEFFICIENT_COUNT, 1]
        label: One-hot label [2]
    """
    # Load and decode audio
    raw = tf.io.read_file(filepath)
    audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    
    # Normalize by max absolute value
    audio = audio / (tf.reduce_max(tf.abs(audio)) + 1e-9)
    
    # Pad to desired length
    pad_amount = tf.maximum(0, derived['DESIRED_SAMPLES'] - tf.shape(audio)[0])
    audio = tf.pad(audio, [[0, pad_amount]])
    audio = audio[:derived['DESIRED_SAMPLES']]
    audio = tf.cast(audio, tf.float32)
    
    # --- NOISE AUGMENTATION (only during training) ---
    if is_training and args.noise_type != 'none' and args.noise_scale > 0.0:
        # Decide if this sample gets noise (noise_frac probability)
        apply_noise = tf.random.uniform([], 0, 1) < args.noise_frac
        
        def apply_noise_fn():
            """Generate and mix noise into audio."""
            if args.noise_type == 'white':
                # White noise
                n = tf.random.normal(
                    tf.shape(audio), 
                    mean=0.0, 
                    stddev=1.0, 
                    dtype=tf.float32
                )
            
            elif args.noise_type == 'pink':
                # Pink noise (rough approximation via low-pass filtering)
                white = tf.random.normal(
                    tf.shape(audio), 
                    mean=0.0, 
                    stddev=1.0, 
                    dtype=tf.float32
                )
                # Simple moving average for low-pass effect
                kernel = tf.ones([16, 1, 1], dtype=tf.float32) / 16.0
                n = tf.nn.conv1d(
                    tf.reshape(white, [1, -1, 1]), 
                    kernel, 
                    stride=1, 
                    padding='SAME'
                )[0, :, 0]
            
            elif args.noise_type == 'brown':
                # Brown noise (integrated white noise)
                white = tf.random.normal(
                    tf.shape(audio), 
                    mean=0.0, 
                    stddev=1.0, 
                    dtype=tf.float32
                )
                n = tf.math.cumsum(white)
                n = n / (tf.reduce_max(tf.abs(n)) + 1e-9)
            
            else:
                # Custom noise from .wav file (babble, factory)
                if noise_data is not None and len(noise_data) > 0:
                    noise_tensor = tf.constant(noise_data, dtype=tf.float32)
                    noise_len = tf.shape(noise_tensor)[0]
                    
                    if noise_len > derived['DESIRED_SAMPLES']:
                        # Use tf.random.uniform and tf.slice (pure TF ops)
                        start_idx = tf.random.uniform(
                            [], 
                            0, 
                            noise_len - derived['DESIRED_SAMPLES'] + 1, 
                            dtype=tf.int32
                        )
                        n = tf.slice(
                            noise_tensor, 
                            [start_idx], 
                            [derived['DESIRED_SAMPLES']]
                        )
                    else:
                        # Pad if noise is shorter
                        pad_amount = derived['DESIRED_SAMPLES'] - noise_len
                        n = tf.pad(noise_tensor, [[0, pad_amount]])
                    
                    n = tf.cast(n, tf.float32)
                else:
                    # Fallback to white noise
                    n = tf.random.normal(
                        [derived['DESIRED_SAMPLES']], 
                        mean=0.0, 
                        stddev=1.0, 
                        dtype=tf.float32
                    )
            
            n = tf.cast(n, tf.float32)
            
            # Match noise RMS to audio RMS
            audio_rms = tf.sqrt(tf.reduce_mean(tf.square(audio)))
            noise_rms = tf.sqrt(tf.reduce_mean(tf.square(n)))
            n = n * (audio_rms / (noise_rms + 1e-9))
            
            # Mix: (random uniform):
            noise_scale_random = tf.random.uniform([], 0, args.noise_scale, dtype=tf.float32)
            mixed = (1.0 - noise_scale_random) * audio + noise_scale_random * n
            return tf.clip_by_value(mixed, -1.0, 1.0)
        
        audio = tf.cond(apply_noise, apply_noise_fn, lambda: audio)
    
    # --- FEATURE EXTRACTION ---
    if args.feature_type == "mfcc":
        features = extract_mfcc(audio, args, derived)
    elif args.feature_type == "lfbe":
        features = extract_lfbe(audio, args, derived)
    else:
        raise ValueError(f"Unknown feature_type: {args.feature_type}")
    
    return features, label


def extract_mfcc(audio, args, derived):
    """Extract MFCC features from audio."""
    stfts = tf.signal.stft(
        audio,
        frame_length=derived['WINDOW_SIZE_SAMPLES'],
        frame_step=derived['WINDOW_STRIDE_SAMPLES'],
        fft_length=None,
        window_fn=tf.signal.hann_window
    )
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = stfts.shape[-1]
    
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        args.mel_num_bins,
        num_spectrogram_bins,
        args.sample_rate,
        args.mel_lower_hz,
        args.mel_upper_hz
    )
    mel = tf.tensordot(spectrograms, linear_to_mel, 1)
    log_mel = tf.math.log(mel + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :args.dct_coefficient_count]
    
    return tf.reshape(mfccs, [derived['SPECTROGRAM_LENGTH'], args.dct_coefficient_count, 1])


def extract_lfbe(audio, args, derived):
    """Extract LFBE features from audio."""
    audio = tf.expand_dims(audio, 0)
    audio = tf.pad(tensor=audio, paddings=tf.constant([[0, 0], [1, 0]]), mode='CONSTANT')
    audio = audio[:, 1:] - args.lfbe_preemphasis * audio[:, :-1]
    audio = tf.squeeze(audio)
    
    stfts = tf.signal.stft(
        audio,
        frame_length=derived['WINDOW_SIZE_SAMPLES'],
        frame_step=derived['WINDOW_STRIDE_SAMPLES'],
        fft_length=None,
        window_fn=functools.partial(tf.signal.hamming_window, periodic=False),
        pad_end=False
    )
    magspec = tf.abs(stfts)
    num_spectrogram_bins = magspec.shape[-1]
    
    powspec = (1 / derived['WINDOW_SIZE_SAMPLES']) * tf.square(magspec)
    powspec = tf.clip_by_value(powspec, 1e-30, tf.reduce_max(powspec))
    
    def log10(x):
        return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))
    
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        args.dct_coefficient_count,
        num_spectrogram_bins,
        args.sample_rate,
        0.0,
        args.sample_rate / 2.0
    )
    mel = tf.tensordot(powspec, linear_to_mel, 1)
    log_mel = 10 * log10(mel)
    log_mel = (log_mel + args.lfbe_power_offset - 32 + 32.0) / 64.0
    log_mel = tf.clip_by_value(log_mel, 0, 1)
    
    return tf.expand_dims(log_mel, -1)


def load_custom_noise(noise_type, noise_dir="noise"):
    """
    Pre-load custom noise file (.wav).
    
    Args:
        noise_type: "babble" or "factory"
        noise_dir: Directory containing noise files
    
    Returns:
        noise_audio: numpy array of noise samples, or None if not found
    """
    noise_file = os.path.join(noise_dir, f"{noise_type}.wav")
    
    if not os.path.exists(noise_file):
        print(f"⚠️  Noise file not found: {noise_file}")
        return None
    
    print(f"Loading {noise_type} noise from {noise_file}...")
    raw = tf.io.read_file(noise_file)
    audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1).numpy()
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    
    print(f"✅ Loaded {len(audio)} samples ({len(audio)/16000:.2f}s at 16kHz)")
    return audio


def make_dataset(filepaths, labels, args, derived, noise_data=None, 
                 shuffle=False, is_training=False):
    """
    Create TensorFlow Dataset with batching, noise augmentation, and mixup.
    
    Args:
        filepaths: Array of audio file paths
        labels: Array of one-hot labels shape (N, 2)
        args: Configuration object
        derived: Dictionary with audio constants
        noise_data: Pre-loaded custom noise array
        shuffle: Whether to shuffle dataset
        is_training: Whether to apply augmentation
    
    Returns:
        TF Dataset batched and prefetched
    """
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=args.seed)
    
    # Load and preprocess audio
    def map_fn(filepath, label):
        return load_and_preprocess(filepath, label, args, derived, 
                                   is_training=is_training, noise_data=noise_data)
    
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size)
    
    # --- MIXUP AUGMENTATION ---
    if is_training and args.mixup_prob > 0.0 and args.mixup_alpha > 0.0:
        def apply_mixup(features, labels):
            """Apply temporal mixup to batch."""
            # Decide if batch gets mixup
            do_mixup = tf.random.uniform([]) < args.mixup_prob
            
            if not do_mixup:
                return features, tf.cast(labels, tf.float32)
            
            batch_size = tf.shape(features)[0]
            
            # Shuffle indices to create pairs
            indices = tf.random.shuffle(tf.range(batch_size))
            features_shuffled = tf.gather(features, indices)
            labels_shuffled = tf.gather(labels, indices)
            
            # Sample lambda from Beta distribution
            lam = tf.random.gamma([batch_size], args.mixup_alpha) / (
                tf.random.gamma([batch_size], args.mixup_alpha) + 
                tf.random.gamma([batch_size], args.mixup_alpha)
            )
            lam = tf.clip_by_value(lam, 0.0, 1.0)
            
            # Mix along time axis (axis 1)
            time_steps = tf.shape(features)[1]
            cut_indices = tf.cast(tf.round(lam * tf.cast(time_steps, tf.float32)), tf.int32)
            
            # Create mask for time axis
            time_range = tf.range(time_steps)[tf.newaxis, :, tf.newaxis, tf.newaxis]
            cut_indices_expanded = cut_indices[:, tf.newaxis, tf.newaxis, tf.newaxis]
            mask = tf.cast(time_range < cut_indices_expanded, tf.float32)
            
            # Mix features
            mixed_features = mask * features + (1.0 - mask) * features_shuffled
            
            # Mix labels
            lam_expanded = tf.expand_dims(tf.cast(lam, tf.float32), axis=1)
            labels_float = tf.cast(labels, tf.float32)
            labels_shuffled_float = tf.cast(labels_shuffled, tf.float32)
            mixed_labels = lam_expanded * labels_float + (1.0 - lam_expanded) * labels_shuffled_float
            
            return mixed_features, mixed_labels
        
        ds = ds.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Just cast labels to float32 for consistency
        ds = ds.map(lambda x, y: (x, tf.cast(y, tf.float32)), 
                   num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(tf.data.AUTOTUNE)


def split_dataset(filepaths, labels, args, derived, noise_data=None):
    """
    Split dataset into train/val/test with stratification.
    Then cap training set size if specified.
    
    Args:
        filepaths: Array of file paths
        labels: Array of one-hot labels
        args: Configuration object
        derived: Dictionary with audio constants
        noise_data: Pre-loaded custom noise
    
    Returns:
        ds_train, ds_val, ds_test: TF Datasets
        n_train, n_val, n_test: Number of samples in each split
    """
    # Shuffle then split
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(filepaths))
    filepaths, labels = filepaths[indices], labels[indices]
    
    n = len(filepaths)
    n_train = int(args.train_split * n)
    n_val = int(args.val_split * n)
    n_test = n - n_train - n_val
    
    train_files = filepaths[:n_train]
    train_labels = labels[:n_train]
    val_files = filepaths[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]
    test_files = filepaths[n_train + n_val:]
    test_labels = labels[n_train + n_val:]
    
    print(f"Dataset split:")
    print(f"  Total pool: {(labels[:, 0]==1).sum()} DD + {(labels[:, 1]==1).sum()} NDD = {len(labels)}")
    print(f"  Before capping: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    # Cap training set size if specified
    if args.dataset_size != -1:
        # Fixed number per class
        dd_idx = np.where(train_labels[:, 0] == 1)[0][:args.dataset_size]
        ndd_idx = np.where(train_labels[:, 1] == 1)[0][:args.dataset_size]
        indices = np.concatenate([dd_idx, ndd_idx])
        train_files, train_labels = train_files[indices], train_labels[indices]
    elif args.dataset_scale < 1.0:
        # Fraction of available
        dd_idx = np.where(train_labels[:, 0] == 1)[0]
        ndd_idx = np.where(train_labels[:, 1] == 1)[0]
        dd_idx = dd_idx[:max(1, int(len(dd_idx) * args.dataset_scale))]
        ndd_idx = ndd_idx[:max(1, int(len(ndd_idx) * args.dataset_scale))]
        indices = np.concatenate([dd_idx, ndd_idx])
        train_files, train_labels = train_files[indices], train_labels[indices]
    
    # Shuffle train again after capping
    train_indices = rng.permutation(len(train_files))
    train_files, train_labels = train_files[train_indices], train_labels[train_indices]
    
    print(f"  After capping: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}\n")
    
    # Create datasets
    ds_train = make_dataset(train_files, train_labels, args, derived, 
                           noise_data=noise_data, shuffle=True, is_training=True)
    ds_val = make_dataset(val_files, val_labels, args, derived, 
                         shuffle=False, is_training=False)
    ds_test = make_dataset(test_files, test_labels, args, derived, 
                          shuffle=False, is_training=False)
    
    return ds_train, ds_val, ds_test, len(train_files), len(val_files), len(test_files)