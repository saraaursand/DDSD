"""
Script to visualize MFCC features using EXACT same processing as training.
Also supports noise and mixup augmentation visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys

# Import from your training code
sys.path.insert(0, 'src')
from ddsd_config import get_config
from ddsd_utils import DD_SUBFOLDERS, NDD_SUBFOLDERS
from ddsd_data_loader import load_custom_noise

def load_audio(file_path, desired_samples=32000):
    """Load audio exactly as in training"""
    raw = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    
    # Normalize by max absolute value
    audio = audio / (tf.reduce_max(tf.abs(audio)) + 1e-9)
    
    # Pad to desired length
    pad_amount = tf.maximum(0, desired_samples - tf.shape(audio)[0])
    audio = tf.pad(audio, [[0, pad_amount]])
    audio = audio[:desired_samples]
    audio = tf.cast(audio, tf.float32)
    
    return audio.numpy()

def extract_mfcc_exact(audio, args, derived):
    """Extract MFCC features EXACTLY as training does"""
    audio = tf.constant(audio, dtype=tf.float32)
    
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
    
    return mfccs.numpy()

def add_noise(audio, noise_type='white', noise_scale=0.1, noise_data=None, derived=None):
    """Add noise augmentation"""
    audio = tf.constant(audio, dtype=tf.float32)
    desired_samples = derived['DESIRED_SAMPLES']
    
    if noise_type == 'white':
        n = tf.random.normal(
            tf.shape(audio), 
            mean=0.0, 
            stddev=1.0, 
            dtype=tf.float32
        )
    
    elif noise_type == 'pink':
        white = tf.random.normal(
            tf.shape(audio), 
            mean=0.0, 
            stddev=1.0, 
            dtype=tf.float32
        )
        kernel = tf.ones([16, 1, 1], dtype=tf.float32) / 16.0
        n = tf.nn.conv1d(
            tf.reshape(white, [1, -1, 1]), 
            kernel, 
            stride=1, 
            padding='SAME'
        )[0, :, 0]
    
    elif noise_type == 'brown':
        white = tf.random.normal(
            tf.shape(audio), 
            mean=0.0, 
            stddev=1.0, 
            dtype=tf.float32
        )
        n = tf.math.cumsum(white)
        n = n / (tf.reduce_max(tf.abs(n)) + 1e-9)
    
    else:  # babble or factory
        if noise_data is not None and len(noise_data) > 0:
            noise_len = len(noise_data)
            if noise_len > desired_samples:
                start_idx = np.random.randint(0, noise_len - desired_samples)
                n = noise_data[start_idx:start_idx + desired_samples]
            else:
                n = np.pad(noise_data, (0, desired_samples - noise_len), 'constant')
            n = tf.constant(n, dtype=tf.float32)
        else:
            n = tf.random.normal(
                tf.shape(audio), 
                mean=0.0, 
                stddev=1.0, 
                dtype=tf.float32
            )
    
    n = tf.cast(n, tf.float32)
    
    # Match noise RMS to audio RMS
    audio_rms = tf.sqrt(tf.reduce_mean(tf.square(audio)))
    noise_rms = tf.sqrt(tf.reduce_mean(tf.square(n)))
    n = n * (audio_rms / (noise_rms + 1e-9))
    
    # Mix
    mixed = (1.0 - noise_scale) * audio + noise_scale * n
    return tf.clip_by_value(mixed, -1.0, 1.0).numpy()

def apply_mixup(mfcc1, mfcc2, alpha=0.2, b_min=0.1, b_max=0.4):
    # Sample lambda from Beta(alpha, alpha) using gamma ratio (matches training)
    gamma1 = np.random.gamma(alpha, 1.0)
    gamma2 = np.random.gamma(alpha, 1.0)
    lam = gamma1 / (gamma1 + gamma2)
    lam = np.clip(lam, 1e-5, 1.0 - 1e-5)
    
    # Spatial mixup along time axis
    time_steps = mfcc1.shape[0]
    cut_idx = int(np.round(lam * time_steps))
    cut_idx = np.clip(cut_idx, 0, time_steps)  # Ensure valid index
    
    # Mix features
    mixed = np.vstack([mfcc1[:cut_idx], mfcc2[cut_idx:]])
    
    # Label smoothing parameter (matches training Equation 7)
    b = -4.0 * (b_max - b_min) * (lam - 0.5) ** 2 + b_max
    
    return mixed, lam, cut_idx, b

def get_label_from_folder(folder_path):
    """Convert folder path to nice label"""
    folder_name = folder_path.replace('\\', '/').split('/')[-1]
    
    labels = {
        'DD_A': 'Device-Directed with "Alexa"',
        'DD_NA': 'Device-Directed without "Alexa"',
        'NDD_S': 'Non-Device-Directed'
    }
    
    return labels.get(folder_name, folder_name)

def build_title(folder_path, noise_type=None, noise_scale=None, mixup_alpha=None):
    """Build title with augmentation info"""
    base_title = get_label_from_folder(folder_path)
    
    augmentations = []
    if noise_type:
        augmentations.append(f"Noise: {noise_type}, scale: {noise_scale}")
    if mixup_alpha:
        augmentations.append(f"Mixup: {mixup_alpha}")
    
    if augmentations:
        return f"{base_title} ({', '.join(augmentations)})"
    return base_title

def plot_mfcc(mfcc, folder_path="", noise_type=None, noise_scale=None, mixup_alpha=None, figsize=(12, 5)):
    """Plot MFCC spectrogram"""
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(
        mfcc.T,  # Transpose for proper orientation
        aspect='auto',
        origin='lower',
        cmap='plasma',
        interpolation='nearest'
    )
    
    # Build title with augmentation info
    title = build_title(folder_path, noise_type, noise_scale, mixup_alpha)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Frame', fontsize=12)
    ax.set_ylabel('MFCC Coefficient', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude', fontsize=11)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main visualization function"""
    
    # Load config
    args, derived = get_config()
    
    print(f"\n{'='*60}")
    print(f"MFCC Visualization - Exact Training Pipeline")
    print(f"{'='*60}\n")
    
    # Get folder
    folder_path = input("Enter folder path (e.g., 'data/DD/DD_A' or 'data/NDD/NDD_S'): ").strip()
    
    if not os.path.isdir(folder_path):
        print(f"❌ Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    # Get audio files
    audio_files = [f for f in os.listdir(folder_path) 
                   if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    
    if not audio_files:
        print(f"❌ Error: No audio files found in {folder_path}")
        sys.exit(1)
    
    first_file = audio_files[0]
    file_path = os.path.join(folder_path, first_file)
    
    print(f"Folder: {folder_path}")
    print(f"File: {first_file}")
    print(f"Total files in folder: {len(audio_files)}\n")
    
    # Load audio
    print(f"Loading audio...")
    audio = load_audio(file_path, derived['DESIRED_SAMPLES'])
    
    # Extract original MFCC
    print(f"Extracting MFCC...")
    mfcc_original = extract_mfcc_exact(audio, args, derived)
    
    print(f"MFCC shape: {mfcc_original.shape}")
    print(f"{'='*60}\n")
    
    # Plot original
    plot_mfcc(mfcc_original, folder_path=folder_path)
    
    # Ask for augmentations
    print(f"\n{'='*60}")
    print("Add augmentations? (Optional)")
    print(f"{'='*60}\n")
    
    # Noise augmentation
    noise_type = None
    noise_scale = None
    add_noise_aug = input("Add noise augmentation? (y/n): ").strip().lower() == 'y'
    if add_noise_aug:
        print("\nNoise types: white, pink, brown, babble, factory")
        noise_type = input("Select noise type: ").strip().lower()
        noise_scale = float(input("Noise scale (0.0-1.0): ").strip())
        
        noise_data = None
        if noise_type in ['babble', 'factory']:
            noise_data = load_custom_noise(noise_type, noise_dir="noise")
        
        audio_noisy = add_noise(audio, noise_type, noise_scale, noise_data, derived)
        mfcc_noisy = extract_mfcc_exact(audio_noisy, args, derived)
        
        plot_mfcc(mfcc_noisy, folder_path=folder_path, 
                  noise_type=noise_type, noise_scale=noise_scale)
    
        # Mixup augmentation
    mixup_alpha = None
    add_mixup_aug = input("\nAdd mixup augmentation? (y/n): ").strip().lower() == 'y'
    if add_mixup_aug:
        # Load second file
        second_file = audio_files[1] if len(audio_files) > 1 else audio_files[0]
        file_path_2 = os.path.join(folder_path, second_file)
        
        print(f"\nMixup file 1: {first_file}")
        print(f"Mixup file 2: {second_file}")
        
        audio_2 = load_audio(file_path_2, derived['DESIRED_SAMPLES'])
        mfcc_2 = extract_mfcc_exact(audio_2, args, derived)
        
        mixup_alpha = float(input("Mixup alpha (default 1.0): ").strip() or "1.0")
        b_min = float(input("Label smoothing min (default 0.1): ").strip() or "0.1")
        b_max = float(input("Label smoothing max (default 0.4): ").strip() or "0.4")
        
        mfcc_mixed, lam, cut_idx, b = apply_mixup(mfcc_original, mfcc_2, mixup_alpha, b_min, b_max)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original
        im1 = axes[0].imshow(mfcc_original.T, aspect='auto', origin='lower', cmap='plasma')
        axes[0].set_title(f"{get_label_from_folder(folder_path)}\n(Sample 1)", fontweight='bold')
        axes[0].set_xlabel('Time Frame')
        axes[0].set_ylabel('MFCC Coefficient')
        plt.colorbar(im1, ax=axes[0])
        
        # Second sample
        im2 = axes[1].imshow(mfcc_2.T, aspect='auto', origin='lower', cmap='plasma')
        axes[1].set_title(f"{get_label_from_folder(folder_path)}\n(Sample 2)", fontweight='bold')
        axes[1].set_xlabel('Time Frame')
        axes[1].set_ylabel('MFCC Coefficient')
        plt.colorbar(im2, ax=axes[1])
        
        # Mixed with visualization info
        im3 = axes[2].imshow(mfcc_mixed.T, aspect='auto', origin='lower', cmap='plasma')
        axes[2].axvline(x=cut_idx-0.5, color='red', linewidth=2, linestyle='--', 
                       label=f'Cut at frame {cut_idx}/{mfcc_original.shape[0]}')
        axes[2].set_title(
            f"Mixed (α={mixup_alpha}, λ={lam:.3f}, b={b:.3f})\n"
            f"({cut_idx} frames from S1, {mfcc_original.shape[0]-cut_idx} from S2)",
            fontweight='bold'
        )
        axes[2].set_xlabel('Time Frame')
        axes[2].set_ylabel('MFCC Coefficient')
        axes[2].legend()
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
        
        # Print mixing info
        print(f"\n{'='*60}")
        print(f"Mixup Summary:")
        print(f"  Alpha (Beta param): {mixup_alpha}")
        print(f"  Lambda (mixing ratio): {lam:.4f}")
        print(f"  Cut index: {cut_idx} / {mfcc_original.shape[0]}")
        print(f"  Label smoothing (b): {b:.4f}")
        print(f"  Sample 1: {cut_idx} frames ({cut_idx/mfcc_original.shape[0]*100:.1f}%)")
        print(f"  Sample 2: {mfcc_original.shape[0]-cut_idx} frames ({(mfcc_original.shape[0]-cut_idx)/mfcc_original.shape[0]*100:.1f}%)")
        print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    print("✅ Done!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()