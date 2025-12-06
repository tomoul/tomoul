#!/usr/bin/env python3
"""
Compare Zig mel spectrogram with Python/Whisper reference.

This script:
1. Computes mel spectrogram using Whisper's exact implementation
2. Exports it to a simple binary format
3. Compares statistics for debugging

Usage:
    python tools/compare_mel.py models/english_man.wav
"""

import argparse
import sys
import struct
from pathlib import Path

import numpy as np


def compute_whisper_mel(audio_path: str, n_mels: int = 80) -> np.ndarray:
    """Compute mel spectrogram using Whisper's exact method."""
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper not installed.")
        sys.exit(1)

    # Load and pad/trim to 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
    print(f"Audio mean: {audio.mean():.6f}, std: {audio.std():.6f}")

    # Use Whisper's log_mel_spectrogram function
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
    mel_np = mel.numpy().astype(np.float32)

    return audio, mel_np


def save_raw_audio(audio: np.ndarray, path: str):
    """Save audio as raw f32 samples for Zig to read."""
    with open(path, 'wb') as f:
        f.write(struct.pack('I', len(audio)))  # uint32 sample count
        f.write(audio.astype(np.float32).tobytes())


def save_raw_mel(mel: np.ndarray, path: str):
    """Save mel as raw f32 for Zig to read."""
    with open(path, 'wb') as f:
        n_mels, n_frames = mel.shape
        f.write(struct.pack('II', n_mels, n_frames))
        f.write(mel.astype(np.float32).tobytes())


def load_raw_mel(path: str) -> np.ndarray:
    """Load raw mel from Zig output."""
    with open(path, 'rb') as f:
        data = f.read()
    n_mels, n_frames = struct.unpack('II', data[:8])
    values = np.frombuffer(data[8:], dtype=np.float32)
    return values.reshape(n_mels, n_frames)


def main():
    parser = argparse.ArgumentParser(description="Compare mel spectrograms")
    parser.add_argument("audio", type=str, help="Input audio file")
    parser.add_argument("--n-mels", type=int, default=80)
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    print(f"=== Computing Whisper mel spectrogram ===")
    audio, mel = compute_whisper_mel(args.audio, args.n_mels)

    print(f"\n=== Mel spectrogram statistics ===")
    print(f"Shape: {mel.shape} [mels, frames]")
    print(f"Range: [{mel.min():.4f}, {mel.max():.4f}]")
    print(f"Mean: {mel.mean():.4f}, Std: {mel.std():.4f}")

    # Print some specific values for comparison
    print(f"\n=== Sample values (mel[0,:5]) ===")
    print(mel[0, :5])

    print(f"\n=== Sample values (mel[40,:5]) ===")
    print(mel[40, :5])

    print(f"\n=== Sample values (mel[-1,:5]) ===")
    print(mel[-1, :5])

    # Save for Zig comparison
    audio_path = args.audio.replace('.wav', '_audio.raw')
    mel_path = args.audio.replace('.wav', '_mel_ref.raw')

    save_raw_audio(audio, audio_path)
    save_raw_mel(mel, mel_path)

    print(f"\n=== Saved reference files ===")
    print(f"Audio: {audio_path}")
    print(f"Mel: {mel_path}")

    # If Zig output exists, compare
    zig_mel_path = args.audio.replace('.wav', '_mel_zig.raw')
    if Path(zig_mel_path).exists():
        print(f"\n=== Comparing with Zig output ===")
        zig_mel = load_raw_mel(zig_mel_path)

        print(f"Zig shape: {zig_mel.shape}")
        print(f"Zig range: [{zig_mel.min():.4f}, {zig_mel.max():.4f}]")
        print(f"Zig mean: {zig_mel.mean():.4f}, std: {zig_mel.std():.4f}")

        if mel.shape == zig_mel.shape:
            diff = np.abs(mel - zig_mel)
            print(f"\nDifference:")
            print(f"  Max: {diff.max():.6f}")
            print(f"  Mean: {diff.mean():.6f}")
            print(f"  MSE: {(diff**2).mean():.6f}")
        else:
            print(f"\nShape mismatch - cannot compute difference")


if __name__ == "__main__":
    main()
