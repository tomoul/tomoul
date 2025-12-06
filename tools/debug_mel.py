#!/usr/bin/env python3
"""Debug mel spectrogram by comparing Zig vs Python values."""

import numpy as np
import struct

# Load the raw mel reference
with open('models/english_man_mel_ref.raw', 'rb') as f:
    n_mels, n_frames = struct.unpack('II', f.read(8))
    mel_ref = np.frombuffer(f.read(), dtype=np.float32).reshape(n_mels, n_frames)

print("=== Python Reference Mel ===")
print(f"Shape: {mel_ref.shape}")
print(f"Range: [{mel_ref.min():.4f}, {mel_ref.max():.4f}]")
print(f"Mean: {mel_ref.mean():.4f}, Std: {mel_ref.std():.4f}")
print(f"mel[0, :5]: {mel_ref[0, :5]}")
print(f"mel[40, :5]: {mel_ref[40, :5]}")
print(f"mel[79, :5]: {mel_ref[79, :5]}")
