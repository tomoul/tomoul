#!/usr/bin/env python3
"""
Export audio file to Mel spectrogram in .tl format for Zig inference.

This uses Whisper's exact preprocessing to ensure compatibility.

Usage:
    python tools/export_mel.py examples/silero-vad/english_man.flac -o mel.tl
    python tools/export_mel.py audio.wav -o mel.tl --n-mels 80
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from tl_format import export_tensors, QuantFormat


def load_and_preprocess_audio(audio_path: str) -> np.ndarray:
    """Load audio using Whisper's preprocessing."""
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper not installed.")
        print("Install with: pip install openai-whisper")
        sys.exit(1)

    # Load and pad/trim to 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    return audio


def compute_mel_spectrogram(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Compute log-Mel spectrogram using Whisper's exact method."""
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper not installed.")
        sys.exit(1)

    # Use Whisper's log_mel_spectrogram function
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)

    # Result is [n_mels, n_frames] torch tensor - keep this format
    mel_np = mel.numpy().astype(np.float32)

    return mel_np


def main():
    parser = argparse.ArgumentParser(
        description="Export audio to Mel spectrogram in .tl format"
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Input audio file (wav, mp3, flac, etc.)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="mel.tl",
        help="Output .tl file (default: mel.tl)"
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of mel bins (default: 80, use 128 for large-v3)"
    )

    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    print(f"Loading audio: {args.audio}")
    audio = load_and_preprocess_audio(args.audio)
    print(f"  Audio length: {len(audio) / 16000:.2f}s ({len(audio)} samples)")

    print(f"\nComputing Mel spectrogram ({args.n_mels} mels)...")
    mel = compute_mel_spectrogram(audio, n_mels=args.n_mels)
    print(f"  Mel shape: {mel.shape} [mels, frames]")

    # Export to .tl format
    print(f"\nExporting to {args.output}...")
    tensors = {"mel": mel}
    export_tensors(tensors, args.output, QuantFormat.F32, verify=False)

    # Print file size
    file_size = Path(args.output).stat().st_size
    print(f"  File size: {file_size / 1024:.1f} KB")

    print(f"\nDone! Use with: zig build -Dmodel=whisper-tiny run -- transcribe {args.output}")


if __name__ == "__main__":
    main()
