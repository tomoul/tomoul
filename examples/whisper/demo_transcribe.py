#!/usr/bin/env python3
"""
Whisper Inference Demo for Tomoul

This demo shows how to use the exported Tomoul Whisper models for transcription.
It processes audio files and outputs the transcribed text.

Usage:
    python examples/whisper/demo_transcribe.py [audio_file]
    python examples/whisper/demo_transcribe.py examples/silero-vad/audio2.mp3
    python examples/whisper/demo_transcribe.py examples/silero-vad/french_audio2.wav

Requirements:
    pip install openai-whisper librosa numpy
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add tools directory to path for tl_format
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from tl_format import load_tl_file


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and preprocess audio file to 16kHz mono."""
    try:
        import librosa
    except ImportError:
        print("Error: librosa not installed. Install with: pip install librosa")
        sys.exit(1)

    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    print(f"  Duration: {len(audio) / sr:.2f}s, Sample rate: {sr}Hz")
    return audio


def compute_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sr: int = 16000,
) -> np.ndarray:
    """Compute log-Mel spectrogram matching Whisper's preprocessing."""
    try:
        import librosa
    except ImportError:
        print("Error: librosa not installed")
        sys.exit(1)

    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=8000,
    )

    # Convert to log scale (same as Whisper)
    log_mel = np.log10(np.maximum(mel, 1e-10))

    # Normalize (Whisper normalizes to max value)
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    return log_mel.astype(np.float32)


def pad_or_trim_mel(mel: np.ndarray, target_frames: int = 3000) -> np.ndarray:
    """Pad or trim mel spectrogram to target number of frames."""
    n_mels, n_frames = mel.shape

    if n_frames > target_frames:
        # Trim
        return mel[:, :target_frames]
    elif n_frames < target_frames:
        # Pad with zeros
        padded = np.zeros((n_mels, target_frames), dtype=mel.dtype)
        padded[:, :n_frames] = mel
        return padded
    return mel


def transcribe_with_openai_whisper(audio_path: str, model_name: str = "tiny"):
    """Transcribe using OpenAI's Whisper for reference."""
    try:
        import whisper
    except ImportError:
        print("OpenAI Whisper not installed, skipping reference transcription")
        return None

    print(f"\n=== OpenAI Whisper ({model_name}) Reference ===")
    model = whisper.load_model(model_name)

    start = time.perf_counter()
    result = model.transcribe(audio_path)
    elapsed = time.perf_counter() - start

    print(f"Time: {elapsed:.2f}s")
    print(f"Language: {result.get('language', 'unknown')}")
    print(f"Text: {result['text']}")

    return result


def demo_tomoul_inference(audio_path: str, model_path: str = "models/whisper_tiny.tl"):
    """Demo Tomoul model inference (loads weights and shows architecture)."""
    print(f"\n=== Tomoul Model Analysis ===")
    print(f"Model: {model_path}")

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Run: python tools/export_whisper.py -m tiny")
        return

    # Load model weights
    start = time.perf_counter()
    tensors, quant_format = load_tl_file(model_path)
    load_time = time.perf_counter() - start

    # Analyze model structure
    encoder_tensors = [k for k in tensors.keys() if k.startswith("encoder.")]
    decoder_tensors = [k for k in tensors.keys() if k.startswith("decoder.")]

    # Count encoder layers
    encoder_blocks = set()
    for k in encoder_tensors:
        if "blocks." in k:
            block_num = k.split("blocks.")[1].split(".")[0]
            encoder_blocks.add(int(block_num))

    # Count decoder layers
    decoder_blocks = set()
    for k in decoder_tensors:
        if "blocks." in k:
            block_num = k.split("blocks.")[1].split(".")[0]
            decoder_blocks.add(int(block_num))

    # Get embedding dimensions
    token_emb = tensors.get("decoder.token_embedding")
    pos_emb = tensors.get("encoder.positional_embedding")

    print(f"\nModel loaded in {load_time:.2f}s")
    print(f"Total tensors: {len(tensors)}")
    print(f"Encoder tensors: {len(encoder_tensors)}")
    print(f"Decoder tensors: {len(decoder_tensors)}")
    print(f"Encoder layers: {len(encoder_blocks)}")
    print(f"Decoder layers: {len(decoder_blocks)}")

    if token_emb is not None:
        print(f"Vocab size: {token_emb.shape[0]}")
        print(f"Hidden dim: {token_emb.shape[1]}")

    if pos_emb is not None:
        print(f"Audio context: {pos_emb.shape[0]} frames")

    # Calculate total parameters
    total_params = sum(t.size for t in tensors.values())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Load and preprocess audio
    audio = load_audio(audio_path)

    # Determine n_mels from model
    conv1_weight = tensors.get("encoder.conv1.weight")
    if conv1_weight is not None:
        n_mels = conv1_weight.shape[1]
    else:
        n_mels = 80

    print(f"\nComputing Mel spectrogram ({n_mels} mels)...")
    mel = compute_mel_spectrogram(audio, n_mels=n_mels)
    mel = pad_or_trim_mel(mel, target_frames=3000)
    print(f"Mel shape: {mel.shape}")

    return tensors, mel


def main():
    # Default audio files
    default_files = [
        "examples/silero-vad/audio2.mp3",
        "examples/silero-vad/french_audio2.wav",
    ]

    # Get audio file from args or use defaults
    if len(sys.argv) > 1:
        audio_files = sys.argv[1:]
    else:
        audio_files = default_files

    print("=" * 60)
    print("Whisper Transcription Demo")
    print("=" * 60)

    for audio_path in audio_files:
        if not Path(audio_path).exists():
            print(f"\nSkipping {audio_path} (not found)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Audio: {audio_path}")
        print("=" * 60)

        # Reference transcription with OpenAI Whisper
        result = transcribe_with_openai_whisper(audio_path, "tiny")

        # Demo Tomoul model loading and preprocessing
        demo_tomoul_inference(audio_path, "models/whisper_tiny.tl")

        # Also test with large-v3-turbo if available
        turbo_path = "models/whisper_large-v3-turbo_q8_0.tl"
        if Path(turbo_path).exists():
            print(f"\n--- Testing large-v3-turbo model ---")
            demo_tomoul_inference(audio_path, turbo_path)

        # Test distil-small.en if available
        distil_path = "models/distil-small.en_q8_0.tl"
        if Path(distil_path).exists():
            print(f"\n--- Testing distil-small.en model ---")
            demo_tomoul_inference(audio_path, distil_path)


if __name__ == "__main__":
    main()
