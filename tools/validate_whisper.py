#!/usr/bin/env python3
"""
Validate Whisper implementation against PyTorch reference.

Creates validation fixtures and tests intermediate outputs.

Usage:
    python validate_whisper.py --create-fixture
    python validate_whisper.py --test-encoder
    python validate_whisper.py --test-full
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import whisper
except ImportError:
    print("Error: OpenAI Whisper not installed.")
    print("Install with: pip install openai-whisper")
    sys.exit(1)

from tl_format import export_tensors, QuantFormat


def create_test_audio() -> np.ndarray:
    """Create a simple test audio signal (sine wave saying nothing useful)."""
    # 3 seconds of audio at 16kHz
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Simple tone at 440Hz (A note)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    return audio


def audio_to_mel(audio: np.ndarray, model) -> torch.Tensor:
    """Convert audio to Mel spectrogram using Whisper's preprocessing."""
    # Whisper expects 30 seconds of audio, padded or trimmed
    audio_tensor = torch.from_numpy(audio).float()

    # Pad/trim to 30 seconds
    audio_tensor = whisper.pad_or_trim(audio_tensor)

    # Compute Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_tensor).to(model.device)

    return mel


def run_encoder(model, mel: torch.Tensor) -> torch.Tensor:
    """Run just the encoder portion."""
    with torch.no_grad():
        return model.encoder(mel.unsqueeze(0))


def run_decoder_step(model, tokens: List[int], audio_features: torch.Tensor) -> torch.Tensor:
    """Run a single decoder step."""
    with torch.no_grad():
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=model.device)
        return model.decoder(tokens_tensor, audio_features)


def create_validation_fixture(output_dir: str, model_name: str = "tiny"):
    """Create a validation fixture with Mel spectrogram and expected outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper {model_name}...")
    model = whisper.load_model(model_name)
    model.eval()

    # Create test audio
    print("Creating test audio...")
    audio = create_test_audio()

    # Convert to Mel spectrogram
    print("Computing Mel spectrogram...")
    mel = audio_to_mel(audio, model)
    mel_np = mel.cpu().numpy()  # [80, 3000]

    # Run encoder
    print("Running encoder...")
    audio_features = run_encoder(model, mel)
    encoder_output_np = audio_features.squeeze(0).cpu().numpy()  # [1500, 384/512/...]

    # Get initial prompt tokens
    initial_tokens = [
        whisper.tokenizer.LANGUAGES["en"],  # This gives the language ID
    ]
    # Actually, let's use the standard prompt
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
    sot = tokenizer.sot
    lang_en = tokenizer.sot + 1  # <|en|> is typically right after SOT in the vocab
    transcribe = tokenizer.transcribe
    no_timestamps = tokenizer.no_timestamps

    prompt_tokens = [sot, lang_en, transcribe, no_timestamps]
    print(f"Prompt tokens: {prompt_tokens}")

    # Run decoder for first prediction
    print("Running decoder step...")
    decoder_output = run_decoder_step(model, prompt_tokens, audio_features)
    decoder_logits_np = decoder_output.squeeze(0).cpu().numpy()  # [seq_len, vocab_size]

    # Get the last token's logits (prediction for next token)
    last_logits = decoder_logits_np[-1]  # [vocab_size]
    predicted_token = int(np.argmax(last_logits))
    print(f"Predicted next token: {predicted_token}")

    # Export fixtures
    fixtures = {
        "mel": mel_np,
        "encoder_output": encoder_output_np,
        "decoder_logits": decoder_logits_np,
        "last_logits": last_logits,
    }

    fixture_path = output_dir / f"whisper_{model_name}_fixture.tl"
    print(f"\nExporting fixture to {fixture_path}...")
    export_tensors(fixtures, str(fixture_path), QuantFormat.F32)

    # Also save metadata as text
    meta_path = output_dir / f"whisper_{model_name}_fixture_meta.txt"
    with open(meta_path, 'w') as f:
        f.write(f"model={model_name}\n")
        f.write(f"mel_shape={mel_np.shape}\n")
        f.write(f"encoder_output_shape={encoder_output_np.shape}\n")
        f.write(f"decoder_logits_shape={decoder_logits_np.shape}\n")
        f.write(f"prompt_tokens={prompt_tokens}\n")
        f.write(f"predicted_token={predicted_token}\n")
        f.write(f"sot={sot}\n")
        f.write(f"eot={tokenizer.eot}\n")
        f.write(f"transcribe={transcribe}\n")
        f.write(f"no_timestamps={no_timestamps}\n")

    print(f"Metadata saved to {meta_path}")
    print("\nFixture created successfully!")
    print(f"  Mel shape: {mel_np.shape}")
    print(f"  Encoder output shape: {encoder_output_np.shape}")
    print(f"  Decoder logits shape: {decoder_logits_np.shape}")

    return fixture_path


def test_transcription(model_name: str = "tiny"):
    """Test full transcription on a simple audio sample."""
    print(f"Loading Whisper {model_name}...")
    model = whisper.load_model(model_name)

    # Create test audio
    audio = create_test_audio()

    # Transcribe
    print("Transcribing test audio...")
    result = model.transcribe(audio, language="en")

    print(f"\nTranscription: '{result['text']}'")
    print(f"Language: {result['language']}")

    # Show tokens
    if 'segments' in result and result['segments']:
        for segment in result['segments']:
            if 'tokens' in segment:
                print(f"Tokens: {segment['tokens'][:20]}...")  # First 20 tokens

    return result


def compare_outputs(zig_output: np.ndarray, torch_output: np.ndarray, name: str, tolerance: float = 1e-4):
    """Compare Zig and PyTorch outputs."""
    if zig_output.shape != torch_output.shape:
        print(f"FAIL {name}: Shape mismatch - Zig {zig_output.shape} vs PyTorch {torch_output.shape}")
        return False

    diff = np.abs(zig_output - torch_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    if max_diff > tolerance:
        print(f"FAIL {name}: Max diff {max_diff:.6f} > tolerance {tolerance}")
        return False

    print(f"PASS {name}: Max diff {max_diff:.6f}, Mean diff {mean_diff:.6f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate Whisper implementation")
    parser.add_argument("--create-fixture", action="store_true", help="Create validation fixture")
    parser.add_argument("--test-transcription", action="store_true", help="Test full transcription")
    parser.add_argument("-m", "--model", type=str, default="tiny", help="Model variant (default: tiny)")
    parser.add_argument("-o", "--output", type=str, default="./models", help="Output directory")

    args = parser.parse_args()

    if args.create_fixture:
        create_validation_fixture(args.output, args.model)
    elif args.test_transcription:
        test_transcription(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
