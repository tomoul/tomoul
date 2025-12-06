#!/usr/bin/env python3
"""
Export Whisper models to .tl format for Tomoul.

Usage:
    python export_whisper.py -m tiny -o ./models/
    python export_whisper.py -m base -o ./models/ -q q8_0
    python export_whisper.py -m large-v3-turbo -o ./models/ -q q8_0
    python export_whisper.py -m distil-small.en -o ./models/ -q q8_0

Supported models:
  - OpenAI Whisper: tiny, base, small, medium, large, large-v3-turbo
  - HuggingFace distil-whisper: distil-small.en

The exporter:
1. Downloads the Whisper model from OpenAI or HuggingFace
2. Extracts encoder and decoder weights
3. Pre-transposes weights for optimal SIMD matmul
4. Exports to .tl format with optional quantization
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Import shared tl_format utilities
from tl_format import export_tensors, QuantFormat


# Model configurations (matches Zig config.zig)
WHISPER_CONFIGS = {
    "tiny": {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 384,
        "n_audio_head": 6,
        "n_audio_layer": 4,
        "n_vocab": 51865,
        "n_text_ctx": 448,
        "n_text_state": 384,
        "n_text_head": 6,
        "n_text_layer": 4,
    },
    "base": {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 512,
        "n_audio_head": 8,
        "n_audio_layer": 6,
        "n_vocab": 51865,
        "n_text_ctx": 448,
        "n_text_state": 512,
        "n_text_head": 8,
        "n_text_layer": 6,
    },
    "small": {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 768,
        "n_audio_head": 12,
        "n_audio_layer": 12,
        "n_vocab": 51865,
        "n_text_ctx": 448,
        "n_text_state": 768,
        "n_text_head": 12,
        "n_text_layer": 12,
    },
    "medium": {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 1024,
        "n_audio_head": 16,
        "n_audio_layer": 24,
        "n_vocab": 51865,
        "n_text_ctx": 448,
        "n_text_state": 1024,
        "n_text_head": 16,
        "n_text_layer": 24,
    },
    "large": {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_vocab": 51865,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": 32,
    },
    # Large-v3-turbo: distilled model with 128 mels and only 4 decoder layers
    "large-v3-turbo": {
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_vocab": 51866,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": 4,
    },
    # Alias for large-v3-turbo
    "turbo": {
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_vocab": 51866,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": 4,
    },
    # Distil-Whisper models (from HuggingFace, not OpenAI)
    "distil-small.en": {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 768,
        "n_audio_head": 12,
        "n_audio_layer": 12,
        "n_vocab": 51864,  # slightly smaller vocab
        "n_text_ctx": 448,
        "n_text_state": 768,
        "n_text_head": 12,
        "n_text_layer": 4,  # distilled decoder
    },
}


def is_distil_whisper(model_name: str) -> bool:
    """Check if this is a distil-whisper model (HuggingFace)."""
    return model_name.startswith("distil-")


def load_whisper_model(model_name: str):
    """Load Whisper model from OpenAI or HuggingFace."""
    if is_distil_whisper(model_name):
        return load_distil_whisper_model(model_name)

    try:
        import whisper
    except ImportError:
        print("Error: OpenAI Whisper not installed.")
        print("Install with: pip install openai-whisper")
        sys.exit(1)

    print(f"Loading Whisper {model_name}...")
    model = whisper.load_model(model_name)
    model.eval()
    return model


def load_distil_whisper_model(model_name: str):
    """Load distil-whisper model from HuggingFace."""
    try:
        from transformers import WhisperForConditionalGeneration
    except ImportError:
        print("Error: transformers not installed.")
        print("Install with: pip install transformers")
        sys.exit(1)

    # Map short name to HuggingFace model ID
    hf_model_map = {
        "distil-small.en": "distil-whisper/distil-small.en",
        "distil-medium.en": "distil-whisper/distil-medium.en",
        "distil-large-v2": "distil-whisper/distil-large-v2",
        "distil-large-v3": "distil-whisper/distil-large-v3",
    }

    hf_model_id = hf_model_map.get(model_name, f"distil-whisper/{model_name}")
    print(f"Loading {hf_model_id} from HuggingFace...")
    model = WhisperForConditionalGeneration.from_pretrained(hf_model_id)
    model.eval()
    return model


def extract_encoder_weights(model, config: dict) -> Dict[str, np.ndarray]:
    """Extract encoder weights from Whisper model."""
    tensors = {}
    enc = model.encoder

    # Conv1D layers
    # PyTorch conv1d: [out_channels, in_channels, kernel_size]
    # Tomoul expects: [out_channels, in_channels, kernel_size]
    tensors["encoder.conv1.weight"] = enc.conv1.weight.detach().cpu().float().numpy()
    tensors["encoder.conv1.bias"] = enc.conv1.bias.detach().cpu().float().numpy()
    tensors["encoder.conv2.weight"] = enc.conv2.weight.detach().cpu().float().numpy()
    tensors["encoder.conv2.bias"] = enc.conv2.bias.detach().cpu().float().numpy()

    # Positional embedding [n_audio_ctx, n_audio_state]
    tensors["encoder.positional_embedding"] = enc.positional_embedding.detach().cpu().float().numpy()

    # Encoder blocks
    n_layers = config["n_audio_layer"]
    for i in range(n_layers):
        block = enc.blocks[i]
        prefix = f"encoder.blocks.{i}"

        # Self-attention LayerNorm
        tensors[f"{prefix}.attn_ln.weight"] = block.attn_ln.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn_ln.bias"] = block.attn_ln.bias.detach().cpu().float().numpy()

        # Self-attention Q, K, V, Out projections
        # Note: Whisper uses nn.Linear which has [out, in] weight shape
        # We transpose for Tomoul's [in, out] matmul convention
        tensors[f"{prefix}.attn.q_weight"] = block.attn.query.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.q_bias"] = block.attn.query.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.k_weight"] = block.attn.key.weight.detach().cpu().float().numpy().T
        # Whisper attention key has no bias
        tensors[f"{prefix}.attn.k_bias"] = np.zeros(config["n_audio_state"], dtype=np.float32)
        tensors[f"{prefix}.attn.v_weight"] = block.attn.value.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.v_bias"] = block.attn.value.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.o_weight"] = block.attn.out.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.o_bias"] = block.attn.out.bias.detach().cpu().float().numpy()

        # FFN LayerNorm
        tensors[f"{prefix}.mlp_ln.weight"] = block.mlp_ln.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp_ln.bias"] = block.mlp_ln.bias.detach().cpu().float().numpy()

        # FFN layers (transposed for Tomoul)
        tensors[f"{prefix}.mlp.fc1.weight"] = block.mlp[0].weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc1.bias"] = block.mlp[0].bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp.fc2.weight"] = block.mlp[2].weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc2.bias"] = block.mlp[2].bias.detach().cpu().float().numpy()

    # Final LayerNorm
    tensors["encoder.ln_post.weight"] = enc.ln_post.weight.detach().cpu().float().numpy()
    tensors["encoder.ln_post.bias"] = enc.ln_post.bias.detach().cpu().float().numpy()

    return tensors


def extract_decoder_weights(model, config: dict) -> Dict[str, np.ndarray]:
    """Extract decoder weights from Whisper model."""
    tensors = {}
    dec = model.decoder

    # Token embedding [n_vocab, n_text_state]
    tensors["decoder.token_embedding"] = dec.token_embedding.weight.detach().cpu().float().numpy()

    # Positional embedding [n_text_ctx, n_text_state]
    tensors["decoder.positional_embedding"] = dec.positional_embedding.detach().cpu().float().numpy()

    # Decoder blocks
    n_layers = config["n_text_layer"]
    for i in range(n_layers):
        block = dec.blocks[i]
        prefix = f"decoder.blocks.{i}"

        # Self-attention LayerNorm
        tensors[f"{prefix}.attn_ln.weight"] = block.attn_ln.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn_ln.bias"] = block.attn_ln.bias.detach().cpu().float().numpy()

        # Self-attention Q, K, V, Out projections (transposed)
        tensors[f"{prefix}.attn.q_weight"] = block.attn.query.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.q_bias"] = block.attn.query.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.k_weight"] = block.attn.key.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.k_bias"] = np.zeros(config["n_text_state"], dtype=np.float32)
        tensors[f"{prefix}.attn.v_weight"] = block.attn.value.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.v_bias"] = block.attn.value.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.o_weight"] = block.attn.out.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.o_bias"] = block.attn.out.bias.detach().cpu().float().numpy()

        # Cross-attention LayerNorm
        tensors[f"{prefix}.cross_attn_ln.weight"] = block.cross_attn_ln.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn_ln.bias"] = block.cross_attn_ln.bias.detach().cpu().float().numpy()

        # Cross-attention Q, K, V, Out projections (transposed)
        tensors[f"{prefix}.cross_attn.q_weight"] = block.cross_attn.query.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.q_bias"] = block.cross_attn.query.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn.k_weight"] = block.cross_attn.key.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.k_bias"] = np.zeros(config["n_text_state"], dtype=np.float32)
        tensors[f"{prefix}.cross_attn.v_weight"] = block.cross_attn.value.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.v_bias"] = block.cross_attn.value.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn.o_weight"] = block.cross_attn.out.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.o_bias"] = block.cross_attn.out.bias.detach().cpu().float().numpy()

        # FFN LayerNorm
        tensors[f"{prefix}.mlp_ln.weight"] = block.mlp_ln.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp_ln.bias"] = block.mlp_ln.bias.detach().cpu().float().numpy()

        # FFN layers (transposed)
        tensors[f"{prefix}.mlp.fc1.weight"] = block.mlp[0].weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc1.bias"] = block.mlp[0].bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp.fc2.weight"] = block.mlp[2].weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc2.bias"] = block.mlp[2].bias.detach().cpu().float().numpy()

    # Final LayerNorm
    tensors["decoder.ln.weight"] = dec.ln.weight.detach().cpu().float().numpy()
    tensors["decoder.ln.bias"] = dec.ln.bias.detach().cpu().float().numpy()

    return tensors


def extract_hf_encoder_weights(model, config: dict) -> Dict[str, np.ndarray]:
    """Extract encoder weights from HuggingFace Whisper model (distil-whisper)."""
    tensors = {}
    enc = model.model.encoder

    # Conv1D layers
    tensors["encoder.conv1.weight"] = enc.conv1.weight.detach().cpu().float().numpy()
    tensors["encoder.conv1.bias"] = enc.conv1.bias.detach().cpu().float().numpy()
    tensors["encoder.conv2.weight"] = enc.conv2.weight.detach().cpu().float().numpy()
    tensors["encoder.conv2.bias"] = enc.conv2.bias.detach().cpu().float().numpy()

    # Positional embedding - HF stores as embed_positions.weight
    tensors["encoder.positional_embedding"] = enc.embed_positions.weight.detach().cpu().float().numpy()

    # Encoder layers
    n_layers = config["n_audio_layer"]
    for i in range(n_layers):
        layer = enc.layers[i]
        prefix = f"encoder.blocks.{i}"

        # Self-attention LayerNorm (HF: self_attn_layer_norm)
        tensors[f"{prefix}.attn_ln.weight"] = layer.self_attn_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn_ln.bias"] = layer.self_attn_layer_norm.bias.detach().cpu().float().numpy()

        # Self-attention (HF: self_attn.q_proj, k_proj, v_proj, out_proj)
        tensors[f"{prefix}.attn.q_weight"] = layer.self_attn.q_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.q_bias"] = layer.self_attn.q_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.k_weight"] = layer.self_attn.k_proj.weight.detach().cpu().float().numpy().T
        # HF Whisper k_proj has no bias
        tensors[f"{prefix}.attn.k_bias"] = np.zeros(config["n_audio_state"], dtype=np.float32)
        tensors[f"{prefix}.attn.v_weight"] = layer.self_attn.v_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.v_bias"] = layer.self_attn.v_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.o_weight"] = layer.self_attn.out_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.o_bias"] = layer.self_attn.out_proj.bias.detach().cpu().float().numpy()

        # FFN LayerNorm (HF: final_layer_norm)
        tensors[f"{prefix}.mlp_ln.weight"] = layer.final_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp_ln.bias"] = layer.final_layer_norm.bias.detach().cpu().float().numpy()

        # FFN layers (HF: fc1, fc2)
        tensors[f"{prefix}.mlp.fc1.weight"] = layer.fc1.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc1.bias"] = layer.fc1.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp.fc2.weight"] = layer.fc2.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc2.bias"] = layer.fc2.bias.detach().cpu().float().numpy()

    # Final LayerNorm (HF: layer_norm)
    tensors["encoder.ln_post.weight"] = enc.layer_norm.weight.detach().cpu().float().numpy()
    tensors["encoder.ln_post.bias"] = enc.layer_norm.bias.detach().cpu().float().numpy()

    return tensors


def extract_hf_decoder_weights(model, config: dict) -> Dict[str, np.ndarray]:
    """Extract decoder weights from HuggingFace Whisper model (distil-whisper)."""
    tensors = {}
    dec = model.model.decoder

    # Token embedding (HF: embed_tokens)
    tensors["decoder.token_embedding"] = dec.embed_tokens.weight.detach().cpu().float().numpy()

    # Positional embedding (HF: embed_positions)
    tensors["decoder.positional_embedding"] = dec.embed_positions.weight.detach().cpu().float().numpy()

    # Decoder layers
    n_layers = config["n_text_layer"]
    for i in range(n_layers):
        layer = dec.layers[i]
        prefix = f"decoder.blocks.{i}"

        # Self-attention LayerNorm
        tensors[f"{prefix}.attn_ln.weight"] = layer.self_attn_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn_ln.bias"] = layer.self_attn_layer_norm.bias.detach().cpu().float().numpy()

        # Self-attention
        tensors[f"{prefix}.attn.q_weight"] = layer.self_attn.q_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.q_bias"] = layer.self_attn.q_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.k_weight"] = layer.self_attn.k_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.k_bias"] = np.zeros(config["n_text_state"], dtype=np.float32)
        tensors[f"{prefix}.attn.v_weight"] = layer.self_attn.v_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.v_bias"] = layer.self_attn.v_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.o_weight"] = layer.self_attn.out_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.o_bias"] = layer.self_attn.out_proj.bias.detach().cpu().float().numpy()

        # Cross-attention LayerNorm (HF: encoder_attn_layer_norm)
        tensors[f"{prefix}.cross_attn_ln.weight"] = layer.encoder_attn_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn_ln.bias"] = layer.encoder_attn_layer_norm.bias.detach().cpu().float().numpy()

        # Cross-attention (HF: encoder_attn)
        tensors[f"{prefix}.cross_attn.q_weight"] = layer.encoder_attn.q_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.q_bias"] = layer.encoder_attn.q_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn.k_weight"] = layer.encoder_attn.k_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.k_bias"] = np.zeros(config["n_text_state"], dtype=np.float32)
        tensors[f"{prefix}.cross_attn.v_weight"] = layer.encoder_attn.v_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.v_bias"] = layer.encoder_attn.v_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn.o_weight"] = layer.encoder_attn.out_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.o_bias"] = layer.encoder_attn.out_proj.bias.detach().cpu().float().numpy()

        # FFN LayerNorm (HF: final_layer_norm)
        tensors[f"{prefix}.mlp_ln.weight"] = layer.final_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp_ln.bias"] = layer.final_layer_norm.bias.detach().cpu().float().numpy()

        # FFN layers
        tensors[f"{prefix}.mlp.fc1.weight"] = layer.fc1.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc1.bias"] = layer.fc1.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp.fc2.weight"] = layer.fc2.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc2.bias"] = layer.fc2.bias.detach().cpu().float().numpy()

    # Final LayerNorm (HF: layer_norm)
    tensors["decoder.ln.weight"] = dec.layer_norm.weight.detach().cpu().float().numpy()
    tensors["decoder.ln.bias"] = dec.layer_norm.bias.detach().cpu().float().numpy()

    return tensors


def export_whisper(
    model_name: str,
    output_dir: str,
    quant_format: str = "f32",
) -> Path:
    """Export Whisper model to Tomoul .tl format."""
    if model_name not in WHISPER_CONFIGS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Supported models: {', '.join(WHISPER_CONFIGS.keys())}")
        sys.exit(1)

    config = WHISPER_CONFIGS[model_name]
    model = load_whisper_model(model_name)

    # Extract weights - use HF extractors for distil-whisper models
    if is_distil_whisper(model_name):
        print("Extracting encoder weights (HuggingFace format)...")
        encoder_tensors = extract_hf_encoder_weights(model, config)

        print("Extracting decoder weights (HuggingFace format)...")
        decoder_tensors = extract_hf_decoder_weights(model, config)
    else:
        print("Extracting encoder weights...")
        encoder_tensors = extract_encoder_weights(model, config)

        print("Extracting decoder weights...")
        decoder_tensors = extract_decoder_weights(model, config)

    # Combine all tensors
    all_tensors = {**encoder_tensors, **decoder_tensors}

    print(f"\nTotal tensors: {len(all_tensors)}")

    # Calculate total parameters
    total_params = sum(t.size for t in all_tensors.values())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Determine quantization format
    quant_format_map = {
        "f32": QuantFormat.F32,
        "q8_0": QuantFormat.Q8_0,
        "q8": QuantFormat.Q8_0,
        "q4_0": QuantFormat.Q4_0,
        "q4": QuantFormat.Q4_0,
    }

    if quant_format.lower() not in quant_format_map:
        print(f"Error: Unknown quantization format '{quant_format}'")
        print(f"Supported formats: {', '.join(quant_format_map.keys())}")
        sys.exit(1)

    qf = quant_format_map[quant_format.lower()]

    # Build output filename
    output_dir = Path(output_dir)
    suffix = "" if qf == QuantFormat.F32 else f"_{quant_format.lower()}"
    output_path = output_dir / f"whisper_{model_name}{suffix}.tl"

    # Export
    print(f"\nExporting to {output_path}...")
    export_tensors(all_tensors, str(output_path), qf)

    return output_path


def export_validation_fixture(model_name: str, output_dir: str):
    """Export a validation fixture for testing."""
    try:
        import whisper
    except ImportError:
        print("Error: OpenAI Whisper not installed.")
        sys.exit(1)

    config = WHISPER_CONFIGS[model_name]
    output_dir = Path(output_dir)

    # Create a dummy Mel spectrogram for testing
    # In practice, this would be computed from real audio
    print("Creating validation fixture...")

    # Generate random Mel spectrogram [80, 3000]
    mel = np.random.randn(config["n_mels"], 3000).astype(np.float32) * 0.1

    fixture_path = output_dir / f"whisper_{model_name}_validation.tl"
    export_tensors(
        {"mel": mel},
        str(fixture_path),
        QuantFormat.F32,
    )
    print(f"Validation fixture: {fixture_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Whisper models to Tomoul .tl format"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="tiny",
        help="Model variant: tiny, base, small, medium, large, large-v3-turbo, distil-small.en (default: tiny)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./models",
        help="Output directory (default: ./models)"
    )
    parser.add_argument(
        "-q", "--quantization",
        type=str,
        default="f32",
        help="Quantization format: f32, q8_0, q4_0 (default: f32)"
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Also export a validation fixture"
    )

    args = parser.parse_args()

    output_path = export_whisper(args.model, args.output, args.quantization)

    if args.fixture:
        export_validation_fixture(args.model, args.output)

    print(f"\nDone! Model exported to: {output_path}")


if __name__ == "__main__":
    main()
