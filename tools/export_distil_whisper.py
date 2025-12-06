#!/usr/bin/env python3
"""
Export HuggingFace distil-whisper models to .tl format for Tomoul.

Usage:
    python export_distil_whisper.py -m distil-whisper/distil-small.en -o ./models/
    python export_distil_whisper.py -m distil-whisper/distil-medium.en -o ./models/ -q q8_0

The exporter:
1. Downloads the distil-whisper model from HuggingFace
2. Extracts encoder and decoder weights
3. Pre-transposes weights for optimal SIMD matmul
4. Exports to .tl format with optional quantization
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# Import shared tl_format utilities
from tl_format import export_tensors, QuantFormat


def load_distil_whisper_model(model_id: str):
    """Load distil-whisper model from HuggingFace."""
    try:
        from transformers import WhisperForConditionalGeneration
    except ImportError:
        print("Error: transformers not installed.")
        print("Install with: pip install transformers")
        sys.exit(1)

    print(f"Loading {model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()
    return model


def get_config_from_model(model) -> dict:
    """Extract configuration from HuggingFace model."""
    cfg = model.config
    return {
        "n_mels": cfg.num_mel_bins,
        "n_audio_ctx": cfg.max_source_positions,
        "n_audio_state": cfg.d_model,
        "n_audio_head": cfg.encoder_attention_heads,
        "n_audio_layer": cfg.encoder_layers,
        "n_vocab": cfg.vocab_size,
        "n_text_ctx": cfg.max_target_positions,
        "n_text_state": cfg.d_model,
        "n_text_head": cfg.decoder_attention_heads,
        "n_text_layer": cfg.decoder_layers,
    }


def extract_encoder_weights(model, config: dict) -> Dict[str, np.ndarray]:
    """Extract encoder weights from HuggingFace Whisper model."""
    tensors = {}
    enc = model.model.encoder

    # Conv1D layers
    # HuggingFace conv1d: [out_channels, in_channels, kernel_size]
    tensors["encoder.conv1.weight"] = enc.conv1.weight.detach().cpu().float().numpy()
    tensors["encoder.conv1.bias"] = enc.conv1.bias.detach().cpu().float().numpy()
    tensors["encoder.conv2.weight"] = enc.conv2.weight.detach().cpu().float().numpy()
    tensors["encoder.conv2.bias"] = enc.conv2.bias.detach().cpu().float().numpy()

    # Positional embedding [n_audio_ctx, n_audio_state]
    # HuggingFace uses embed_positions.weight
    tensors["encoder.positional_embedding"] = enc.embed_positions.weight.detach().cpu().float().numpy()

    # Encoder blocks
    n_layers = config["n_audio_layer"]
    for i in range(n_layers):
        layer = enc.layers[i]
        prefix = f"encoder.blocks.{i}"

        # Self-attention LayerNorm (layer_norm before attention)
        tensors[f"{prefix}.attn_ln.weight"] = layer.self_attn_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn_ln.bias"] = layer.self_attn_layer_norm.bias.detach().cpu().float().numpy()

        # Self-attention Q, K, V, Out projections
        # HuggingFace uses nn.Linear which has [out, in] weight shape
        # We transpose for Tomoul's [in, out] matmul convention
        tensors[f"{prefix}.attn.q_weight"] = layer.self_attn.q_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.q_bias"] = layer.self_attn.q_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.k_weight"] = layer.self_attn.k_proj.weight.detach().cpu().float().numpy().T
        # HuggingFace Whisper k_proj has no bias
        if layer.self_attn.k_proj.bias is not None:
            tensors[f"{prefix}.attn.k_bias"] = layer.self_attn.k_proj.bias.detach().cpu().float().numpy()
        else:
            tensors[f"{prefix}.attn.k_bias"] = np.zeros(config["n_audio_state"], dtype=np.float32)
        tensors[f"{prefix}.attn.v_weight"] = layer.self_attn.v_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.v_bias"] = layer.self_attn.v_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.o_weight"] = layer.self_attn.out_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.o_bias"] = layer.self_attn.out_proj.bias.detach().cpu().float().numpy()

        # FFN LayerNorm (final_layer_norm)
        tensors[f"{prefix}.mlp_ln.weight"] = layer.final_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp_ln.bias"] = layer.final_layer_norm.bias.detach().cpu().float().numpy()

        # FFN layers (transposed for Tomoul)
        tensors[f"{prefix}.mlp.fc1.weight"] = layer.fc1.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc1.bias"] = layer.fc1.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp.fc2.weight"] = layer.fc2.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc2.bias"] = layer.fc2.bias.detach().cpu().float().numpy()

    # Final LayerNorm
    tensors["encoder.ln_post.weight"] = enc.layer_norm.weight.detach().cpu().float().numpy()
    tensors["encoder.ln_post.bias"] = enc.layer_norm.bias.detach().cpu().float().numpy()

    return tensors


def extract_decoder_weights(model, config: dict) -> Dict[str, np.ndarray]:
    """Extract decoder weights from HuggingFace Whisper model."""
    tensors = {}
    dec = model.model.decoder

    # Token embedding [n_vocab, n_text_state]
    tensors["decoder.token_embedding"] = dec.embed_tokens.weight.detach().cpu().float().numpy()

    # Positional embedding [n_text_ctx, n_text_state]
    tensors["decoder.positional_embedding"] = dec.embed_positions.weight.detach().cpu().float().numpy()

    # Decoder blocks
    n_layers = config["n_text_layer"]
    for i in range(n_layers):
        layer = dec.layers[i]
        prefix = f"decoder.blocks.{i}"

        # Self-attention LayerNorm
        tensors[f"{prefix}.attn_ln.weight"] = layer.self_attn_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn_ln.bias"] = layer.self_attn_layer_norm.bias.detach().cpu().float().numpy()

        # Self-attention Q, K, V, Out projections (transposed)
        tensors[f"{prefix}.attn.q_weight"] = layer.self_attn.q_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.q_bias"] = layer.self_attn.q_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.k_weight"] = layer.self_attn.k_proj.weight.detach().cpu().float().numpy().T
        if layer.self_attn.k_proj.bias is not None:
            tensors[f"{prefix}.attn.k_bias"] = layer.self_attn.k_proj.bias.detach().cpu().float().numpy()
        else:
            tensors[f"{prefix}.attn.k_bias"] = np.zeros(config["n_text_state"], dtype=np.float32)
        tensors[f"{prefix}.attn.v_weight"] = layer.self_attn.v_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.v_bias"] = layer.self_attn.v_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.attn.o_weight"] = layer.self_attn.out_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.attn.o_bias"] = layer.self_attn.out_proj.bias.detach().cpu().float().numpy()

        # Cross-attention LayerNorm
        tensors[f"{prefix}.cross_attn_ln.weight"] = layer.encoder_attn_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn_ln.bias"] = layer.encoder_attn_layer_norm.bias.detach().cpu().float().numpy()

        # Cross-attention Q, K, V, Out projections (transposed)
        tensors[f"{prefix}.cross_attn.q_weight"] = layer.encoder_attn.q_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.q_bias"] = layer.encoder_attn.q_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn.k_weight"] = layer.encoder_attn.k_proj.weight.detach().cpu().float().numpy().T
        if layer.encoder_attn.k_proj.bias is not None:
            tensors[f"{prefix}.cross_attn.k_bias"] = layer.encoder_attn.k_proj.bias.detach().cpu().float().numpy()
        else:
            tensors[f"{prefix}.cross_attn.k_bias"] = np.zeros(config["n_text_state"], dtype=np.float32)
        tensors[f"{prefix}.cross_attn.v_weight"] = layer.encoder_attn.v_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.v_bias"] = layer.encoder_attn.v_proj.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.cross_attn.o_weight"] = layer.encoder_attn.out_proj.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.cross_attn.o_bias"] = layer.encoder_attn.out_proj.bias.detach().cpu().float().numpy()

        # FFN LayerNorm
        tensors[f"{prefix}.mlp_ln.weight"] = layer.final_layer_norm.weight.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp_ln.bias"] = layer.final_layer_norm.bias.detach().cpu().float().numpy()

        # FFN layers (transposed)
        tensors[f"{prefix}.mlp.fc1.weight"] = layer.fc1.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc1.bias"] = layer.fc1.bias.detach().cpu().float().numpy()
        tensors[f"{prefix}.mlp.fc2.weight"] = layer.fc2.weight.detach().cpu().float().numpy().T
        tensors[f"{prefix}.mlp.fc2.bias"] = layer.fc2.bias.detach().cpu().float().numpy()

    # Final LayerNorm
    tensors["decoder.ln.weight"] = dec.layer_norm.weight.detach().cpu().float().numpy()
    tensors["decoder.ln.bias"] = dec.layer_norm.bias.detach().cpu().float().numpy()

    return tensors


def export_distil_whisper(
    model_id: str,
    output_dir: str,
    quant_format: str = "f32",
) -> Path:
    """Export distil-whisper model to Tomoul .tl format."""
    model = load_distil_whisper_model(model_id)
    config = get_config_from_model(model)

    print(f"\nModel configuration:")
    print(f"  Encoder: {config['n_audio_layer']} layers, {config['n_audio_state']} dim, {config['n_audio_head']} heads")
    print(f"  Decoder: {config['n_text_layer']} layers, {config['n_text_state']} dim, {config['n_text_head']} heads")
    print(f"  Vocab: {config['n_vocab']}, Mels: {config['n_mels']}")

    # Extract weights
    print("\nExtracting encoder weights...")
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

    # Build output filename from model_id
    # distil-whisper/distil-small.en -> distil-small.en
    model_name = model_id.split("/")[-1]
    output_dir = Path(output_dir)
    suffix = "" if qf == QuantFormat.F32 else f"_{quant_format.lower()}"
    output_path = output_dir / f"{model_name}{suffix}.tl"

    # Export
    print(f"\nExporting to {output_path}...")
    export_tensors(all_tensors, str(output_path), qf)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace distil-whisper model to Tomoul .tl format"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="distil-whisper/distil-small.en",
        help="HuggingFace model ID (default: distil-whisper/distil-small.en)"
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

    args = parser.parse_args()

    output_path = export_distil_whisper(args.model, args.output, args.quantization)

    print(f"\nDone! Model exported to: {output_path}")


if __name__ == "__main__":
    main()
