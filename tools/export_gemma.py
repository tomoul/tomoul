#!/usr/bin/env python3
"""
Export Google Gemma models to .tl format for Zig inference.

Exports:
  - Model weights (float32 or quantized) in .tl format
  - SentencePiece tokenizer as binary vocab file
  - Reference logits for validation (JSON)

Usage:
    python tools/export_gemma.py -o artifacts/ --variant 2b
    python tools/export_gemma.py -o artifacts/ --variant 2b -q q8_k
    python tools/export_gemma.py -o artifacts/ --variant 7b -q q8_k

Requires: pip install transformers torch sentencepiece protobuf
Note: Gemma requires HuggingFace access token (huggingface-cli login).
"""

import json
import struct
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
import argparse

from tl_format import QuantFormat, export_tensors, add_quantize_args, get_quant_format

# HuggingFace model IDs
VARIANT_MAP = {
    "2b": "google/gemma-2b",
    "7b": "google/gemma-7b",
    "9b": "google/gemma-2-9b",
    "27b": "google/gemma-2-27b",
}

# Test prompts for reference logits
TEST_PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "def fibonacci(n):",
]


def export_model(output_dir: Path, variant: str, quant_format: int = QuantFormat.F32):
    """Export Gemma model weights to .tl format."""
    model_id = VARIANT_MAP[variant]
    print(f"Loading model: {model_id}")

    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Num KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max position: {config.max_position_embeddings}")

    # Collect and rename tensors to match Zig weight loader naming
    tensors = {}
    pretranspose = set()

    state_dict = model.state_dict()
    print(f"\nCollecting {len(state_dict)} tensors:")

    for name, param in state_dict.items():
        tensor = param.detach().cpu().float().contiguous().numpy()

        # Map HuggingFace names to our naming convention
        tl_name = map_weight_name(name, variant)
        if tl_name is None:
            print(f"  SKIP: {name}")
            continue

        # Pre-transpose 2D weight matrices for efficient matmul (row-major → Zig layout)
        # In HF: weight is [out_features, in_features], we need [in_features, out_features]
        if tensor.ndim == 2 and ".weight" in name and "embed" not in name:
            tensor = np.ascontiguousarray(tensor.T)
            pretranspose.add(tl_name)

        tensors[tl_name] = tensor
        suffix = " (pre-transposed)" if tl_name in pretranspose else ""
        print(f"  {name} → {tl_name}: {list(tensor.shape)}{suffix}")

    # Export weights
    output_dir.mkdir(parents=True, exist_ok=True)
    format_suffixes = {
        QuantFormat.F32: "",
        QuantFormat.Q8_0: "_q8",
        QuantFormat.Q4_0: "_q4",
        QuantFormat.Q8_K: "_q8k",
        QuantFormat.F16: "_f16",
    }
    suffix = format_suffixes.get(quant_format, "")
    model_path = output_dir / f"gemma_{variant}{suffix}.tl"
    export_tensors(tensors, str(model_path), quant_format=quant_format, verify=True)

    # Export tokenizer
    tokenizer_path = output_dir / f"gemma_{variant}_vocab.bin"
    export_tokenizer(model_id, tokenizer_path)

    # Export reference logits
    ref_path = output_dir / f"gemma_{variant}_reference.json"
    export_reference_logits(model, model_id, ref_path)

    print(f"\nExport complete!")
    print(f"  Weights:    {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Tokenizer:  {tokenizer_path} ({tokenizer_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Reference:  {ref_path}")


def map_weight_name(hf_name: str, variant: str) -> str | None:
    """Map HuggingFace weight name to Tomoul .tl tensor name."""
    # model.embed_tokens.weight → token_embedding.weight
    if hf_name == "model.embed_tokens.weight":
        return "token_embedding.weight"

    # model.norm.weight → norm.weight
    if hf_name == "model.norm.weight":
        return "norm.weight"

    # lm_head.weight → skip (tied to embed_tokens)
    if hf_name == "lm_head.weight":
        return None

    # model.layers.{i}.* → layers.{i}.*
    if hf_name.startswith("model.layers."):
        rest = hf_name[len("model.layers."):]
        # Extract layer number
        dot_pos = rest.index(".")
        layer_num = rest[:dot_pos]
        layer_rest = rest[dot_pos + 1:]

        # Attention projections
        # self_attn.q_proj.weight → self_attn.q_proj.weight
        # self_attn.k_proj.weight → self_attn.k_proj.weight
        # self_attn.v_proj.weight → self_attn.v_proj.weight
        # self_attn.o_proj.weight → self_attn.o_proj.weight
        if layer_rest.startswith("self_attn."):
            return f"layers.{layer_num}.{layer_rest}"

        # MLP projections
        # mlp.gate_proj.weight → mlp.gate_proj.weight
        # mlp.up_proj.weight → mlp.up_proj.weight
        # mlp.down_proj.weight → mlp.down_proj.weight
        if layer_rest.startswith("mlp."):
            return f"layers.{layer_num}.{layer_rest}"

        # Layer norms
        # input_layernorm.weight → input_layernorm.weight
        # post_attention_layernorm.weight → post_attention_layernorm.weight
        if "layernorm" in layer_rest or "norm" in layer_rest:
            return f"layers.{layer_num}.{layer_rest}"

        return f"layers.{layer_num}.{layer_rest}"

    print(f"  WARNING: unmapped weight: {hf_name}")
    return hf_name


def export_tokenizer(model_id: str, output_path: Path):
    """
    Export SentencePiece tokenizer as binary vocab file.

    Binary format:
      u32: vocab_size
      u32: max_token_len
      For each token:
        f32: score (merge priority)
        u32: token_len
        [token_len]u8: token bytes
    """
    print(f"\nExporting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    vocab_size = tokenizer.vocab_size
    # Get the underlying SentencePiece model
    sp_model = tokenizer.sp_model

    max_token_len = 0
    tokens = []

    for token_id in range(vocab_size):
        piece = sp_model.id_to_piece(token_id)
        score = sp_model.get_score(token_id)
        piece_bytes = piece.encode("utf-8")
        max_token_len = max(max_token_len, len(piece_bytes))
        tokens.append((piece_bytes, score))

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", max_token_len))

        for piece_bytes, score in tokens:
            f.write(struct.pack("<f", score))
            f.write(struct.pack("<I", len(piece_bytes)))
            f.write(piece_bytes)

    print(f"  Vocab size: {vocab_size}")
    print(f"  Max token length: {max_token_len}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def export_reference_logits(model, model_id: str, output_path: Path):
    """Export reference logits for validation."""
    print(f"\nGenerating reference logits...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    references = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Save last position's top-10 logits and the full logit stats
        last_logits = logits[0, -1, :].numpy()
        top_indices = np.argsort(last_logits)[-10:][::-1]
        top_values = last_logits[top_indices].tolist()
        top_tokens = [tokenizer.decode([int(idx)]) for idx in top_indices]

        references.append({
            "prompt": prompt,
            "input_ids": input_ids[0].tolist(),
            "logits_mean": float(np.mean(last_logits)),
            "logits_std": float(np.std(last_logits)),
            "logits_max": float(np.max(last_logits)),
            "logits_min": float(np.min(last_logits)),
            "top_10": [
                {"token": tok, "token_id": int(idx), "logit": float(val)}
                for tok, idx, val in zip(top_tokens, top_indices.tolist(), top_values)
            ],
        })
        print(f"  '{prompt}' → top token: '{top_tokens[0]}' ({top_values[0]:.2f})")

    with open(output_path, "w") as f:
        json.dump(references, f, indent=2)


def add_quantize_args(parser):
    """Add quantization arguments to an argument parser."""
    parser.add_argument(
        "-q", "--quantize",
        choices=["f32", "q8_0", "q4_0", "q8_k", "f16"],
        default="f32",
        help="Quantization format (default: f32)",
    )


def get_quant_format(args) -> int:
    """Get quantization format from parsed args."""
    fmt_map = {
        "f32": QuantFormat.F32,
        "q8_0": QuantFormat.Q8_0,
        "q4_0": QuantFormat.Q4_0,
        "q8_k": QuantFormat.Q8_K,
        "f16": QuantFormat.F16,
    }
    return fmt_map[args.quantize]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Gemma to .tl format")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--variant",
        choices=["2b", "7b", "9b", "27b"],
        default="2b",
        help="Gemma variant (default: 2b)",
    )
    add_quantize_args(parser)
    args = parser.parse_args()

    quant_format = get_quant_format(args)
    export_model(args.output_dir, args.variant, quant_format)
