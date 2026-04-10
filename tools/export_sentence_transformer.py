#!/usr/bin/env python3
"""
Export sentence-transformers/all-MiniLM-L6-v2 to .tl format for Zig inference.

Exports:
  - Model weights (float32 and optionally Q8 quantized) in .tl format
  - WordPiece vocabulary (vocab.txt)
  - Reference embeddings for 5 test sentences (JSON)

Usage:
    python tools/export_sentence_transformer.py -o artifacts/
    python tools/export_sentence_transformer.py -o artifacts/ -q q8_0
"""

import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from pathlib import Path
import argparse

# Import shared format utilities
from tl_format import QuantFormat, export_tensors, add_quantize_args, get_quant_format

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "I had pizza for lunch yesterday",
    "The capital of France is Paris",
    "Quantum computing uses qubits instead of classical bits",
]


def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]  # First element: all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def export_model(output_dir: Path, quant_format: int = QuantFormat.F32):
    """Export all-MiniLM-L6-v2 weights to .tl format."""
    print(f"Loading model: {MODEL_NAME}")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max position embeddings: {config.max_position_embeddings}")

    # Collect all tensors with BERT naming
    tensors = {}
    # Track which weight matrices need pre-transposing for efficient Q8 loading.
    # Without this, Zig must dequantize → transpose → re-quantize, causing double
    # quantization error. Pre-transposing lets Zig load Q8 weights directly.
    pretranspose = set()
    print("\nModel weights:")
    for name, param in model.named_parameters():
        tensor = param.detach().cpu().float().contiguous().numpy()
        # Pre-transpose 2D weight matrices in encoder layers (not embeddings)
        if quant_format != QuantFormat.F32 and tensor.ndim == 2 and 'encoder.layer' in name and name.endswith('.weight'):
            tensor = np.ascontiguousarray(tensor.T)
            pretranspose.add(name)
        tensors[name] = tensor
        suffix = " (pre-transposed)" if name in pretranspose else ""
        print(f"  {name}: {list(tensor.shape)}{suffix}")

    # Export model weights
    output_dir.mkdir(parents=True, exist_ok=True)
    format_suffixes = {QuantFormat.Q8_K: "_q8k", QuantFormat.F16: "_f16"}
    format_suffix = format_suffixes.get(quant_format, "")
    model_path = output_dir / f"all_minilm_l6_v2{format_suffix}.tl"
    export_tensors(tensors, str(model_path), quant_format=quant_format, verify=True)

    # Export vocabulary (WordPiece)
    vocab_path = output_dir / "all_minilm_l6_v2_vocab.txt"
    export_vocab(tokenizer, vocab_path)

    # Generate and export reference embeddings
    ref_path = output_dir / "minilm_reference_embeddings.json"
    export_reference_embeddings(model, tokenizer, ref_path)

    print(f"\nExport complete!")
    print(f"  Model:      {model_path}")
    print(f"  Vocab:      {vocab_path}")
    print(f"  Reference:  {ref_path}")

    return model_path, vocab_path, ref_path


def export_vocab(tokenizer, output_path: Path):
    """Export WordPiece vocabulary (one token per line, ordered by ID)."""
    print(f"\nExporting vocabulary to {output_path}")

    # Sort by token ID
    sorted_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1])

    with open(output_path, "w", encoding="utf-8") as f:
        for token, _idx in sorted_vocab:
            # Escape special characters
            token_escaped = token.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
            f.write(f"{token_escaped}\n")

    print(f"  Exported {len(sorted_vocab)} tokens")


def export_reference_embeddings(model, tokenizer, output_path: Path):
    """Generate reference embeddings for test sentences using HuggingFace."""
    print(f"\nGenerating reference embeddings for {len(TEST_SENTENCES)} sentences...")

    encoded = tokenizer(TEST_SENTENCES, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded)

    # Mean pooling
    embeddings = mean_pooling(model_output, encoded["attention_mask"])

    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    embeddings_np = embeddings.cpu().numpy()

    # Also export tokenizer outputs for each sentence (for tokenizer validation)
    tokenizer_outputs = []
    for sentence in TEST_SENTENCES:
        enc = tokenizer(sentence, add_special_tokens=True, return_tensors="pt")
        token_ids = enc["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        tokenizer_outputs.append({
            "text": sentence,
            "token_ids": token_ids,
            "tokens": tokens,
        })

    ref_data = {
        "model": MODEL_NAME,
        "dimensions": int(embeddings_np.shape[1]),
        "sentences": TEST_SENTENCES,
        "embeddings": embeddings_np.tolist(),
        "tokenizer_reference": tokenizer_outputs,
    }

    with open(output_path, "w") as f:
        json.dump(ref_data, f, indent=2)

    print(f"  Saved {embeddings_np.shape[0]}x{embeddings_np.shape[1]} embeddings")

    # Print cosine similarities for sanity check
    print("\n  Pairwise cosine similarities:")
    for i in range(len(TEST_SENTENCES)):
        for j in range(i + 1, len(TEST_SENTENCES)):
            cos_sim = np.dot(embeddings_np[i], embeddings_np[j])
            print(f"    [{i}] vs [{j}]: {cos_sim:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Export all-MiniLM-L6-v2 to .tl format for Zig inference",
    )
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output directory")
    add_quantize_args(parser)

    args = parser.parse_args()
    quant_format = get_quant_format(args.quantize)
    export_model(Path(args.output_dir), quant_format)


if __name__ == "__main__":
    main()
