#!/usr/bin/env python3
"""
Export Qwen3.5-0.8B weights from HuggingFace to Tomoul .tl format.

Usage:
  python export_qwen3_5.py --model Qwen/Qwen3.5-0.8B --output artifacts/qwen3_5_0.8b.tl
  python export_qwen3_5.py --model Qwen/Qwen3.5-0.8B --output artifacts/qwen3_5_0.8b_q8k.tl --quant q8k
  python export_qwen3_5.py --model Qwen/Qwen3.5-0.8B --tokenizer-output artifacts/qwen3_5_vocab.bin

Requirements:
  pip install transformers safetensors torch
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Add tools directory for tl_format
sys.path.insert(0, str(Path(__file__).parent))
from tl_format import export_tensors, QuantFormat


# Layer layout: [L L L A] × 6
FULL_ATTENTION_INTERVAL = 4
NUM_LAYERS = 24


def is_full_attention(layer_idx: int) -> bool:
    """Check if layer uses full attention (vs DeltaNet)."""
    return (layer_idx + 1) % FULL_ATTENTION_INTERVAL == 0


def load_model_weights(model_path: str) -> dict:
    """Load weights from HuggingFace safetensors."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("Error: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    import glob

    weight_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    if not weight_files:
        # Try downloading from HuggingFace
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", trust_remote_code=True
        )
        state_dict = {k: v.float().numpy() for k, v in model.state_dict().items()}
        return state_dict

    state_dict = {}
    for f in weight_files:
        with safe_open(f, framework="numpy") as sf:
            for key in sf.keys():
                state_dict[key] = sf.get_tensor(key).astype(np.float32)

    return state_dict


def map_tensor_name(hf_name: str, layer_idx: int | None = None) -> str | None:
    """Map HuggingFace tensor name to Tomoul .tl tensor name."""

    # Global tensors
    if hf_name == "model.embed_tokens.weight":
        return "embed_tokens.weight"
    if hf_name == "model.norm.weight":
        return "norm.weight"
    if hf_name == "lm_head.weight":
        return None  # Tied to embed_tokens

    # Layer tensors
    if not hf_name.startswith("model.layers."):
        return None  # Skip unknown tensors

    parts = hf_name.split(".")
    layer_idx = int(parts[2])
    remainder = ".".join(parts[3:])

    # Common weights (all layers)
    common_map = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attn_layernorm.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
    }

    if remainder in common_map:
        return f"layers.{layer_idx}.{common_map[remainder]}"

    # Full attention layers
    if is_full_attention(layer_idx):
        attn_map = {
            "self_attn.q_proj.weight": "self_attn.q_proj.weight",
            "self_attn.k_proj.weight": "self_attn.k_proj.weight",
            "self_attn.v_proj.weight": "self_attn.v_proj.weight",
            "self_attn.o_proj.weight": "self_attn.o_proj.weight",
            "self_attn.q_norm.weight": "self_attn.q_norm.weight",
            "self_attn.k_norm.weight": "self_attn.k_norm.weight",
        }
        if remainder in attn_map:
            return f"layers.{layer_idx}.{attn_map[remainder]}"
    else:
        # DeltaNet layers
        deltanet_map = {
            "linear_attn.in_proj_qkv.weight": "deltanet.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight": "deltanet.in_proj_z.weight",
            "linear_attn.in_proj_b.weight": "deltanet.in_proj_b.weight",
            "linear_attn.in_proj_a.weight": "deltanet.in_proj_a.weight",
            "linear_attn.conv1d.weight": "deltanet.conv1d.weight",
            "linear_attn.conv1d.bias": "deltanet.conv1d.bias",
            "linear_attn.A_log": "deltanet.A_log",
            "linear_attn.dt_bias": "deltanet.dt_bias",
            "linear_attn.out_proj.weight": "deltanet.out_proj.weight",
            "linear_attn.norm.weight": "deltanet.norm.weight",
        }
        if remainder in deltanet_map:
            return f"layers.{layer_idx}.{deltanet_map[remainder]}"

    print(f"  [SKIP] Unknown tensor: {hf_name}")
    return None


def process_weights(state_dict: dict) -> dict:
    """Process and map all weights for export."""
    tensors = {}
    skipped = []

    for hf_name, weight in state_dict.items():
        tl_name = map_tensor_name(hf_name)
        if tl_name is None:
            skipped.append(hf_name)
            continue

        weight = np.asarray(weight, dtype=np.float32)

        # Handle RMSNorm (1+w): add 1.0 to all Qwen3_5RMSNorm weights
        # (Gemma-style: initialized to zeros, forward uses (1+weight)*rms_norm(x))
        # Matches: input_layernorm.weight, post_attn_layernorm.weight, norm.weight,
        #          self_attn.q_norm.weight, self_attn.k_norm.weight
        if "layernorm.weight" in tl_name or tl_name == "norm.weight" or "q_norm.weight" in tl_name or "k_norm.weight" in tl_name:
            weight = weight + 1.0
            print(f"  [RMSNorm 1+w] {hf_name} → {tl_name}")

        # Handle conv1d: reshape [C, 1, K] → [C, K]
        if "conv1d.weight" in tl_name and weight.ndim == 3:
            weight = weight.squeeze(1)  # [C, 1, K] → [C, K]
            print(f"  [Reshape] {hf_name} {state_dict[hf_name].shape} → {weight.shape}")

        tensors[tl_name] = weight
        print(f"  {hf_name} → {tl_name}  {weight.shape}")

    if skipped:
        print(f"\nSkipped {len(skipped)} tensors:")
        for name in skipped[:10]:
            print(f"  {name}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    # Synthesize zero conv1d.bias for DeltaNet layers that lack it
    # (HF model uses bias=False for conv1d)
    qkv_dim = 6144  # 2*key_dim + value_dim
    for i in range(NUM_LAYERS):
        if not is_full_attention(i):
            bias_name = f"layers.{i}.deltanet.conv1d.bias"
            if bias_name not in tensors:
                tensors[bias_name] = np.zeros(qkv_dim, dtype=np.float32)
                print(f"  [Synth] {bias_name} = zeros({qkv_dim})")

    return tensors


def export_tokenizer(model_path: str, output_path: str):
    """Export tokenizer to binary format for Zig."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    # Build ID → token bytes mapping
    id_to_token = {}
    for token_str, token_id in vocab.items():
        # Encode the token string to bytes
        if isinstance(token_str, str):
            token_bytes = token_str.encode("utf-8")
        else:
            token_bytes = token_str
        id_to_token[token_id] = token_bytes

    # Get merges if available
    merges = []
    if hasattr(tokenizer, "bpe_ranks") and tokenizer.bpe_ranks:
        # tiktoken-style
        for (a, b), rank in sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1]):
            merged = a + b
            a_id = vocab.get(a.decode("utf-8") if isinstance(a, bytes) else a)
            b_id = vocab.get(b.decode("utf-8") if isinstance(b, bytes) else b)
            m_id = vocab.get(merged.decode("utf-8") if isinstance(merged, bytes) else merged)
            if a_id is not None and b_id is not None and m_id is not None:
                merges.append((a_id, b_id, m_id))
    if not merges and hasattr(tokenizer, "backend_tokenizer"):
        # Try HuggingFace tokenizer model (handles string and list merge formats)
        try:
            model_info = json.loads(tokenizer.backend_tokenizer.to_str())
            if "model" in model_info and "merges" in model_info["model"]:
                for merge_entry in model_info["model"]["merges"]:
                    # Handle both "a b" string format and ["a", "b"] list format
                    if isinstance(merge_entry, str):
                        parts = merge_entry.split(" ")
                        if len(parts) != 2:
                            continue
                        a_str, b_str = parts
                    elif isinstance(merge_entry, (list, tuple)) and len(merge_entry) == 2:
                        a_str, b_str = merge_entry[0], merge_entry[1]
                    else:
                        continue
                    merged_str = a_str + b_str
                    a_id = vocab.get(a_str)
                    b_id = vocab.get(b_str)
                    m_id = vocab.get(merged_str)
                    if a_id is not None and b_id is not None and m_id is not None:
                        merges.append((a_id, b_id, m_id))
        except Exception as e:
            print(f"Warning: Could not extract merges: {e}")

    max_token_len = max(len(t) for t in id_to_token.values()) if id_to_token else 0

    print(f"Vocab size: {vocab_size}, Merges: {len(merges)}, Max token len: {max_token_len}")

    # Write binary format
    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", vocab_size, len(merges), max_token_len))

        # Write tokens in ID order
        for token_id in range(vocab_size):
            token_bytes = id_to_token.get(token_id, b"")
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)

        # Write merges
        for a_id, b_id, m_id in merges:
            f.write(struct.pack("<III", a_id, b_id, m_id))

    print(f"Tokenizer exported to {output_path} ({Path(output_path).stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5-0.8B to Tomoul .tl format")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HuggingFace model path or ID")
    parser.add_argument("--output", default=None, help="Output .tl file path")
    parser.add_argument("--quant", choices=["f32", "q8", "q8k", "q4", "f16"], default="f32",
                        help="Quantization format")
    parser.add_argument("--tokenizer-output", default=None, help="Export tokenizer to binary file")
    parser.add_argument("--verify", action="store_true", help="Verify quantization accuracy")
    args = parser.parse_args()

    if args.tokenizer_output:
        export_tokenizer(args.model, args.tokenizer_output)

    if args.output:
        print(f"Loading model from {args.model}...")
        state_dict = load_model_weights(args.model)
        print(f"Loaded {len(state_dict)} tensors")

        print("\nMapping tensors...")
        tensors = process_weights(state_dict)
        print(f"\nMapped {len(tensors)} tensors for export")

        # Map quant format
        quant_map = {
            "f32": QuantFormat.F32,
            "q8": QuantFormat.Q8_0,
            "q8k": QuantFormat.Q8_K,
            "q4": QuantFormat.Q4_0,
            "f16": QuantFormat.F16,
        }
        quant_format = quant_map[args.quant]

        print(f"\nExporting to {args.output} (format: {args.quant})...")
        export_tensors(tensors, args.output, quant_format, verify=args.verify)
        print(f"Done! Output: {args.output} ({Path(args.output).stat().st_size / 1024 / 1024:.1f} MB)")

    if not args.output and not args.tokenizer_output:
        parser.print_help()


if __name__ == "__main__":
    main()
