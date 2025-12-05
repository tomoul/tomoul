#!/usr/bin/env python3
"""
Export XLM-RoBERTa punctuation models to .tl format for Zig inference.

This script exports HuggingFace transformer-based punctuation models
(XLM-RoBERTa, DistilBERT) to the Tomoul binary format.

Supported models:
  - oliverguhr/fullstop-punctuation-multilang-large (XLM-RoBERTa - punctuation)
  - oliverguhr/fullstop-punctuation-multilang-small (DistilBERT - punctuation)
  - kredor/punctuate-all (token classification)
  - ProsusAI/finbert (DistilBERT - financial sentiment)
  - Any XLM-RoBERTa or DistilBERT model from HuggingFace

Usage:
    python tools/export_xlm_roberta_punctuation.py -m oliverguhr/fullstop-punctuation-multilang-large -o models/
    python tools/export_xlm_roberta_punctuation.py -m oliverguhr/fullstop-punctuation-multilang-small -o models/
    python tools/export_xlm_roberta_punctuation.py -m ProsusAI/finbert -o artifacts/
"""

import torch
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import struct
import numpy as np
from pathlib import Path
import argparse
import re

MAGIC = b'TOUL'
VERSION = 1

# Known model configurations for verification
KNOWN_MODELS = {
    "oliverguhr/fullstop-punctuation-multilang-large": {
        "task": "token_classification",
        "labels": ["O", ",", ".", "?", ".U", ",U"],
        "test_text": "hello my name is john how are you",
    },
    "kredor/punctuate-all": {
        "task": "token_classification",
        "labels": ["O", ",", ".", "?", "-", ":"],
        "test_text": "hello my name is john how are you",
    },
    "oliverguhr/fullstop-punctuation-multilingual-sonar-base": {
        "task": "token_classification",
        "labels": ["O", ",", ".", "?", ".U", ",U"],
        "test_text": "hello my name is john how are you",
    },
    "ProsusAI/finbert": {
        "task": "sequence_classification",
        "labels": ["positive", "negative", "neutral"],
        "test_text": "The stock market had a great day with major gains.",
    },
}


def get_model_short_name(model_name: str) -> str:
    """Extract a short name from the HuggingFace model identifier."""
    # Remove owner prefix and convert to snake_case
    short = model_name.split("/")[-1]
    # Convert hyphens to underscores
    short = short.replace("-", "_")
    return short


def load_model(model_name: str):
    """Load the appropriate model type based on configuration."""
    print(f"Loading model from HuggingFace: {model_name}")
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Determine model type from config or known models
    model_info = KNOWN_MODELS.get(model_name, {})
    task = model_info.get("task")
    
    if task is None:
        # Try to infer from config
        if hasattr(config, "num_labels") and config.num_labels > 1:
            # Check architecture hints
            if "token" in str(type(config)).lower() or config.num_labels > 3:
                task = "token_classification"
            else:
                task = "sequence_classification"
        else:
            task = "token_classification"  # Default
    
    if task == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    model.eval()
    return model, tokenizer, task


def export_distilbert_model(model_name: str, output_dir: Path, verify: bool = True):
    """Export a DistilBERT model to .tl format."""
    model, tokenizer, task = load_model(model_name)
    
    short_name = get_model_short_name(model_name)

    print(f"\nModel config:")
    print(f"  Model: {model_name}")
    print(f"  Task: {task}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Num heads: {model.config.num_attention_heads}")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Num labels: {model.config.num_labels}")
    print(f"  Max position embeddings: {model.config.max_position_embeddings}")

    # Collect all tensors with clean names
    tensors = {}

    # Extract all weights from the model
    print("\nModel weights:")
    for name, param in model.named_parameters():
        # Remove "distilbert." prefix for cleaner names
        clean_name = name.replace("distilbert.", "")
        tensor = param.detach().cpu().float().contiguous()
        tensors[clean_name] = tensor
        print(f"  {clean_name}: {list(tensor.shape)}")

    # Export model weights
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{short_name}.tl"
    export_tensors(tensors, model_path)

    # Export vocabulary
    vocab_path = output_dir / f"{short_name}_vocab.txt"
    export_vocab(tokenizer, vocab_path)
    
    # Export model metadata
    meta_path = output_dir / f"{short_name}_meta.txt"
    export_metadata(model, model_name, task, meta_path)

    # Verification
    if verify:
        verify_model(model, tokenizer, model_name, task)

    print(f"\nExport complete!")
    print(f"  Model: {model_path}")
    print(f"  Vocab: {vocab_path}")
    print(f"  Meta:  {meta_path}")

    return model_path, vocab_path


def export_metadata(model, model_name: str, task: str, output_path: Path):
    """Export model metadata for loading in Zig."""
    print(f"\nExporting metadata to {output_path}")
    
    model_info = KNOWN_MODELS.get(model_name, {})
    labels = model_info.get("labels", [f"label_{i}" for i in range(model.config.num_labels)])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"task={task}\n")
        f.write(f"hidden_size={model.config.hidden_size}\n")
        f.write(f"num_hidden_layers={model.config.num_hidden_layers}\n")
        f.write(f"num_attention_heads={model.config.num_attention_heads}\n")
        f.write(f"intermediate_size={model.config.intermediate_size}\n")
        f.write(f"vocab_size={model.config.vocab_size}\n")
        f.write(f"max_position_embeddings={model.config.max_position_embeddings}\n")
        f.write(f"num_labels={model.config.num_labels}\n")
        f.write(f"labels={','.join(labels)}\n")


def export_tensors(tensors: dict, output_path: Path):
    """Export tensors to .tl binary format."""
    tensor_count = len(tensors)

    print(f"\nExporting {tensor_count} tensors to {output_path}")

    with open(output_path, 'wb') as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', tensor_count))
        f.write(struct.pack('<I', 0))  # Reserved

        # Write tensor metadata
        entries = []
        for name, tensor in tensors.items():
            name_bytes = name.encode('utf-8')
            shape = list(tensor.shape)
            data_size = tensor.numel() * 4  # float32

            # Name
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            # Shape
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))

            # Placeholder for offset (will fill later)
            offset_pos = f.tell()
            f.write(struct.pack('<Q', 0))  # offset
            f.write(struct.pack('<Q', data_size))  # size

            entries.append((tensor, offset_pos))

        # Write tensor data
        for tensor, offset_pos in entries:
            current_pos = f.tell()

            # Go back and write the offset
            f.seek(offset_pos)
            f.write(struct.pack('<Q', current_pos))
            f.seek(0, 2)  # Back to end

            # Write data
            f.write(tensor.numpy().astype(np.float32).tobytes())

    file_size = output_path.stat().st_size
    print(f"Exported: {file_size / 1024 / 1024:.2f} MB")


def export_vocab(tokenizer, output_path: Path):
    """Export vocabulary for tokenization."""
    print(f"\nExporting vocabulary to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
            # Escape special characters
            token_escaped = token.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t')
            f.write(f"{token_escaped}\n")

    print(f"Exported vocab: {len(tokenizer.vocab)} tokens")


def verify_model(model, tokenizer, model_name: str, task: str):
    """Run test inference to verify the model works."""
    print("\n=== Verification ===")
    
    model_info = KNOWN_MODELS.get(model_name, {})
    test_text = model_info.get("test_text", "hello my name is john how are you")
    labels = model_info.get("labels", [f"label_{i}" for i in range(model.config.num_labels)])

    inputs = tokenizer(test_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        
    if task == "token_classification":
        predictions = torch.argmax(outputs.logits, dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        print(f"Input: {test_text}")
        print("Token predictions:")
        for token, pred in zip(tokens, predictions[0]):
            label = labels[pred.item()] if pred.item() < len(labels) else f"label_{pred.item()}"
            print(f"  {token:15s}: {label}")
    else:
        # Sequence classification
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        
        print(f"Input: {test_text}")
        print("Class probabilities:")
        for i, (label, prob) in enumerate(zip(labels, probs[0])):
            marker = " <--" if i == predicted_class else ""
            print(f"  {label:15s}: {prob.item():.4f}{marker}")
        print(f"\nPredicted: {labels[predicted_class]}")


def print_model_architecture(model_name: str):
    """Print detailed model architecture."""
    model, _, task = load_model(model_name)
    
    print(f"\n=== Model Architecture: {model_name} ===")
    print(f"Task: {task}")

    # Try to get distilbert attribute (varies by model)
    base = getattr(model, 'distilbert', None) or getattr(model, 'bert', None) or model
    
    # Embeddings
    if hasattr(base, 'embeddings'):
        emb = base.embeddings
        print("\nEmbeddings:")
        if hasattr(emb, 'word_embeddings'):
            print(f"  word_embeddings: {list(emb.word_embeddings.weight.shape)}")
        if hasattr(emb, 'position_embeddings'):
            print(f"  position_embeddings: {list(emb.position_embeddings.weight.shape)}")
        if hasattr(emb, 'LayerNorm'):
            print(f"  LayerNorm: gamma={list(emb.LayerNorm.weight.shape)}, beta={list(emb.LayerNorm.bias.shape)}")

    # Transformer layers
    transformer = getattr(base, 'transformer', None) or getattr(base, 'encoder', None)
    if transformer:
        layers = getattr(transformer, 'layer', None) or []
        print(f"\nTransformer Layers ({len(layers)} layers):")
        for i, layer in enumerate(layers):
            print(f"\n  Layer {i}:")
            # Attention
            attn = getattr(layer, 'attention', None)
            if attn:
                if hasattr(attn, 'q_lin'):
                    print(f"    attention.q_lin: {list(attn.q_lin.weight.shape)}")
                    print(f"    attention.k_lin: {list(attn.k_lin.weight.shape)}")
                    print(f"    attention.v_lin: {list(attn.v_lin.weight.shape)}")
                    print(f"    attention.out_lin: {list(attn.out_lin.weight.shape)}")
            if hasattr(layer, 'sa_layer_norm'):
                print(f"    sa_layer_norm: {list(layer.sa_layer_norm.weight.shape)}")
            # FFN
            ffn = getattr(layer, 'ffn', None)
            if ffn:
                if hasattr(ffn, 'lin1'):
                    print(f"    ffn.lin1: {list(ffn.lin1.weight.shape)}")
                    print(f"    ffn.lin2: {list(ffn.lin2.weight.shape)}")
            if hasattr(layer, 'output_layer_norm'):
                print(f"    output_layer_norm: {list(layer.output_layer_norm.weight.shape)}")

    # Classifier
    if hasattr(model, 'classifier'):
        print("\nClassifier:")
        print(f"  weight: {list(model.classifier.weight.shape)}")
        print(f"  bias: {list(model.classifier.bias.shape)}")


def list_known_models():
    """List supported models with their configurations."""
    print("\n=== Supported Models ===\n")
    for name, info in KNOWN_MODELS.items():
        print(f"  {name}")
        print(f"    Task: {info['task']}")
        print(f"    Labels: {', '.join(info['labels'])}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Export DistilBERT-based models to .tl format for Zig inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/export_distilbert.py -m oliverguhr/fullstop-punctuation-multilang-small -o artifacts/
    python tools/export_distilbert.py -m ProsusAI/finbert -o artifacts/
    python tools/export_distilbert.py --list-models
    python tools/export_distilbert.py -m oliverguhr/fullstop-punctuation-multilang-small --architecture
"""
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="oliverguhr/fullstop-punctuation-multilang-large",
        help="HuggingFace model name (default: oliverguhr/fullstop-punctuation-multilang-large)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("artifacts"),
        help="Output directory (default: artifacts)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification"
    )
    parser.add_argument(
        "--architecture",
        action="store_true",
        help="Print model architecture details"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List known/supported models"
    )

    args = parser.parse_args()

    if args.list_models:
        list_known_models()
        return

    if args.architecture:
        print_model_architecture(args.model)
        return

    export_distilbert_model(args.model, args.output, verify=not args.no_verify)


if __name__ == "__main__":
    main()
