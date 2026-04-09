#!/usr/bin/env python3
"""
Sentence Transformer Benchmark - PyTorch (CPU and CUDA)

Compares PyTorch inference performance against Tomoul Zig implementation.
Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings).

Usage:
    python benchmark_pytorch.py
"""

import os
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Test sentences (same as benchmark.js and Zig tests)
SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "I had pizza for lunch yesterday",
    "The capital of France is Paris",
    "Quantum computing uses qubits instead of classical bits",
]

# Longer batch for throughput testing
BATCH_SENTENCES = [
    "The weather is nice today",
    "Python is a popular programming language",
    "Neural networks can learn complex patterns",
    "The stock market closed higher on Friday",
    "Renewable energy is becoming more affordable",
    "The quick brown fox jumps over the lazy dog",
    "Deep learning has revolutionized computer vision",
    "Coffee is one of the most consumed beverages worldwide",
    "The Olympic Games are held every four years",
    "Artificial intelligence is transforming healthcare",
]

NUM_ITERATIONS = int(os.environ.get('BENCH_ITERS', '20'))


def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def embed_sentences(model, tokenizer, sentences, device):
    """Embed a list of sentences and return normalized embeddings."""
    encoded = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        model_output = model(**encoded)

    embeddings = mean_pooling(model_output, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def benchmark_device(device_name):
    """Run benchmark on specified device."""
    print(f"\n{'='*60}")
    print(f"PyTorch Benchmark - {device_name.upper()}")
    print(f"{'='*60}")

    device = torch.device(device_name)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    load_end = time.perf_counter()
    print(f"Model loaded in {load_end - load_start:.2f}s")

    # Warm-up
    print("\nWarm-up run...")
    _ = embed_sentences(model, tokenizer, [SENTENCES[0]], device)
    if device_name == "cuda":
        torch.cuda.synchronize()

    # --- Single sentence benchmark ---
    print(f"\n--- Single Sentence ---")
    single_times = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        emb = embed_sentences(model, tokenizer, [SENTENCES[0]], device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        single_times.append((end - start) * 1000)
        if i == 0:
            print(f"  Embedding dim: {emb.shape[1]}")
            norm = np.linalg.norm(emb[0])
            print(f"  L2 norm: {norm:.6f}")
            print(f"  First 5 values: {emb[0][:5]}")

    avg = sum(single_times) / len(single_times)
    min_t = min(single_times)
    max_t = max(single_times)
    p50 = sorted(single_times)[len(single_times) // 2]
    print(f"\n  Single sentence ({NUM_ITERATIONS} runs):")
    print(f"    Average: {avg:.2f}ms")
    print(f"    Median:  {p50:.2f}ms")
    print(f"    Min:     {min_t:.2f}ms")
    print(f"    Max:     {max_t:.2f}ms")

    # --- 5 sentence benchmark (sequential, one at a time) ---
    print(f"\n--- 5 Sentences (sequential) ---")
    seq_times = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for sentence in SENTENCES:
            _ = embed_sentences(model, tokenizer, [sentence], device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        seq_times.append((end - start) * 1000)

    avg = sum(seq_times) / len(seq_times)
    min_t = min(seq_times)
    p50 = sorted(seq_times)[len(seq_times) // 2]
    print(f"  5 sentences sequential ({NUM_ITERATIONS} runs):")
    print(f"    Average: {avg:.2f}ms  ({avg/5:.2f}ms per sentence)")
    print(f"    Median:  {p50:.2f}ms  ({p50/5:.2f}ms per sentence)")
    print(f"    Min:     {min_t:.2f}ms  ({min_t/5:.2f}ms per sentence)")

    # --- Batch 10 sentences ---
    print(f"\n--- 10 Sentences (batched) ---")
    batch_times = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = embed_sentences(model, tokenizer, BATCH_SENTENCES, device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        batch_times.append((end - start) * 1000)

    avg = sum(batch_times) / len(batch_times)
    min_t = min(batch_times)
    p50 = sorted(batch_times)[len(batch_times) // 2]
    print(f"  10 sentences batched ({NUM_ITERATIONS} runs):")
    print(f"    Average: {avg:.2f}ms  ({avg/10:.2f}ms per sentence)")
    print(f"    Median:  {p50:.2f}ms  ({p50/10:.2f}ms per sentence)")
    print(f"    Min:     {min_t:.2f}ms  ({min_t/10:.2f}ms per sentence)")

    return {
        "device": device_name,
        "single_avg": sum(single_times) / len(single_times),
        "single_p50": sorted(single_times)[len(single_times) // 2],
        "seq5_avg": sum(seq_times) / len(seq_times),
        "batch10_avg": sum(batch_times) / len(batch_times),
    }


def main():
    print("=" * 60)
    print("Sentence Transformer Benchmark - PyTorch")
    print("  Model: all-MiniLM-L6-v2 (384-dim)")
    print("=" * 60)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = []

    # Benchmark CPU
    results.append(benchmark_device("cpu"))

    # Benchmark CUDA if available
    if torch.cuda.is_available():
        results.append(benchmark_device("cuda"))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"\n  {r['device'].upper()}:")
        print(f"    Single sentence:    {r['single_avg']:.2f}ms avg, {r['single_p50']:.2f}ms p50")
        print(f"    5 sent sequential:  {r['seq5_avg']:.2f}ms avg ({r['seq5_avg']/5:.2f}ms/sent)")
        print(f"    10 sent batched:    {r['batch10_avg']:.2f}ms avg ({r['batch10_avg']/10:.2f}ms/sent)")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
