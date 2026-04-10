#!/usr/bin/env python3
"""
Gemma 2B Benchmark - PyTorch (CPU and CUDA)

Compares PyTorch inference performance against Tomoul Zig implementation.
Uses google/gemma-2b (2 billion params, decoder-only LLM).

Usage:
    python benchmark_pytorch.py

Requires:
    pip install torch transformers accelerate
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2b"

# Test prompts (same as benchmark.js)
PROMPTS = [
    "The meaning of life is",
    "In a world where technology",
    "Once upon a time in a land far away",
    "The quick brown fox",
    "Explain the theory of relativity in simple terms:",
]

NUM_ITERATIONS = int(os.environ.get('BENCH_ITERS', '5'))
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '64'))


def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, device):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if temperature == 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=50,
            )

    # Decode only the generated tokens (skip input)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids)


def benchmark_device(device_name):
    """Run benchmark on specified device."""
    print(f"\n{'='*60}")
    print(f"PyTorch Benchmark - {device_name.upper()}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"{'='*60}")

    device = torch.device(device_name)
    dtype = torch.float16 if device_name == "cuda" else torch.float32

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    load_end = time.perf_counter()
    print(f"Model loaded in {load_end - load_start:.2f}s")

    # Warm-up
    print("\nWarm-up run...")
    text, ntok = generate_text(model, tokenizer, PROMPTS[0], 16, 0.0, device)
    if device_name == "cuda":
        torch.cuda.synchronize()
    print(f"  Output ({ntok} tokens): \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

    # --- Single prompt benchmark (greedy) ---
    print(f"\n--- Single Prompt (greedy, {MAX_TOKENS} tokens max) ---")
    single_times = []
    total_tokens = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        text, ntok = generate_text(model, tokenizer, PROMPTS[0], MAX_TOKENS, 0.0, device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        single_times.append((end - start) * 1000)
        total_tokens.append(ntok)

    avg = sum(single_times) / len(single_times)
    min_t = min(single_times)
    max_t = max(single_times)
    p50 = sorted(single_times)[len(single_times) // 2]
    avg_tok = sum(total_tokens) / len(total_tokens)
    print(f"\n  Single prompt ({NUM_ITERATIONS} runs):")
    print(f"    Average: {avg:.1f}ms")
    print(f"    Median:  {p50:.1f}ms")
    print(f"    Min:     {min_t:.1f}ms")
    print(f"    Max:     {max_t:.1f}ms")
    print(f"    Tokens/sec: {(avg_tok / (avg / 1000)):.1f}")

    # --- 5 prompts sequential ---
    print(f"\n--- 5 Prompts (sequential, greedy) ---")
    seq_times = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for prompt in PROMPTS:
            generate_text(model, tokenizer, prompt, MAX_TOKENS, 0.0, device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        seq_times.append((end - start) * 1000)

    avg_seq = sum(seq_times) / len(seq_times)
    min_t = min(seq_times)
    p50_seq = sorted(seq_times)[len(seq_times) // 2]
    print(f"  5 prompts sequential ({NUM_ITERATIONS} runs):")
    print(f"    Average: {avg_seq:.1f}ms  ({avg_seq/5:.1f}ms per prompt)")
    print(f"    Median:  {p50_seq:.1f}ms  ({p50_seq/5:.1f}ms per prompt)")
    print(f"    Min:     {min_t:.1f}ms  ({min_t/5:.1f}ms per prompt)")

    # --- Short generation (16 tokens) ---
    print(f"\n--- Latency Test (16 tokens, greedy) ---")
    short_times = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        generate_text(model, tokenizer, PROMPTS[0], 16, 0.0, device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        short_times.append((end - start) * 1000)

    avg_short = sum(short_times) / len(short_times)
    p50_short = sorted(short_times)[len(short_times) // 2]
    print(f"  Short generation ({NUM_ITERATIONS} runs):")
    print(f"    Average: {avg_short:.1f}ms")
    print(f"    Median:  {p50_short:.1f}ms")

    # --- Sampling ---
    print(f"\n--- Sampling (temperature=0.7, {MAX_TOKENS} tokens) ---")
    samp_times = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        generate_text(model, tokenizer, PROMPTS[0], MAX_TOKENS, 0.7, device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        samp_times.append((end - start) * 1000)

    avg_samp = sum(samp_times) / len(samp_times)
    p50_samp = sorted(samp_times)[len(samp_times) // 2]
    print(f"  Sampling ({NUM_ITERATIONS} runs):")
    print(f"    Average: {avg_samp:.1f}ms")
    print(f"    Median:  {p50_samp:.1f}ms")

    return {
        "device": device_name,
        "single_avg": avg,
        "single_p50": p50,
        "seq5_avg": avg_seq,
        "short_avg": avg_short,
        "short_p50": p50_short,
        "sampling_avg": avg_samp,
    }


def main():
    print("=" * 60)
    print("Gemma 2B Benchmark - PyTorch")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max tokens: {MAX_TOKENS}")
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
        print(f"    Single (greedy):     {r['single_avg']:.1f}ms avg, {r['single_p50']:.1f}ms p50")
        print(f"    Short (16 tok):      {r['short_avg']:.1f}ms avg, {r['short_p50']:.1f}ms p50")
        print(f"    5 seq (greedy):      {r['seq5_avg']:.1f}ms avg ({r['seq5_avg']/5:.1f}ms/prompt)")
        print(f"    Sampling (t=0.7):    {r['sampling_avg']:.1f}ms avg")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
