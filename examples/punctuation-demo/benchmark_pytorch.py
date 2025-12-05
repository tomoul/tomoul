#!/usr/bin/env python3
"""
Punctuation Benchmark - PyTorch (CPU and CUDA)

Compares PyTorch inference performance against Tomoul Zig implementation.

Usage:
    python benchmark_pytorch.py                    # Default: large model
    TOMOUL_MODEL=sonar-base python benchmark_pytorch.py  # Sonar-base model
"""

import os
import time
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Model selection: 'large' or 'sonar-base'
MODEL_CHOICE = os.environ.get('TOMOUL_MODEL', 'large')

MODELS = {
    'large': 'oliverguhr/fullstop-punctuation-multilang-large',
    'sonar-base': 'oliverguhr/fullstop-punctuation-multilingual-sonar-base',
}

MODEL_NAME = MODELS.get(MODEL_CHOICE, MODELS['large'])

# Test texts (same as benchmark.js)
SHORT_TEXT = "hello world how are you doing today"
LONG_TEXT = "hello world this is a test of the punctuation restoration system we are going to see how fast it can process this paragraph which contains exactly one hundred words or close to it the goal is to measure the performance difference between the python implementation using pytorch and the zig implementation using native code with manual memory management this benchmark will help us understand if the zero interpreter overhead and better memory control in zig provides a significant speedup compared to python we expect zig to be two to five times faster than python for this workload lets see if that prediction holds true"

NUM_ITERATIONS = 10

# Label mapping for punctuation
LABELS = ["O", ",", ".", "?", ".U", ",U"]

def process_text(model, tokenizer, text, device):
    """Process text and return punctuated result."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Reconstruct text with punctuation
    result = []
    for token, pred in zip(tokens, predictions):
        if token in ['<s>', '</s>', '<pad>']:
            continue

        # Handle subword tokens (starting with ▁ in XLM-RoBERTa)
        if token.startswith('▁'):
            if result:
                result.append(' ')
            result.append(token[1:])
        else:
            result.append(token)

        # Add punctuation
        label = LABELS[pred.item()]
        if label == ',':
            result.append(',')
        elif label == '.':
            result.append('.')
        elif label == '?':
            result.append('?')
        elif label == '.U':
            result.append('.')
        elif label == ',U':
            result.append(',')

    return ''.join(result)

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
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    load_end = time.perf_counter()
    print(f"Model loaded in {load_end - load_start:.2f}s")

    # Warm-up
    print("\nWarm-up run...")
    _ = process_text(model, tokenizer, SHORT_TEXT, device)
    if device_name == "cuda":
        torch.cuda.synchronize()

    # Benchmark short text
    print(f"\n--- Short Text ({len(SHORT_TEXT.split())} words) ---")
    times = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = process_text(model, tokenizer, SHORT_TEXT, device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
        if i == 0:
            print(f'Output: "{result}"')

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"\nShort text ({NUM_ITERATIONS} runs):")
    print(f"  Average: {avg:.2f}ms")
    print(f"  Min: {min_t:.2f}ms")
    print(f"  Max: {max_t:.2f}ms")

    # Benchmark long text
    print(f"\n--- Long Text (~100 words) ---")
    times = []
    for i in range(NUM_ITERATIONS):
        if device_name == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = process_text(model, tokenizer, LONG_TEXT, device)
        if device_name == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
        if i == 0:
            print(f'Output: "{result[:100]}..."')

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"\nLong text ({NUM_ITERATIONS} runs):")
    print(f"  Average: {avg:.2f}ms")
    print(f"  Min: {min_t:.2f}ms")
    print(f"  Max: {max_t:.2f}ms")

    return {
        'device': device_name,
        'short_avg': avg,
        'long_avg': avg,
    }

def main():
    print("="*60)
    print("Punctuation Benchmark - PyTorch")
    print("="*60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = []

    # Benchmark CPU
    results.append(benchmark_device("cpu"))

    # Benchmark CUDA if available
    if torch.cuda.is_available():
        results.append(benchmark_device("cuda"))

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
