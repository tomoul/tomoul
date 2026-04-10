#!/usr/bin/env python3
"""
Silero VAD Benchmark - PyTorch (CPU)

Compares PyTorch inference performance against Tomoul Zig implementation.
Uses snakers4/silero-vad, processing 512-sample chunks (32ms @ 16kHz).

Usage:
    pip install torch torchaudio
    python benchmark_pytorch.py

    # Customize iterations
    BENCH_RUNS=20 python benchmark_pytorch.py
"""

import os
import time
import wave
import struct
import numpy as np
import torch

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms @ 16kHz
NUM_RUNS = int(os.environ.get("BENCH_RUNS", "50"))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model():
    """Load Silero VAD via torch.hub."""
    print("Loading model via torch.hub...")
    start = time.perf_counter()
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    elapsed = time.perf_counter() - start
    print(f"Model loaded in {elapsed:.2f}s\n")
    return model


def read_wav(filepath):
    """Read a 16kHz mono PCM WAV file and return float32 samples in [-1, 1]."""
    with wave.open(filepath, "rb") as wf:
        assert wf.getnchannels() == 1, "WAV must be mono"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit"
        assert wf.getframerate() == SAMPLE_RATE, f"WAV must be {SAMPLE_RATE}Hz"
        frames = wf.readframes(wf.getnframes())
    n_samples = len(frames) // 2
    samples = struct.unpack(f"<{n_samples}h", frames)
    return np.array(samples, dtype=np.float32) / 32768.0


def benchmark_single_chunk(model, num_iters=500):
    """Benchmark single chunk latency (like the JS benchmark)."""
    chunk = np.sin(2 * np.pi * 440 * np.arange(CHUNK_SIZE) / SAMPLE_RATE).astype(np.float32) * 0.5
    tensor = torch.from_numpy(chunk)

    # Warmup
    for _ in range(10):
        model(tensor, SAMPLE_RATE)
    model.reset_states()

    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        model(tensor, SAMPLE_RATE)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    min_t = times[0]

    print(f"--- Single Chunk Latency ({num_iters} calls) ---")
    print(f"  Average: {avg:.3f}ms")
    print(f"  Median:  {p50:.3f}ms")
    print(f"  Min:     {min_t:.3f}ms\n")

    return {"avg": avg, "p50": p50, "min": min_t}


def benchmark_audio(model, name, samples, num_runs):
    """Process audio in 512-sample chunks and measure timing."""
    num_chunks = len(samples) // CHUNK_SIZE
    duration_ms = (len(samples) / SAMPLE_RATE) * 1000

    times = []
    for _ in range(num_runs):
        model.reset_states()
        start = time.perf_counter()
        for c in range(num_chunks):
            chunk = samples[c * CHUNK_SIZE : (c + 1) * CHUNK_SIZE]
            tensor = torch.from_numpy(chunk)
            model(tensor, SAMPLE_RATE)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    per_chunk = avg / num_chunks
    rtf = avg / duration_ms

    print(f"--- {name} ({num_chunks} chunks, {duration_ms / 1000:.2f}s audio) ---")
    print(f"  Total:     {avg:.2f}ms avg, {p50:.2f}ms p50")
    print(f"  Per chunk: {per_chunk:.3f}ms ({CHUNK_SIZE / SAMPLE_RATE * 1000:.1f}ms audio = 32ms)")
    print(f"  RTF:       {rtf:.4f} ({1/rtf:.0f}× realtime)\n")

    return {"avg": avg, "p50": p50, "per_chunk": per_chunk, "rtf": rtf, "num_chunks": num_chunks}


def main():
    print("=" * 60)
    print("Silero VAD Benchmark (PyTorch CPU)")
    print("=" * 60)
    print()

    model = load_model()

    print("Running benchmarks...\n")

    # Single chunk throughput
    single = benchmark_single_chunk(model, 500)

    # WAV file benchmarks
    wav_files = ["speech.wav", "silence.wav", "noise.wav"]
    results = {}

    for wav in wav_files:
        wav_path = os.path.join(SCRIPT_DIR, wav)
        if os.path.exists(wav_path):
            samples = read_wav(wav_path)
            results[wav] = benchmark_audio(model, wav, samples, NUM_RUNS)
        else:
            print(f"Skipping {wav} (not found)")

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Single chunk latency: {single['avg']:.3f}ms avg, {single['p50']:.3f}ms p50")
    for name, r in results.items():
        print(f"  {name}: {r['avg']:.2f}ms total, {r['per_chunk']:.3f}ms/chunk, {1/r['rtf']:.0f}× RT")
    print("=" * 60)


if __name__ == "__main__":
    main()
