#!/usr/bin/env python3
"""
Benchmark script for comparing Whisper implementations.
Compares PyTorch/OpenAI Whisper with Tomoul.
"""

import time
import subprocess
import sys

def benchmark_pytorch_whisper(audio_path: str, model_name: str = "tiny", runs: int = 3, num_threads: int = 1):
    """Benchmark PyTorch Whisper implementation."""
    try:
        import torch
        import whisper
    except ImportError:
        print("Installing whisper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper", "-q"])
        import torch
        import whisper

    # Set thread count for fair comparison
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    print(f"\n{'='*60}")
    print(f"PyTorch Whisper Benchmark (model: {model_name}, threads: {num_threads})")
    print(f"{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Num threads: {torch.get_num_threads()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Load model on CPU (do this once, outside timing)
    print(f"\nLoading model '{model_name}' on CPU...")
    load_start = time.perf_counter()
    model = whisper.load_model(model_name, device="cpu")
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.3f}s")

    # Warmup run
    print("\nWarmup run...")
    _ = model.transcribe(audio_path, language="en", fp16=False)

    # Benchmark runs
    times = []
    for i in range(runs):
        print(f"Run {i+1}/{runs}...", end=" ", flush=True)
        start = time.perf_counter()
        result = model.transcribe(audio_path, language="en", fp16=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"{elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n--- PyTorch Whisper Results ---")
    print(f"Average: {avg_time:.3f}s")
    print(f"Min:     {min_time:.3f}s")
    print(f"Max:     {max_time:.3f}s")
    print(f"\nTranscription: {result['text'][:200]}...")

    return {
        "avg": avg_time,
        "min": min_time,
        "max": max_time,
        "text": result["text"]
    }


def benchmark_tomoul(audio_path: str, runs: int = 3, mode: str = "zblas"):
    """Benchmark Tomoul Whisper implementation."""
    print(f"\n{'='*60}")
    print(f"Tomoul Whisper Benchmark (mode: {mode})")
    print(f"{'='*60}")

    # Build with specified mode
    build_flags = {
        "zblas": "-Dzblas=true",
        "blas": "-Dblas=true",
        "fallback": "-Dzblas=false"
    }

    flag = build_flags.get(mode, "-Dzblas=true")
    print(f"Building with {flag}...")

    build_result = subprocess.run(
        ["zig", "build", "-Dmodel=whisper-tiny", "-Doptimize=ReleaseFast", flag],
        capture_output=True,
        text=True
    )
    if build_result.returncode != 0:
        print(f"Build failed: {build_result.stderr}")
        return None

    # Warmup run
    print("Warmup run...")
    subprocess.run(
        ["./zig-out/bin/tomoul_whisper-tiny", "transcribe", audio_path],
        capture_output=True
    )

    # Benchmark runs
    times = []
    output_text = ""
    for i in range(runs):
        print(f"Run {i+1}/{runs}...", end=" ", flush=True)
        start = time.perf_counter()
        result = subprocess.run(
            ["./zig-out/bin/tomoul_whisper-tiny", "transcribe", audio_path],
            capture_output=True,
            text=True
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"{elapsed:.3f}s")

        # Extract timing from output
        for line in result.stdout.split('\n'):
            if 'Total:' in line:
                output_text = line

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n--- Tomoul ({mode}) Results ---")
    print(f"Average: {avg_time:.3f}s")
    print(f"Min:     {min_time:.3f}s")
    print(f"Max:     {max_time:.3f}s")

    return {
        "avg": avg_time,
        "min": min_time,
        "max": max_time,
    }


def main():
    audio_path = "models/english_man.wav"
    runs = 3

    print("="*60)
    print("Whisper Benchmark: PyTorch vs Tomoul")
    print("="*60)
    print(f"Audio: {audio_path}")
    print(f"Runs: {runs}")

    results = {}

    # Benchmark PyTorch
    try:
        results["pytorch"] = benchmark_pytorch_whisper(audio_path, "tiny", runs)
    except Exception as e:
        print(f"PyTorch benchmark failed: {e}")
        results["pytorch"] = None

    # Benchmark Tomoul modes
    for mode in ["zblas", "fallback"]:
        try:
            results[f"tomoul_{mode}"] = benchmark_tomoul(audio_path, runs, mode)
        except Exception as e:
            print(f"Tomoul {mode} benchmark failed: {e}")
            results[f"tomoul_{mode}"] = None

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Implementation':<20} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10}")
    print("-"*50)

    for name, data in results.items():
        if data:
            print(f"{name:<20} {data['avg']:<10.3f} {data['min']:<10.3f} {data['max']:<10.3f}")

    # Speedup comparison
    if results.get("pytorch") and results.get("tomoul_fallback"):
        speedup = results["pytorch"]["avg"] / results["tomoul_fallback"]["avg"]
        print(f"\nTomoul vs PyTorch speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
