#!/usr/bin/env python3
"""
Benchmark Qwen3.5-0.8B: Tomoul (Zig CPU) vs PyTorch (CPU + CUDA).

Measures:
  - Load time
  - Prefill latency (tokens/sec for prompt processing)
  - Decode latency (tokens/sec for generation)
  - Peak memory usage

Usage:
    python benchmark_qwen3_5.py                    # All benchmarks
    python benchmark_qwen3_5.py --pytorch-only      # Only PyTorch
    python benchmark_qwen3_5.py --tomoul-only       # Only Tomoul
    python benchmark_qwen3_5.py --prompt "Hello"    # Custom prompt
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import torch


@dataclass
class BenchmarkResult:
    name: str
    load_time_s: float = 0.0
    prefill_time_ms: float = 0.0
    prefill_tokens: int = 0
    decode_time_ms: float = 0.0
    decode_tokens: int = 0
    total_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    output_text: str = ""

    @property
    def prefill_tok_s(self) -> float:
        return self.prefill_tokens / (self.prefill_time_ms / 1000) if self.prefill_time_ms > 0 else 0

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / (self.decode_time_ms / 1000) if self.decode_time_ms > 0 else 0


PROMPT = "Explain the theory of relativity in simple terms."
GEN_TOKENS = 100
WARMUP_RUNS = 1
BENCH_RUNS = 3


def benchmark_pytorch(model_path: str, device: str, prompt: str,
                      gen_tokens: int, runs: int) -> BenchmarkResult:
    """Benchmark PyTorch Qwen3.5-0.8B inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    result = BenchmarkResult(name=f"PyTorch ({device.upper()})")

    # Load
    print(f"\n{'='*60}")
    print(f"Benchmarking: {result.name}")
    print(f"{'='*60}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    result.load_time_s = time.perf_counter() - t0
    print(f"Load time: {result.load_time_s:.2f}s")

    # Tokenize with chat template
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
    result.prefill_tokens = input_ids.shape[1]
    print(f"Prompt tokens: {result.prefill_tokens}")

    # Warmup
    print("Warmup...", end=" ", flush=True)
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    print("done")

    # Benchmark using model.generate() for correct Qwen3.5 DeltaNet cache handling
    total_times = []

    for r in range(runs):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        total_times.append(elapsed * 1000)

        new_ids = output_ids[0, input_ids.shape[1]:]
        result.decode_tokens = len(new_ids)
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        result.output_text = gen_text[:200]

        print(f"  Run {r+1}/{runs}: {elapsed*1000:.1f}ms "
              f"({result.decode_tokens} tokens)")

    result.decode_time_ms = sum(total_times) / len(total_times)

    if device == "cuda":
        result.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\n  Avg decode: {result.decode_time_ms:.1f}ms "
          f"({result.decode_tok_s:.0f} tok/s)")
    if result.peak_memory_mb > 0:
        print(f"  Peak GPU mem: {result.peak_memory_mb:.0f} MB")
    print(f"  Output: '{result.output_text[:100]}...'")

    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def benchmark_tomoul(tomoul_exe: str, weights_path: str, tokenizer_path: str,
                     prompt: str, gen_tokens: int, runs: int,
                     use_gpu: bool = False) -> BenchmarkResult:
    """Benchmark Tomoul Qwen3.5-0.8B inference."""
    mode = "Zig GPU" if use_gpu else "Zig CPU"
    result = BenchmarkResult(name=f"Tomoul ({mode})")

    print(f"\n{'='*60}")
    print(f"Benchmarking: {result.name}")
    print(f"{'='*60}")

    if not os.path.isfile(tomoul_exe):
        print(f"  [SKIP] Executable not found: {tomoul_exe}")
        return result

    cmd = [
        tomoul_exe, "generate", prompt,
        "--weights", weights_path,
        "--tokenizer", tokenizer_path,
        "--max-tokens", str(gen_tokens),
    ]
    if use_gpu:
        cmd.append("--gpu")

    # Warmup
    print("Warmup...", end=" ", flush=True)
    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
        print("done")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"failed: {e}")
        return result

    # Benchmark runs
    times = []
    for r in range(runs):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)

        # Parse timing from stderr (std.debug.print output)
        output = proc.stderr if proc.stderr else proc.stdout
        result.output_text = output.strip()[:200]

        # Extract tok/s from output line like: "--- 100 tokens in 1234.5ms (81.1 tok/s) ---"
        for line in output.split("\n"):
            if "tok/s" in line and "tokens in" in line:
                try:
                    parts = line.strip().split()
                    tok_idx = parts.index("tokens")
                    result.decode_tokens = int(parts[tok_idx - 1])
                    ms_part = parts[tok_idx + 2]
                    result.total_time_ms = float(ms_part.rstrip("ms"))
                except (ValueError, IndexError):
                    pass

        print(f"  Run {r+1}/{runs}: {elapsed*1000:.1f}ms (wall clock)")

    avg_wall = sum(times) / len(times)
    if result.total_time_ms > 0:
        result.decode_time_ms = result.total_time_ms
    else:
        result.decode_time_ms = avg_wall

    print(f"\n  Avg wall time: {avg_wall:.1f}ms")
    if result.decode_tokens > 0:
        print(f"  Avg decode: {result.decode_tok_s:.0f} tok/s")

    return result


def print_summary(results: list):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"{'BENCHMARK SUMMARY':^70}")
    print(f"{'='*70}")

    header = f"{'Implementation':<25} {'Prefill':>10} {'Decode':>10} {'Decode':>10} {'Memory':>10}"
    units =  f"{'':25} {'(tok/s)':>10} {'(tok/s)':>10} {'(ms)':>10} {'(MB)':>10}"
    print(header)
    print(units)
    print("-" * 70)

    for r in results:
        if r.load_time_s == 0 and r.decode_time_ms == 0:
            continue
        mem_str = f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb > 0 else "N/A"
        prefill_str = f"{r.prefill_tok_s:.0f}" if r.prefill_tok_s > 0 else "N/A"
        print(f"{r.name:<25} {prefill_str:>10} {r.decode_tok_s:>10.0f} "
              f"{r.decode_time_ms:>10.1f} {mem_str:>10}")

    # Speedup comparisons
    valid = [r for r in results if r.decode_tok_s > 0]
    if len(valid) >= 2:
        print(f"\n{'Relative Performance':^70}")
        print("-" * 70)
        baseline = valid[0]
        for r in valid[1:]:
            speedup = r.decode_tok_s / baseline.decode_tok_s
            marker = "faster" if speedup > 1 else "slower"
            print(f"  {r.name} vs {baseline.name}: {speedup:.2f}x {marker}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.5-0.8B inference")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                        help="HuggingFace model path")
    parser.add_argument("--prompt", default=PROMPT,
                        help="Test prompt")
    parser.add_argument("--gen-tokens", type=int, default=GEN_TOKENS,
                        help="Number of tokens to generate")
    parser.add_argument("--runs", type=int, default=BENCH_RUNS,
                        help="Number of benchmark runs")
    parser.add_argument("--pytorch-only", action="store_true",
                        help="Only run PyTorch benchmarks")
    parser.add_argument("--tomoul-only", action="store_true",
                        help="Only run Tomoul benchmarks")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Skip CUDA benchmark")
    parser.add_argument("--no-pytorch-cpu", action="store_true",
                        help="Skip PyTorch CPU benchmark (very slow for F32)")
    parser.add_argument("--tomoul-exe",
                        default="zig-out/bin/tomoul_qwen3_5-0.8b.exe",
                        help="Path to Tomoul executable")
    parser.add_argument("--weights",
                        default="artifacts/qwen3_5_0.8b_q8k.tl",
                        help="Path to .tl weights file")
    parser.add_argument("--tokenizer",
                        default="artifacts/qwen3_5_vocab.bin",
                        help="Path to tokenizer binary")
    args = parser.parse_args()

    print(f"Qwen3.5-0.8B Inference Benchmark")
    print(f"Prompt: '{args.prompt[:60]}...'")
    print(f"Generate: {args.gen_tokens} tokens, {args.runs} runs")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {'available - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")

    results = []

    if not args.tomoul_only:
        # PyTorch CPU
        if not args.no_pytorch_cpu:
            try:
                r = benchmark_pytorch(args.model, "cpu", args.prompt,
                                      args.gen_tokens, args.runs)
                results.append(r)
            except Exception as e:
                print(f"PyTorch CPU benchmark failed: {e}")

        # PyTorch CUDA
        if torch.cuda.is_available() and not args.no_cuda:
            try:
                r = benchmark_pytorch(args.model, "cuda", args.prompt,
                                      args.gen_tokens, args.runs)
                results.append(r)
            except Exception as e:
                print(f"PyTorch CUDA benchmark failed: {e}")

    if not args.pytorch_only:
        # Tomoul CPU
        try:
            r = benchmark_tomoul(args.tomoul_exe, args.weights, args.tokenizer,
                                 args.prompt, args.gen_tokens, args.runs,
                                 use_gpu=False)
            results.append(r)
        except Exception as e:
            print(f"Tomoul CPU benchmark failed: {e}")

        # Tomoul GPU (Vulkan)
        if not args.no_cuda:
            try:
                r = benchmark_tomoul(args.tomoul_exe, args.weights, args.tokenizer,
                                     args.prompt, args.gen_tokens, args.runs,
                                     use_gpu=True)
                results.append(r)
            except Exception as e:
                print(f"Tomoul GPU benchmark failed: {e}")

    print_summary(results)


if __name__ == "__main__":
    main()
