# Qwen3.5-0.8B — LLM Inference

0.8B-parameter hybrid Gated DeltaNet + GQA language model from [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B). Runs natively on CPU and GPU (Vulkan) — no Python, no CUDA, no PyTorch.

**Hugging Face:** [tomoul/qwen3.5-0.8b](https://huggingface.co/tomoul/qwen3.5-0.8b)

## Architecture

| Parameter | Value |
|---|---|
| Model | Qwen3.5-0.8B |
| Hidden dim | 1,024 |
| Layers | 24 in [L L L A] × 6 pattern |
| DeltaNet layers | 18 (linear-time recurrence) |
| Full-attention layers | 6 (GQA: 8Q / 2KV, head_dim=256) |
| FFN intermediate | 3,584 (SwiGLU) |
| Vocab size | 248,320 |
| RoPE theta | 10,000,000 (partial factor 0.25) |
| RMS norm eps | 1e-6 |
| Attn output gate | yes |

**Pipeline:** tokenize → chat template → embedding → 24 layers (RMSNorm → DeltaNet/GQA → residual → RMSNorm → SwiGLU FFN → residual) → final RMSNorm → LM head → logits → argmax → decode loop.

### Weight Variants

| Variant | File | Size | Notes |
|---|---|---|---|
| **Q8_K** | `qwen3_5_0.8b_q8k.tl` | ~820 MB | 1 byte/weight + scales, on-the-fly GPU dequant |
| F32 | `qwen3_5_0.8b.tl` | ~2.2 GB | full precision |

Q8_K quantizes all projection matrices to 8-bit with per-32-element block scales. Embeddings, norms, and attention weights remain F32.

## Benchmarks

### GPU (Vulkan) — Windows (RTX 4090) — May 2026

Measured on Windows, NVIDIA GeForce RTX 4090 (24 GB VRAM), Vulkan compute shaders, Zig 0.15.2 ReleaseFast, 200-token decode, 5-run average:

| Engine | Backend | Weights | Decode (tok/s) | VRAM (MB) | CPU→GPU Speedup |
|---|---|---|---|---|---|
| **Tomoul** (Zig, ReleaseFast) | Vulkan | **Q8_K** | **72** | **807** | **8.0×** |
| Tomoul CPU baseline | — | Q8_K | 9 | 0 | — |
| PyTorch CUDA | CUDA | FP32 | 25 | 1,487 | — |

Tomoul Vulkan is **2.88× faster than PyTorch CUDA** on decode throughput while using **1.84× less VRAM** — and **zero dependency on vendor libraries** (no cuBLAS, no cuDNN, no CUDA toolkit). The entire Tomoul runtime is a **~700 KB executable** vs ~2 GB for PyTorch + CUDA.

**Key advantage:** Vulkan runs on any GPU with a Vulkan 1.3 driver (NVIDIA, AMD, Intel) — no CUDA required.

### GPU Optimization History

The 72 tok/s result is the accumulation of multiple phases of optimization. Full details in [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md).

| Phase | Decode (tok/s) | Key change |
|---|---|---|
| 0 — Baseline | 21.6 | Individual SGEMV submits per projection |
| 1 — Q8K-direct SGEMV | 43 | 1 byte/weight, on-the-fly dequant in shader |
| 2 — Fused command buffers | 47 | o_proj+residual+FFN in 1 submit |
| 3 — Vec4 kernel tuning | 53 | Coalesced 16-byte loads, bit-exact RMSNorm |
| 4 — GPU-resident state | 59 | Hidden state stays on GPU across layers |
| 5 — Fused final RMSNorm + LM head | 72 | 1 submit for token-end |

### GPU Shaders

| Shader | Purpose | Bindings |
|---|---|---|
| `sgemv.comp` | F32 matvec: y = A·x | 3 (weight/input/output) |
| `sgemv_q8k.comp` | Q8K matvec w/ on-the-fly dequant | 4 (packed/scales/input/output) |
| `rmsnorm.comp` | In-place RMSNorm | 2 (x/weight) |
| `silu_mul.comp` | SiLU(gate) × up | 2 (gate/up) |
| `residual_add.comp` | hidden += residual | 2 (hidden/residual) |
| `add_dup.comp` | hidden = residual = a + b | 4 |

All shaders compiled to SPIR-V via `@webgpu/glslang` (npm, no Vulkan SDK required). Sources at `src/gpu/shaders/vulkan/*.comp`.

### CPU — Windows (RTX 4090 CPU) — May 2026

| Engine | Weights | Decode (tok/s) | Notes |
|---|---|---|---|
| Tomoul (Zig, ReleaseFast) | Q8_K | 9 | single-threaded |
| Tomoul (Zig, ReleaseFast) | Q8_K | 15 | multi-threaded (threadpool) |

## Quick Start

### Generate Weights

```bash
# Export Q8_K weights (recommended: ~820 MB)
python tools/export_qwen3_5.py --model Qwen/Qwen3.5-0.8B -o artifacts/ -q q8_k

# Export vocabulary
python tools/export_qwen3_5.py --vocab-only -o artifacts/
```

Requires: `pip install torch transformers`

### Build & Run (CLI)

```bash
# Build CLI with GPU acceleration
zig build -Dmodel=qwen3_5-0.8b -Doptimize=ReleaseFast

# Generate text (GPU)
./zig-out/bin/tomoul_qwen3_5-0.8b generate --gpu \
  --weights artifacts/qwen3_5_0.8b_q8k.tl \
  --tokenizer artifacts/qwen3_5_vocab.bin \
  --max-tokens 256 \
  "What is 2+2?"

# Generate text (CPU only)
./zig-out/bin/tomoul_qwen3_5-0.8b generate \
  "Explain quantum computing in simple terms." \
  -n 128
```

### CLI Options

```
tomoul_qwen3_5-0.8b generate [OPTIONS] <prompt>

Options:
  -w, --weights <path>   Path to .tl weights file (default: artifacts/qwen3_5_0.8b_q8k.tl)
  -t, --tokenizer <path> Path to tokenizer binary (default: artifacts/qwen3_5_vocab.bin)
  -n, --max-tokens <N>   Maximum tokens to generate (default: 256)
  --max-cache-len <N>    KV cache size for full-attention layers (default: 4096)
  --gpu                  Enable GPU acceleration (Vulkan compute shaders)
```

### Benchmark

```bash
# Quick benchmark
python tools/benchmark_qwen3_5.py --tomoul-only --runs 5 --gen-tokens 200

# Full comparison (requires PyTorch + transformers)
python tools/benchmark_qwen3_5.py --runs 3
```

## Integration (Zig)

In your `build.zig.zon`:

```zig
.dependencies = .{
    .tomoul = .{
        .url = "https://github.com/tomoul/tomoul/archive/<commit>.tar.gz",
        .hash = "...",
    },
},
```

In your `build.zig`:

```zig
const tomoul_dep = b.dependency("tomoul", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("qwen3_5", tomoul_dep.module("qwen3_5"));
```

Usage:

```zig
const qwen3_5 = @import("qwen3_5");

var model = try qwen3_5.Qwen3_5.load(allocator, loader);
defer model.deinit();

const logits = model.forwardStep(token_id);
// logits is []const f32 of length vocab_size (248,320)
```

## Files

| File | Purpose |
|---|---|
| `model.zig` | Core model: weight loading, forward pass, dispatch hooks |
| `config.zig` | Model configuration and derived dimensions |
| `deltanet.zig` | Gated DeltaNet recurrence (conv1d, delta rule, outer product) |
| `qwen3_5_gpu.zig` | GPU accelerator: Vulkan pipelines, buffers, fused dispatchers |
| `cli.zig` | CLI executable with `--gpu` flag |
| `tokenizer.zig` | BPE tokenizer with chat template formatting |
| `c.zig` | C FFI binding |
| `shaders/` | SPIR-V bytecode embedded at compile time |
| `GPU_OPTIMIZATIONS.md` | Full optimization history and architecture docs |

## Shader Compilation

```powershell
# Install @webgpu/glslang (one-time)
npm install

# Compile all shaders
node compile_shaders.js

# Compile specific shaders
node compile_shaders.js rmsnorm sgemv sgemv_q8k

# Compile single shader inline
node -e "const fs=require('fs');require('@webgpu/glslang')().then(g=>{
  const spv=g.compileGLSL(fs.readFileSync('src/gpu/shaders/vulkan/NAME.comp','utf8'),'compute');
  const buf=Buffer.from(spv.buffer,spv.byteOffset,spv.byteLength);
  fs.writeFileSync('src/gpu/shaders/vulkan/NAME.spv',buf);
  fs.writeFileSync('src/models/qwen3_5/shaders/NAME.spv',buf);
  process.exit(0);
});"
```

No Vulkan SDK required — `@webgpu/glslang` runs glslangValidator in WASM. Note: use `.then()` chaining, not `await` (async/await hangs in Node 24's eval context).

## References

- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) — original model
- [Gated DeltaNet paper](https://arxiv.org/abs/2412.06464) — Yang et al., 2024
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Vulkan Compute Shaders](https://docs.vulkan.org/spec/latest/chapters/compute.html) — Khronos Group

## License

MIT — see repository root.
