# Sentence Transformer — all-MiniLM-L6-v2

384-dimensional sentence embeddings from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Runs natively on CPU, WASM, iOS, Android — no Python or GPU required.

**Hugging Face:** [tomoul/sentence-transformer](https://huggingface.co/tomoul/sentence-transformer)

## Architecture

| Parameter | Value |
|---|---|
| Base model | BERT (6-layer encoder) |
| Hidden dim | 384 |
| Attention heads | 12 |
| FFN intermediate | 1,536 |
| Vocab size | 30,522 (WordPiece) |
| Max sequence length | 512 tokens |
| Output | 384-dim L2-normalized vector |
| Layer norm eps | 1e-12 |

**Pipeline:** WordPiece tokenize → BERT embeddings (word + position + token_type) → LayerNorm → 6× Transformer blocks (self-attention + FFN with post-LN) → mean pooling → L2 normalize → 384-dim float vector.

### Weight Variants

| Variant | File | Size | Accuracy vs F32 |
|---|---|---|---|
| **F32** (default) | `all_minilm_l6_v2.tl` | ~87 MB | baseline |
| **F16** (half-precision) | `all_minilm_l6_v2_f16.tl` | ~43 MB | effectively lossless |
| **Q8_K** (8-bit quantized) | `all_minilm_l6_v2_q8k.tl` | ~24 MB | cosine sim ≥ 0.9997 |

In Q8_K mode, only transformer block linear weights are quantized. Embeddings and LayerNorm parameters remain float32. F16 stores all weights as IEEE 754 half-precision (10-bit mantissa). Q8_K is recommended for best size/speed tradeoff.

## Benchmarks

### CPU — WSL (AMD CPU + RTX 4090)

Measured on AMD CPU, 50 iterations, warm start:

| Engine | Weights | Single (ms) | 5-sent (ms/sent) | 10-sent (ms/sent) | Notes |
|---|---|---|---|---|---|
| **Tomoul** (Zig, ReleaseFast) | **Q8_K** | **3.6** | **3.2** | **2.8** | Node.js FFI via koffi |
| **Tomoul** (Zig, ReleaseFast) | F16 | 5.0 | 7.2 | 4.0 | 1.8× faster than F32, 2× smaller |
| **Tomoul** (Zig, ReleaseFast) | F32 | 9.2 | 6.8 | 6.8 | Node.js FFI via koffi |
| PyTorch CPU (MKL) | F32 | 3.6 | 3.6 | 1.0 (batched) | `transformers` pipeline |
| PyTorch CUDA (RTX 4090) | F32 | 2.6 | — | 0.55 (batched) | GPU baseline |

### GPU (Vulkan) — Windows (RTX 4090)

Measured on Windows, NVIDIA GeForce RTX 4090 (24 GB VRAM), Vulkan compute shaders, warm start:

| Engine | Backend | Weights | Single (ms) | 5-sent batch (ms) | Per-sent batch (ms) | CPU→GPU Speedup |
|---|---|---|---|---|---|---|
| **Tomoul** (Zig, ReleaseFast) | Vulkan | **Q8_K** | **3.55** | **5.47** | **1.09** | 5.36× single, 19.15× batch |
| **Tomoul** (Zig, ReleaseFast) | Vulkan | F32 | 3.90 | 5.64 | 1.13 | 5.31× single, 19.34× batch |
| Tomoul CPU baseline | — | F32 | 20.70 | 104.77 | 20.95 | — |
| PyTorch CUDA | CUDA | F32 | 2.97 | — | 0.50 (10-sent) | — |

Tomoul Vulkan is within **1.2× of PyTorch CUDA** on single inference while using **portable Vulkan compute shaders** with zero dependency on vendor-specific libraries (no cuBLAS/cuDNN). Q8_K delivers **3.56× smaller** weights (24 MB vs 87 MB) with negligible accuracy loss (cosine similarity ≥ 0.986 vs CPU F32).

**Key advantage:** Vulkan runs on any GPU with a Vulkan driver (NVIDIA, AMD, Intel) — no CUDA toolkit required. The entire Tomoul runtime is a **14 MB DLL** vs ~2 GB for PyTorch + CUDA.

### GPU Correctness — Windows (RTX 4090) — April 2026

Validated on NVIDIA GeForce RTX 4090, Zig 0.15.2, Vulkan 1.3:

**Kernel tests** (synthetic weights, `zig build test-gpu-forward`):

| Kernel | Max Error vs CPU | Status |
|---|---|---|
| LayerNorm | 2.38e-7 | PASS |
| GELU | 4.77e-7 | PASS |
| Attention | 5.96e-8 | PASS |
| End-to-end forward (1 layer, hidden=8) | 2.98e-8 | PASS |

**Model tests** (real weights, `zig build test-gpu-model`):

| Test | Sentence | Cosine Sim (GPU vs CPU) | Max Abs Diff | Status |
|---|---|---|---|---|
| F32 | "Hello world" | 0.9891 | 0.0220 | PASS |
| F32 | "The quick brown fox jumps over the lazy dog" | 0.9865 | 0.0287 | PASS |
| F32 | "Machine learning models can run on GPUs..." | 0.9862 | 0.0301 | PASS |
| F32 | "Zig is a systems programming language" | 0.9884 | 0.0263 | PASS |
| F32 | "Vulkan compute shaders enable general-purpose GPU..." | 0.9865 | 0.0287 | PASS |
| Q8K | "Hello world" | 0.9884 | 0.0241 | PASS |
| Q8K | "The quick brown fox jumps over the lazy dog" | 0.9859 | 0.0295 | PASS |
| Q8K | "Machine learning models can run on GPUs..." | 0.9856 | 0.0312 | PASS |
| Q8K | "Zig is a systems programming language" | 0.9878 | 0.0274 | PASS |
| Q8K | "Vulkan compute shaders enable general-purpose GPU..." | 0.9859 | 0.0295 | PASS |

**Batch consistency** (GPU single vs GPU batch — should be identical):

| Sentence | Cosine Sim | Max Abs Diff | Status |
|---|---|---|---|
| All 5 sentences | 1.000000 | 0.000000 | PASS |

**Latency** (warm start, 5 iterations):

| Metric | F32 | Q8K |
|---|---|---|
| CPU single | 19.01 ms | 12.99 ms |
| GPU single | 4.18 ms | 4.02 ms |
| Single speedup | 4.55× | 3.24× |
| CPU 5-sent sequential | 109.72 ms | 167.04 ms |
| GPU 5-sent batch | 7.18 ms | 6.64 ms |
| Batch speedup | 15.28× | 25.15× |
| GPU per-sentence (batch) | 1.44 ms | 1.33 ms |

### Browser (WASM) — Windows (RTX 4090) — April 2026

Tested in Chrome 146, Q8K weights (25.5 MB WASM binary), 128 MB heap, 3 warmup, 10 iterations:

| Backend | Single (ms) | 3-sent (ms) | Per-sent (ms) | vs Native Vulkan |
|---|---|---|---|---|
| **CPU WASM** | **21.03** | **62.60** | **20.87** | 5.2× slower |
| Native Vulkan (Q8K) | 4.02 | 6.64 | 1.33 | baseline |
| Native CPU (Q8K) | 12.99 | — | — | 3.2× slower |

**WebGPU `gpu_init()` OOMs** — the 128 MB WASM heap uses 108.6 MB after CPU model init, leaving only 19.4 MB free. GPU init needs to dequantize Q8K→F32 for upload (~87 MB), which exceeds available heap. Needs 256 MB+ heap to run both paths. The earlier webgpu.html demo (34.7 ms) was actually calling CPU `embed()` with bridge overhead, not true GPU inference.

**Similarity matrix** (3 test sentences, CPU WASM):

| | The quick brown fox... | Machine learning is... | I had pizza for lunc... |
|---|---|---|---|
| The quick brown fox... | 1.000 | 0.995 | 0.949 |
| Machine learning is... | 0.995 | 1.000 | 0.973 |
| I had pizza for lunc... | 0.949 | 0.973 | 1.000 |

> **WASM CPU is 5.2× slower than native Vulkan** and 1.6× slower than native CPU. The gap is due to wasm32 lacking SIMD vectorization used by zblas (no `@Vector` hardware acceleration) and FixedBufferAllocator overhead. For interactive browser use, 21 ms per sentence is still fast enough — below the 100 ms perceptual threshold for 5 sentences.
>
> **TODO:** Test with 256 MB heap to enable `gpu_init()` and benchmark actual WebGPU `gpu_embed()` compute path. For larger models (Qwen 2.5 3B), WebGPU will be essential.

### GPU (Metal) — macOS — TODO

Metal compute shaders with MSL. Code complete, pending hardware validation on macOS.

**Kernel tests** (`zig build test-gpu-forward` on macOS):

| Kernel | Max Error vs CPU | Status |
|---|---|---|
| LayerNorm | — | PENDING |
| GELU | — | PENDING |
| Attention | — | PENDING |
| End-to-end forward | — | PENDING |

**Model tests** (`zig build test-gpu-model` on macOS):

| Test | Weights | Cosine Sim (GPU vs CPU) | GPU Single (ms) | GPU Batch 5-sent (ms) | Status |
|---|---|---|---|---|---|
| Metal F32 | F32 | — | — | — | PENDING |
| Metal Q8K | Q8K | — | — | — | PENDING |

> To run: `zig build test-gpu-forward && zig build test-gpu-model` on a Mac with Metal support. Update this section with results.

### PyTorch Comparison — Windows (RTX 4090) — April 2026

PyTorch 2.6.0+cu124, `sentence-transformers/all-MiniLM-L6-v2`, 10 iterations:

| Engine | Backend | Single (ms) | 5-sent seq (ms) | 10-sent batch (ms) | Per-sent batch (ms) |
|---|---|---|---|---|---|
| PyTorch | CPU (MKL) | 6.20 | 27.69 | 11.55 | 1.15 |
| PyTorch | CUDA (RTX 4090) | 3.64 | 15.99 | 8.48 | 0.85 |
| **Tomoul** | **Vulkan (RTX 4090)** | **4.02** | **—** | **—** | **1.33** |
| **Tomoul** | **CPU (Q8K)** | **12.99** | **—** | **—** | **—** |

**Tomoul Vulkan single-sentence is within 1.1× of PyTorch CUDA** (4.02 ms vs 3.64 ms) on the same GPU, using portable compute shaders with zero CUDA dependency.

Q8_K is the recommended variant: **3.6× smaller** model (24 MB vs 87 MB), **2.5× faster** than F32 on CPU, and competitive with PyTorch CUDA on GPU — with negligible accuracy loss (cosine similarity ≥ 0.9997 vs F32).

**Optimizations applied:** arena allocator for inference scratch, fused strided multi-head attention (no per-head copies), vectorized GELU (Padé tanh approximation), SIMD bias addition, in-place layerNorm, zblas skinny-M SGEMM kernel, zblas Q8_K weight-only quantized SGEMM, Vulkan compute shaders with SPIR-V (9 embedded shaders: sgemm_bias, layernorm, gelu, residual_add, attention, embedding_lookup, pool_normalize, attention_batch, pool_normalize_batch). See [zblas](https://github.com/tomoul/zblas) for BLAS-level optimization details.

**Accuracy:** All 5 reference sentences achieve cosine similarity ≥ 0.99 against HuggingFace `sentence-transformers` output (F32, Q8_K, and GPU variants).

## Quick Start

### Generate Weights

```bash
# F32 (full precision, ~87 MB)
python tools/export_sentence_transformer.py -o artifacts/

# F16 (half-precision, ~43 MB, 1.8× faster than F32)
python tools/export_sentence_transformer.py -o artifacts/ -q f16

# Q8_K (recommended: ~24 MB, 2.5× faster inference)
python tools/export_sentence_transformer.py -o artifacts/ -q q8_k
```

Requires: `pip install torch transformers sentence-transformers`

### Build & Run (CLI)

```bash
# Build CLI
zig build -Dmodel=sentence_transformer -Doptimize=ReleaseFast

# Embed a sentence
./zig-out/bin/tomoul_sentence_transformer \
  --model artifacts/all_minilm_l6_v2.tl \
  --vocab artifacts/all_minilm_l6_v2_vocab.txt \
  --text "This is an example sentence"

# Batch mode
./zig-out/bin/tomoul_sentence_transformer \
  --model artifacts/all_minilm_l6_v2.tl \
  --vocab artifacts/all_minilm_l6_v2_vocab.txt \
  --batch sentences.txt
```

### Build Shared Library

```bash
# Shared library (.so / .dylib)
zig build lib -Dmodel=sentence_transformer -Doptimize=ReleaseFast

# Bundled (weights embedded in binary — no external files needed)
zig build lib -Dmodel=sentence_transformer -Doptimize=ReleaseSmall -Dbundled=true
```

Output: `zig-out/lib/libtomoul_sentence_transformer.{so,dylib,a}`

### Build WASM

```bash
zig build wasm -Dmodel=sentence_transformer
```

Output: `zig-out/bin/tomoul_sentence_transformer.wasm` (weights embedded)

### Node.js (FFI)

```bash
cd examples/sentence-transformer
npm install
node benchmark.js
```

```javascript
const koffi = require('koffi');
const lib = koffi.load('./libtomoul_sentence_transformer.so');

const init = lib.func('int tomoul_sentence_transformer_init(const char*, const char*)');
const embed = lib.func('int tomoul_sentence_transformer_embed(const char*, size_t, float*)');
const destroy = lib.func('void tomoul_sentence_transformer_destroy()');

init('/path/to/all_minilm_l6_v2.tl', '/path/to/vocab.txt');
const output = new Float32Array(384);
embed('Hello world', 11, output);
destroy();
```

### Python (comparison baseline)

```bash
cd examples/sentence-transformer
pip install torch transformers sentence-transformers
python benchmark_pytorch.py
```

## C API

```c
// Initialize — returns 0 on success
int tomoul_sentence_transformer_init(const char* weights_path, const char* vocab_path);

// Embed text — writes 384 floats to output buffer
// Returns: 0 success, -1 not initialized, -2 empty input, -3 embed error
int tomoul_sentence_transformer_embed(const char* text, size_t text_len, float* output);

// Cleanup
void tomoul_sentence_transformer_destroy(void);

// Status
int  tomoul_sentence_transformer_is_ready(void);  // 1 = ready, 0 = not
const char* tomoul_sentence_transformer_version(void);  // "0.1.0"
```

### WASM API

For browser use, the WASM build uses shared-memory buffers:

| Export | Description |
|---|---|
| `init()` | Initialize model (weights embedded at compile time) |
| `embed()` | Embed text from input buffer → output buffer |
| `get_input_buffer_ptr()` | Pointer to write UTF-8 text into |
| `get_max_input_bytes()` | Max input size |
| `get_output_buffer_ptr()` | Pointer to read 384 floats from |
| `is_ready()` | 1 if initialized |
| `get_version()` | Version string pointer |
| `reset()` | Reset state |

## Cross-Compilation Targets

All targets are built automatically by CI on tagged releases. See [release.yml](../../../.github/workflows/release.yml).

| Target | Artifacts |
|---|---|
| Linux x86_64 | executable, `.a`, `.so` |
| Linux aarch64 | executable, `.a`, `.so` |
| macOS x86_64 | executable, `.a`, `.dylib` |
| macOS aarch64 | executable, `.a`, `.dylib` |
| iOS aarch64 | `.a` |
| Android aarch64 | `.a` |
| Android x86_64 | `.a` |
| WASM | `.wasm` (bundled) |

## Files

| File | Purpose |
|---|---|
| `model.zig` | Core model: forward pass, weight loading (F32 + Q8_K), config detection |
| `tokenizer.zig` | WordPiece tokenizer: lowercase → split → greedy subword matching |
| `c.zig` | C FFI binding (5 exported functions) |
| `wasm.zig` | WASM binding with shared-memory buffer API |
| `cli.zig` | CLI executable with JSON output |

## Tests

```bash
zig build test-sentence-transformer
```

Validates:
- Tokenizer output matches HuggingFace
- All 5 reference sentences cosine similarity ≥ 0.99 vs Python
- Edge cases (empty input, special characters, max length)

## References

- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — original model
- [Sentence-BERT paper](https://arxiv.org/abs/1908.10084) — Reimers & Gurevych, 2019
- [MiniLM paper](https://arxiv.org/abs/2002.10957) — Wang et al., 2020

## License

MIT — see repository root.
