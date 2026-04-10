# Gemma 2B Demo

Text generation using Tomoul's native implementation of **Gemma 2B** (2 billion parameter decoder-only LLM).

## Quick Start

### Prerequisites

```bash
# 1. Export model weights (requires GPU, ~4GB VRAM)
cd ../..
python3 tools/export_gemma.py --variant 2b -o artifacts/

# 2. Build the native library
zig build lib -Dmodel=gemma-2b -Doptimize=ReleaseFast

# 3. Install Node.js dependencies
cd examples/gemma-demo
npm install
```

### Run the Demo Server

```bash
npm start
# Open http://localhost:8010
```

### Run Benchmarks

```bash
# Tomoul (via Node.js FFI)
npm run benchmark

# PyTorch baseline (requires torch + transformers)
pip install torch transformers accelerate
npm run benchmark:python
```

## Architecture

```
┌─────────────────────┐    FFI (koffi)    ┌───────────────────────────────────┐
│    Node.js / Web    │ ◄───────────────► │       libtomoul_gemma-2b          │
│   server.js / UI    │                   │  (Zig, SIMD, GQA, RoPE, KV-cache)│
└─────────────────────┘                   └───────────────────────────────────┘
```

## C API

| Function | Description |
|----------|-------------|
| `tomoul_gemma_init(weights, tokenizer)` | Load model and tokenizer from disk |
| `tomoul_gemma_generate(prompt, len, max_tokens, temperature)` | Generate text |
| `tomoul_gemma_get_output_buffer_ptr()` | Get pointer to output text |
| `tomoul_gemma_get_output_length()` | Get output text length |
| `tomoul_gemma_is_ready()` | Check model readiness (1=ready) |
| `tomoul_gemma_destroy()` | Free all resources |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8010` | Server port |
| `TOMOUL_VARIANT` | `f32` | Weight variant: `f32`, `q8`, `q8k`, `q4` |
| `BENCH_ITERS` | `5` | Benchmark iterations |
| `MAX_TOKENS` | `64` | Max tokens for benchmark |

## Model Details

- **Architecture**: Decoder-only transformer with GQA (grouped-query attention)
- **Parameters**: 2 billion
- **Vocabulary**: 256,000 tokens (SentencePiece BPE)
- **Context**: 8,192 tokens
- **Attention**: 8 query heads, 1 KV head (MQA), 256-dim heads
- **Normalization**: RMSNorm (pre-norm)
- **Activation**: GeGLU (SiLU gate)
- **Position encoding**: RoPE (base 10,000)
