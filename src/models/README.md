# Tomoul Models

Native ML inference models implemented in Zig. Each model compiles to a standalone shared library, CLI binary, or WASM module — no Python runtime, no GPU required.

## Available Models

| Model | Type | Description | Build Target |
|---|---|---|---|
| [Sentence Transformer](sentence_transformer/) | Text | all-MiniLM-L6-v2 sentence embeddings (384-dim) | `sentence_transformer` |
| [Silero VAD](silero_vad/) | Audio | Voice activity detection (LSTM, real-time) | `silero_vad` |
| [Whisper](whisper/) | Audio | OpenAI Whisper speech-to-text (tiny → large) | `whisper-tiny`, `whisper-base`, ... |
| [XLM-RoBERTa Punctuation](xlm_roberta_punctuation/) | Text | Punctuation restoration (multilingual) | `fullstop-punctuation-multilang-large`, `fullstop-punctuation-multilingual-sonar-base` |

## Build Targets

```bash
# CLI
zig build -Dmodel=<model> --release=fast

# Shared library (.so / .dylib / .a)
zig build lib -Dmodel=<model> --release=fast

# WASM (weights embedded)
zig build wasm -Dmodel=<model>

# Bundled library (weights embedded in binary)
zig build lib -Dmodel=<model> --release=small -Dbundled=true
```

## Benchmarks Summary

Measured on AMD CPU (AVX2) + RTX 4090:

| Model | Tomoul CPU | PyTorch CPU | PyTorch CUDA | Speedup vs PyTorch CPU |
|---|---|---|---|---|
| Sentence Transformer Q8_K (single) | 3.5ms | 3.6ms | 2.6ms | **~1× (matches)** |
| Silero VAD (per chunk) | 0.071ms | 0.204ms | — | **1.8× faster** |
| Punctuation Sonar-Base Q8 (7 words) | 31ms | 12ms | 5ms | 0.4× (slower) |
| Punctuation Large Q8 (7 words) | 147ms | 37ms | 8ms | 0.3× (slower) |
| Whisper Tiny (13.7s audio) | 20,965ms | 409ms | 219ms | 0.02× (no KV-cache) |

Tomoul excels on small, latency-sensitive models (VAD, sentence embeddings) where zblas SIMD kernels dominate. Larger models with long sequences and autoregressive decoding (whisper, punctuation) are active optimization targets.

## Adding a New Model

1. Create a directory under `src/models/<name>/` with:
   - `model.zig` — Model architecture and inference
   - `c.zig` — C-compatible shared library API
   - `cli.zig` — Command-line interface
   - `wasm.zig` — WebAssembly bindings (optional)
   - `README.md` — Documentation

2. Register the model in [registry.zig](registry.zig) with its build configuration.

3. Add weight export script in `tools/export_<name>.py`.
