# Sentence Transformer Demo

Sentence embedding generation using Tomoul's native implementation of **all-MiniLM-L6-v2** (384-dimensional embeddings).

## Quick Start

### Prerequisites

```bash
# 1. Export model weights
cd ../..
python3 tools/export_sentence_transformer.py -o artifacts/

# 2. Build the native library
zig build lib -Dmodel=sentence_transformer -Doptimize=ReleaseFast

# 3. Install Node.js dependencies
cd examples/sentence-transformer
npm install
```

### Run the Demo Server

```bash
npm start
# Open http://localhost:8003
```

### Run Benchmarks

```bash
# Tomoul (via Node.js FFI)
npm run benchmark

# PyTorch baseline
pip install sentence-transformers torch
npm run benchmark:python
```

## Architecture

```
┌─────────────────────┐    FFI (koffi)    ┌───────────────────────────────────┐
│    Node.js / Web    │ ◄───────────────► │  libtomoul_sentence_transformer   │
│   server.js / UI    │                   │  (Zig, SIMD, zero-copy)           │
└─────────────────────┘                   └───────────────────────────────────┘
```

## C API

```c
// Initialize with weights + vocabulary
int tomoul_sentence_transformer_init(const char* weights_path, const char* vocab_path);

// Generate 384-dim embedding for text
int tomoul_sentence_transformer_embed(const uint8_t* text, size_t text_len, float output[384]);

// Cleanup
void tomoul_sentence_transformer_destroy();

// Status check
int tomoul_sentence_transformer_is_ready();  // 1 = ready, 0 = not
```

## Files

| File | Description |
|------|-------------|
| `benchmark.js` | Node.js FFI benchmark — single & batch sentences |
| `benchmark_pytorch.py` | Python/PyTorch baseline benchmark |
| `server.js` | HTTP server with REST API |
| `index.html` | Interactive web UI with similarity matrix |
| `package.json` | Node.js dependencies |

## Model Details

- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Layers**: 6 (BERT-based)
- **Vocab**: 30,522 WordPiece tokens
- **Weights**: 87MB (F32), 24MB (Q8_K)
