# XLM-RoBERTa Punctuation — Punctuation Restoration

Automatic punctuation and capitalization restoration using XLM-RoBERTa transformer models. Supports [oliverguhr/fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) and [oliverguhr/fullstop-punctuation-multilingual-sonar-base](https://huggingface.co/oliverguhr/fullstop-punctuation-multilingual-sonar-base). Runs natively on CPU, WASM — no Python or GPU required.

## Architecture

| Parameter | Large | Sonar-Base |
|---|---|---|
| Base model | XLM-RoBERTa | XLM-RoBERTa |
| Layers | 24 | 12 |
| Hidden dim | 1,024 | 768 |
| Attention heads | 16 | 12 |
| FFN intermediate | 4,096 | 3,072 |
| Vocab size | 250,002 (SentencePiece) | 250,002 (SentencePiece) |
| Output classes | 4 (O, COMMA, PERIOD, QUESTION) | 4 (O, COMMA, PERIOD, QUESTION) |

**Pipeline:** SentencePiece tokenize → XLM-RoBERTa embeddings (word + position) → LayerNorm → N× Transformer blocks (self-attention + FFN) → Token classification head → Punctuation labels → Reconstruct punctuated text.

### Weight Variants

| Model | Variant | File | Description |
|---|---|---|---|
| Large | F32 | `fullstop_punctuation_multilang_large.tl` | Full precision |
| Large | **Q8** | `fullstop_punctuation_multilang_large_q8.tl` | 4× smaller, 1.9× faster |
| Large | Q8_K | `fullstop_punctuation_multilang_large_q8k.tl` | Block-wise 8-bit, better accuracy |
| Large | Q4 | `fullstop_punctuation_multilang_large_q4.tl` | 8× smaller |
| Sonar-Base | Q8 | `fullstop_punctuation_multilingual_sonar_base_q8.tl` | 3.5× faster than large |

## Benchmarks

Measured on AMD CPU (AVX2) + RTX 4090, Q8 quantized, 10 iterations, warm start:

### Sonar-Base (12 layers, 768 hidden)

| Engine | Device | Short 7 words (ms) | Long ~100 words (ms) |
|---|---|---|---|
| **Tomoul** (Zig, Q8) | CPU | **31** | **293** |
| PyTorch (transformers) | CPU (MKL) | 12 | 48 |
| PyTorch (transformers) | CUDA (RTX 4090) | 5 | 10 |

### Large (24 layers, 1024 hidden)

| Engine | Device | Short 7 words (ms) | Long ~100 words (ms) |
|---|---|---|---|
| **Tomoul** (Zig, Q8) | CPU | **147** | **984** |
| PyTorch (transformers) | CPU (MKL) | 37 | 162 |
| PyTorch (transformers) | CUDA (RTX 4090) | 8 | 13 |

Tomoul is currently ~2.7× slower than PyTorch CPU on punctuation due to the large sequence lengths (100+ words = 200+ tokens) exceeding the skinny-M SGEMM fast path. The large-M SGEMM kernel and attention path are the next optimization targets. Sonar-Base is **3.4× faster** than Large with comparable accuracy for most use cases.

## Quick Start

### Generate Weights

```bash
# Large model (24 layers, ~4 GB F32)
python tools/export_xlm_roberta_punctuation.py -o artifacts/ --model large

# Large Q8 quantized (~1 GB)
python tools/export_xlm_roberta_punctuation.py -o artifacts/ --model large -q q8

# Sonar-Base Q8 (~300 MB)
python tools/export_xlm_roberta_punctuation.py -o artifacts/ --model sonar-base -q q8
```

Requires: `pip install torch transformers sentencepiece`

### Build & Run (CLI)

```bash
# Build CLI (Large model)
zig build -Dmodel=fullstop-punctuation-multilang-large --release=fast

# Punctuate text
./zig-out/bin/tomoul_fullstop-punctuation-multilang-large \
  "hello world how are you" \
  artifacts/fullstop_punctuation_multilang_large_q8.tl \
  artifacts/fullstop_punctuation_multilang_large_vocab.txt

# Sonar-Base (3.5× faster, recommended for real-time)
zig build -Dmodel=fullstop-punctuation-multilingual-sonar-base --release=fast

./zig-out/bin/tomoul_fullstop-punctuation-multilingual-sonar-base \
  "hello world how are you" \
  artifacts/fullstop_punctuation_multilingual_sonar_base_q8.tl \
  artifacts/fullstop_punctuation_multilingual_sonar_base_vocab.txt
```

### Build Shared Library

```bash
# Shared library (.so / .dylib)
zig build lib -Dmodel=fullstop-punctuation-multilingual-sonar-base --release=fast
```

Output: `zig-out/lib/libtomoul_fullstop-punctuation-multilingual-sonar-base.{so,dylib,a}`

### Build WASM

```bash
zig build wasm -Dmodel=fullstop-punctuation-multilang-large
```

Output: `zig-out/bin/tomoul_fullstop-punctuation-multilang-large.wasm` (weights embedded)

### C API

```c
#include <stdint.h>
#include <stddef.h>

// Initialize model with weights and vocabulary
int  tomoul_xlm_roberta_punctuation_init(const char* weights_path, const char* vocab_path);

// Process unpunctuated text → punctuated text
// Returns length of output on success, negative error code on failure
int  tomoul_xlm_roberta_punctuation_process(const uint8_t* input, size_t input_len,
                                             uint8_t* output, size_t output_capacity);

// Cleanup
void tomoul_xlm_roberta_punctuation_destroy();

// Status
int         tomoul_xlm_roberta_punctuation_is_ready();
const char* tomoul_xlm_roberta_punctuation_version();
int         tomoul_xlm_roberta_punctuation_quant_format(); // 0=not loaded, 1=F32, 2=Q8, 3=Q4, 4=Q8_K
```

### Node.js (FFI)

```bash
cd examples/punctuation-demo
npm install
node benchmark.js
```

```javascript
const koffi = require('koffi');
const lib = koffi.load('./libtomoul_fullstop-punctuation-multilingual-sonar-base.so');

const init = lib.func('int tomoul_xlm_roberta_punctuation_init(string, string)');
const process = lib.func('int tomoul_xlm_roberta_punctuation_process(_In_ uint8_t*, size_t, _Out_ uint8_t*, size_t)');
const destroy = lib.func('void tomoul_xlm_roberta_punctuation_destroy()');

init('artifacts/fullstop_punctuation_multilingual_sonar_base_q8.tl',
     'artifacts/fullstop_punctuation_multilingual_sonar_base_vocab.txt');

const input = Buffer.from('hello world how are you doing today', 'utf8');
const output = Buffer.alloc(input.length * 2);
const len = process(input, input.length, output, output.length);
console.log(output.toString('utf8', 0, len));
// → "Hello world, how are you doing today?"

destroy();
```
