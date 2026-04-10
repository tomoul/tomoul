# Whisper — Speech-to-Text

OpenAI Whisper speech recognition from [openai/whisper](https://github.com/openai/whisper). Runs natively on CPU — no Python or GPU required. Supports multiple model sizes from tiny (39M) to large-v3-turbo (809M).

## Architecture

| Parameter | Tiny | Base | Small | Medium | Large |
|---|---|---|---|---|---|
| Parameters | 39M | 74M | 244M | 769M | 1550M |
| Encoder layers | 4 | 6 | 12 | 24 | 32 |
| Decoder layers | 4 | 6 | 12 | 24 | 32 |
| Hidden dim | 384 | 512 | 768 | 1024 | 1280 |
| Attention heads | 6 | 8 | 12 | 16 | 20 |
| Mel bins | 80 | 80 | 80 | 80 | 80/128 |
| Vocab size | 51,865 | 51,865 | 51,865 | 51,865 | 51,865 |
| Max audio | 30s | 30s | 30s | 30s | 30s |
| Max tokens | 448 | 448 | 448 | 448 | 448 |

Additional variants: `large-v3-turbo` (1280 hidden, 32 encoder + 4 decoder layers, 128 mels), `distil-small.en` (768 hidden, 12 encoder + 4 decoder layers).

**Pipeline:** WAV (16kHz mono) → STFT → Mel spectrogram [80, 3000] → 2× Conv1d (stride 1, stride 2) + GELU → Positional embedding → N× Encoder blocks (self-attention + FFN) → LayerNorm → Audio features [1500, hidden_dim] → Autoregressive decoder (causal self-attention + cross-attention + FFN) → Greedy token generation → BPE token IDs.

## Benchmarks

Measured on AMD CPU + RTX 4090, whisper-tiny, 13.7s audio file, 5 iterations, warm start:

| Engine | Device | Avg (ms) | Notes |
|---|---|---|---|
| PyTorch (openai-whisper) | **CUDA (RTX 4090)** | **219** | fp16, batched attention |
| PyTorch (openai-whisper) | CPU (MKL) | 409 | fp32 |
| **Tomoul** (Zig, ReleaseFast) | CPU | 20,965 | Greedy decode, no KV-cache |

Tomoul's whisper implementation does not yet use KV-cache for the autoregressive decoder, resulting in O(n²) recomputation per generated token. This is the primary optimization target — KV-cache alone is expected to bring CPU performance close to PyTorch. The encoder runs a single forward pass and is already efficient.

## Quick Start

### Generate Weights

```bash
# Tiny (39M, ~150 MB)
python tools/export_whisper.py -o models/ --variant tiny

# Base (74M, ~290 MB)
python tools/export_whisper.py -o models/ --variant base
```

Requires: `pip install torch openai-whisper`

### Build & Run (CLI)

```bash
# Build CLI
zig build -Dmodel=whisper-tiny --release=fast

# Transcribe a WAV file (16kHz mono)
./zig-out/bin/tomoul_whisper-tiny transcribe audio.wav

# Transcribe from pre-computed mel spectrogram
./zig-out/bin/tomoul_whisper-tiny transcribe-mel mel.tl

# Override weights path
./zig-out/bin/tomoul_whisper-tiny transcribe audio.wav --weights models/whisper_tiny_q8_0.tl

# Validate encoder against PyTorch reference
./zig-out/bin/tomoul_whisper-tiny validate-encoder
```

### Build Shared Library

```bash
# Shared library (.so / .dylib)
zig build lib -Dmodel=whisper-tiny --release=fast
```

Output: `zig-out/lib/libtomoul_whisper-tiny.{so,dylib,a}`

### C API

```c
#include <stdint.h>

// Initialize model from weights file (auto-detects variant from filename)
int tomoul_whisper_init(const char* weights_path);

// Transcribe WAV file → token IDs
int tomoul_whisper_transcribe_file(const char* audio_path, uint32_t* output, size_t capacity);

// Transcribe mel spectrogram → token IDs
int tomoul_whisper_transcribe_mel(const float* mel_data, size_t n_mels, size_t n_frames,
                                  uint32_t* output, size_t capacity);

// Encode-only (returns audio features)
int tomoul_whisper_encode(const float* mel_data, size_t n_mels, size_t n_frames,
                          float* output, size_t capacity);

// Cleanup
void tomoul_whisper_destroy();

// Status
int  tomoul_whisper_is_ready();
const char* tomoul_whisper_version();
const char* tomoul_whisper_variant();
```

### Node.js (FFI)

```bash
cd examples/whisper-demo
npm install
node benchmark.js
```

```javascript
const koffi = require('koffi');
const lib = koffi.load('./libtomoul_whisper-tiny.so');

const init = lib.func('int tomoul_whisper_init(string)');
const transcribe = lib.func('int tomoul_whisper_transcribe_file(string, _Out_ uint32_t*, size_t)');
const destroy = lib.func('void tomoul_whisper_destroy()');

init('models/whisper_tiny.tl');

const outputBuffer = Buffer.alloc(1024 * 4);
const numTokens = transcribe('audio.wav', outputBuffer, 1024);
// Read token IDs from outputBuffer

destroy();
```

### HTTP Server (Pure Zig)

```bash
zig build -Dmodel=whisper-tiny --release=fast
./zig-out/bin/whisper-server --weights models/whisper_tiny.tl --port 8080
```

Endpoints:
- `POST /transcribe` — Transcribe audio file (JSON body: `{"audio_path": "path/to/file.wav"}`)
- `GET /status` — Check if model is ready
- `GET /test` — Quick test with default audio file

## Cross-Compile

```bash
# Linux x86_64
zig build lib -Dmodel=whisper-tiny --release=fast -Dtarget=x86_64-linux

# Linux aarch64
zig build lib -Dmodel=whisper-tiny --release=fast -Dtarget=aarch64-linux

# macOS x86_64
zig build lib -Dmodel=whisper-tiny --release=fast -Dtarget=x86_64-macos

# macOS Apple Silicon
zig build lib -Dmodel=whisper-tiny --release=fast -Dtarget=aarch64-macos
```
