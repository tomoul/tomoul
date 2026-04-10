# Silero VAD — Voice Activity Detection

Real-time voice activity detection from [snakers4/silero-vad](https://github.com/snakers4/silero-vad). Runs natively on CPU, WASM, iOS, Android — no Python or GPU required.

## Architecture

| Parameter | Value |
|---|---|
| Base model | LSTM-based VAD |
| Input | 512 samples @ 16kHz (32ms chunks) |
| Context buffer | 64 samples carried between chunks |
| STFT | Learned basis [258, 1, 256], hop=128 |
| Encoder | 4× Conv1d (strides 1, 2, 2, 1) |
| LSTM | hidden_size=128, 4 gates packed |
| Decoder | ReLU → 1×1 Conv → Sigmoid |
| Output | Speech probability [0.0 – 1.0] |
| Weights | ~2.2 MB |

**Pipeline:** Context prepend (64+512=576 samples) → Learned STFT (hop=128, pad_right=64) → Conv encoder (128→64→64→128, strides 1,2,2,1) → LSTM (hidden_size=128) → ReLU → 1×1 Conv decoder → Sigmoid → speech probability.

## Benchmarks

Measured on AMD CPU (AVX2), 20 iterations, warm start:

| Metric | Tomoul (zblas SIMD) | PyTorch CPU (JIT) |
|---|---|---|
| Single chunk latency | **0.071ms** avg | 0.204ms avg |
| Per-chunk (speech) | **0.087ms** (367× RT) | 0.161ms (200× RT) |
| Per-chunk (silence) | **0.091ms** (353× RT) | 0.157ms (205× RT) |
| Per-chunk (noise) | **0.094ms** (343× RT) | 0.153ms (211× RT) |
| Real-time factor | **0.0028** (367× RT) | 0.005 (200× RT) |

Tomoul is **1.8× faster** than PyTorch on Silero VAD thanks to SIMD-optimized Conv1d kernel with pre-repacked weights (zblas). The STFT, encoder conv layers, and bias additions all run through AVX2-vectorized paths. The LSTM matvec (512×128) uses zblas sgemv with SIMD dot products.

## Quick Start

### Generate Weights

```bash
python tools/export_silero_vad.py -o artifacts/
```

### Build & Run (CLI)

```bash
# Build CLI
zig build -Dmodel=silero_vad -Doptimize=ReleaseFast

# Process a WAV file (16kHz mono PCM)
./zig-out/bin/tomoul_silero_vad \
  --model artifacts/silero_vad.tl \
  --input audio.wav
```

### Build Shared Library

```bash
# Shared library (.so / .dylib)
zig build lib -Dmodel=silero_vad --release=fast

# Bundled (weights embedded in binary — no external files needed)
zig build lib -Dmodel=silero_vad -Doptimize=ReleaseSmall -Dbundled=true
```

Output: `zig-out/lib/libtomoul_silero_vad.{so,dylib,a}`

### Build WASM

```bash
zig build wasm -Dmodel=silero_vad
```

Output: `zig-out/bin/tomoul_silero_vad.wasm` (weights embedded, ~2.2 MB)

### Node.js (FFI)

```bash
cd examples/silero-vad
npm install
node benchmark.js
```

```javascript
const koffi = require('koffi');
const lib = koffi.load('./libtomoul_silero_vad.so');

const init = lib.func('bool tomoul_vad_init(uint8_t*, size_t)');
const process = lib.func('float tomoul_vad_process(float*, size_t)');
const reset = lib.func('void tomoul_vad_reset()');
const free = lib.func('void tomoul_vad_free()');

const weights = fs.readFileSync('artifacts/silero_vad.tl');
init(weights, weights.length);

const chunk = new Float32Array(512); // 32ms @ 16kHz
const probability = process(chunk, 512); // 0.0–1.0
reset();  // call between audio streams
free();
```

## C API

```c
// Initialize — returns true on success
// Bundled mode: pass NULL, 0 to use embedded weights
// Lite mode: pass weights pointer and length
bool tomoul_vad_init(const uint8_t* weights_ptr, size_t weights_len);

// Process audio chunk — returns speech probability [0.0, 1.0]
// Negative values indicate errors: -1.0 = not initialized, -2.0 = invalid input
float tomoul_vad_process(const float* audio_ptr, size_t samples_len);

// Reset LSTM state and context buffer (call between audio streams)
void tomoul_vad_reset();

// Free all resources
void tomoul_vad_free();
```

### WASM API

For browser use, the WASM build uses shared-memory buffers:

| Export | Description |
|---|---|
| `init()` | Initialize model (weights embedded at compile time) |
| `process_audio(num_samples)` | Process chunk from input buffer → speech probability |
| `get_input_buffer_ptr()` | Pointer to write f32 audio samples into |
| `reset_states()` | Reset LSTM state and context buffer |
| `is_ready()` | 1 if initialized |
| `get_version()` | Version string pointer |

## Cross-Compilation Targets

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
| `model.zig` | Core model: forward pass (STFT → Conv encoder → LSTM → decoder), weight loading, state management |
| `c.zig` | C FFI binding (4 exported functions), supports bundled and lite modes |
| `wasm.zig` | WASM binding with shared-memory buffer API (4 MB fixed heap) |
| `cli.zig` | CLI executable for WAV file processing |

## Tests

```bash
zig build test -Dmodel=silero_vad
```

## References

- [silero-vad](https://github.com/snakers4/silero-vad) — Original model by Silero Team
- [Browser demo](../../examples/silero-vad/) — Real-time VAD in the browser via WebAssembly
