# Silero VAD WebAssembly Demo

Real-time Voice Activity Detection in the browser using WebAssembly.

## Quick Start

```bash
# From this directory
npm start

# Or specify a port
node server.js 8080

# Or run directly
./server.js 3000
```

Then open http://localhost:8000 in your browser.

## Features

- **Real-time VAD**: Detects speech in audio with low latency
- **WebAssembly**: Fast inference directly in the browser
- **No dependencies**: Pure WebAssembly, no external libraries needed
- **Small footprint**: ~2.2 MB bundled WASM module

## Files

- `index.html` - Demo web interface
- `vad-processor.js` - Audio worklet processor
- `server.js` - Node.js development server
- `package.json` - npm configuration
- `tomoul_silero_vad.wasm` - Bundled VAD model (2.2 MB)
- `speech.wav` - Speech sample (3.2s, 98% speech)
- `silence.wav` - Silence sample (3.0s, pure silence)
- `noise.wav` - White noise sample (3.0s)

## Development

The server serves all files from `examples/silero-vad/`, including the pre-built WASM module.

To rebuild the WASM module:
```bash
# From project root
zig build wasm -Dmodel=silero_vad

# Copy to examples directory
cp zig-out/bin/tomoul_silero_vad.wasm examples/silero-vad/

# Refresh browser to see changes
```

## How it Works

1. **WASM Module**: Silero VAD model compiled to WebAssembly
2. **Audio Worklet**: Processes audio in real-time on separate thread
3. **Main Thread**: Updates UI with VAD results

The audio is processed in 512-sample chunks (32ms at 16kHz), and the model outputs a probability score from 0.0 (silence) to 1.0 (speech).

## Browser Requirements

- Modern browser with WebAssembly support
- Audio Worklet API support (Chrome 66+, Firefox 76+, Safari 14.1+)
- Microphone access for live recording

## Troubleshooting

**WASM fails to load**: Make sure you're using the development server (not `file://`). WASM requires proper MIME types (`application/wasm`).

**CORS errors**: The server includes proper CORS headers. Make sure you're accessing via `http://localhost`, not `127.0.0.1` or other addresses.

**Microphone not working**: Check browser permissions and ensure you're using HTTPS (or localhost for development).

## Building from Source

```bash
# Build WASM module
zig build wasm -Dmodel=silero_vad

# Build native CLI (optional)
zig build -Dmodel=silero_vad -Dbundled=true -Doptimize=ReleaseSmall

# Run tests
zig build test -Dmodel=silero_vad
```

## Performance Benchmarks

Native FFI (shared library) performance on x86_64, measured via Node.js koffi:

| Metric | Value |
|--------|-------|
| Single chunk latency | **0.346ms** avg, 0.330ms p50 |
| Per-chunk (speech) | **0.355ms** (32ms audio → 90× realtime) |
| Per-chunk (silence) | **0.357ms** (32ms audio → 90× realtime) |
| Per-chunk (noise) | **0.359ms** (32ms audio → 90× realtime) |
| Real-time factor | **0.011** (90× realtime) |

**Hardware**: Benchmarked with AVX2 SIMD via zblas.  
**Note**: At hidden_size=128, the LSTM matvec operations (512×128) are small enough that scalar and SIMD paths show identical throughput — the model is dominated by STFT and Conv1d encoder layers.

### Running the benchmark

```bash
# Build the native shared library
cd /path/to/tomoul
zig build lib -Dmodel=silero_vad --release=fast

# Run benchmark (default 50 runs per WAV file)
cd examples/silero-vad
npm install
node benchmark.js

# Customize iteration count
BENCH_RUNS=20 node benchmark.js
```

## License

MIT License - See root LICENSE file
