# Tomoul

A minimalist AI inference engine written in Zig. No Python runtime, no ONNX, no containers — just weights on disk and a binary that runs.

---

## What is this?

Most AI tooling has a packaging problem. You train a model in PyTorch, and now to ship it you either carry a 4 GB Python environment around or fight with ONNX/TensorRT until something works. Neither option is great if you're building a web app, a mobile SDK, or a server that needs to start cold in under a second.

Tomoul is a different approach. The model architectures are implemented directly in Zig. You export weights from HuggingFace once using the Python scripts in `tools/`, and from that point forward everything is pure Zig — no interpreter, no GC, no dynamic loading. The same code compiles to a native binary on Linux/macOS, a `.wasm` for the browser, or a static lib for iOS/Android.

The focus isn't on LLMs (though Whisper is in progress). It's on the models that sit around LLMs and do the unglamorous work: detecting when someone is actually talking, fixing punctuation on raw transcripts, generating embeddings. These are the models that end up in every production speech pipeline and are somehow always the hardest part to deploy.

---

## Models

| Model | Architecture | Status | Size |
|-------|-------------|--------|------|
| Silero VAD | LSTM | ✅ Ready | ~2.2 MB WASM |
| XLM-RoBERTa Punctuation | Transformer | ✅ Ready | 2.1 GB weights |
| Whisper (tiny, distil-small, large-v3-turbo) | Encoder-Decoder Transformer | 🚧 In Progress | varies |

---

## Architecture

Three layers, each with a clear job:

```
Layer 3 — Stack      (src/models/)       VAD, Punctuation, Whisper
Layer 2 — Bridge     (tools/)            Python export scripts: PyTorch → .tl
Layer 1 — Core       (src/core/)         Tensor, ops, loader, quantization
```

The `.tl` format is a simple binary container — magic bytes, dimensions, raw floats. Nothing exotic. You can read `tools/tl_format.py` in about five minutes.

The core ops (`ops.zig`) support optional OpenBLAS or the bundled pure-Zig zblas for matrix math. WASM builds use the Zig fallback automatically.

---

## Quick Start

You need [Zig 0.13.x](https://ziglang.org/download/).

```bash
# Run Silero VAD on an audio file
zig build -Dmodel=silero_vad
./zig-out/bin/tomoul_silero_vad speech.wav

# Build the WASM module and try it in the browser
zig build wasm -Dmodel=silero_vad
cp zig-out/bin/tomoul_silero_vad.wasm examples/silero-vad/
cd examples/silero-vad && npm start
# → http://localhost:8000

# Punctuation restoration demo
cd examples/punctuation-demo && npm install && npm start
# → http://localhost:8001
```

For faster matrix ops on native builds, pass `-Dblas=true` (requires `libopenblas-dev`):

```bash
zig build -Dmodel=whisper_tiny -Dblas=true
```

Or use the bundled zblas (no external deps, still fast):

```bash
zig build -Dmodel=whisper_tiny -Dzblas=true
```

---

## Exporting Weights

The `tools/` directory has a script for each model. They pull from HuggingFace and write a `.tl` file:

```bash
pip install -r tools/requirements.txt

python tools/export_silero_vad.py       # → artifacts/silero_vad.tl
python tools/export_whisper.py          # → models/whisper_tiny.tl
python tools/export_transformer.py      # → artifacts/fullstop_punctuation_multilang_large.tl
```

Pre-exported artifacts for the smaller models are already in `artifacts/` and `models/` if you just want to run things.

---

## Quantization

INT8 and INT4 quantized versions of models are supported. The quantized artifacts are in `artifacts/` (look for `_q8.tl`, `_q4.tl`). They're notably smaller and run faster on CPU:

```bash
python tools/export_quantized.py --model whisper_tiny --bits 8
```

---

## C API

Every model exposes a C-compatible API so you can call it from any language. The headers are in `release/include/`. Example for punctuation:

```c
#include "tomoul_fullstop-punctuation-multilang-large.h"

tomoul_xlm_roberta_punctuation_init("weights.tl", "vocab.txt");

char output[1024];
tomoul_xlm_roberta_punctuation_process(
    "hello world how are you",
    23,
    output,
    sizeof(output)
);
// output → "Hello world, how are you?"

tomoul_xlm_roberta_punctuation_destroy();
```

Pre-built shared libraries for Linux x86_64, Linux aarch64, macOS x86_64, and macOS arm64 are in `release/lib/`.

---

## Directory Layout

```
src/
  core/          Tensor, ops, loader, attention, quantization, audio
  models/
    silero_vad/
    xlm_roberta_punctuation/
    whisper/
tools/           Python exporters (HuggingFace → .tl)
artifacts/       Pre-exported model weights
models/          Whisper model weights
examples/
  silero-vad/    Browser VAD demo (WASM)
  punctuation-demo/  Browser punctuation demo (native lib via Node)
  whisper/       Python validation script
release/         Prebuilt binaries, shared libs, headers
docs/            Implementation roadmap, BLAS notes
```

---

## Why Zig?

A few practical reasons:

- Compiles to WASM without a separate toolchain
- Produces static binaries with zero runtime dependencies
- The code is readable enough that porting a new model architecture is not a weekend-destroying experience
- No allocator surprises — memory ownership is explicit everywhere

If you've looked at `llama.cpp` and thought "I get what this is doing but I wouldn't want to add a model," Zig is a bit easier to work in for this kind of thing.

---

## Contributing

The most useful thing is adding model architectures. The pattern is:

1. Write `tools/export_yourmodel.py` to dump weights to `.tl`
2. Write `src/models/yourmodel/model.zig` using the ops in `src/core/ops.zig`
3. Add a `cli.zig` for the command-line interface
4. Register in `src/models/registry.zig`

The ops in `ops.zig` already cover matmul, attention, LayerNorm, RoPE, GELU, sigmoid, softmax, LSTM cells, and convolutions. Most Transformer variants can be assembled from those without writing any new math.

---

## License

MIT
