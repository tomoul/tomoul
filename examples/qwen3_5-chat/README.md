# Qwen3.5-0.8B Chat Example

A zero-dependency web chat UI for Tomoul's Qwen3.5-0.8B inference engine.

## Quick Start

```bash
# From the tomoul project root:
node examples/qwen3_5-chat/server.js
```

Then open **http://localhost:3000** in your browser.

## Options

```bash
node examples/qwen3_5-chat/server.js \
  --port 8080 \
  --weights artifacts/qwen3_5_0.8b_f32.tl \
  --max-tokens 512
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 3000 | Server port |
| `--weights` | `artifacts/qwen3_5_0.8b_q8k.tl` | Model weights file |
| `--tokenizer` | `artifacts/qwen3_5_vocab.bin` | Tokenizer file |
| `--binary` | `zig-out/bin/tomoul_qwen3_5-0.8b.exe` | Inference binary |
| `--max-tokens` | 256 | Default max generation tokens |

## API

### POST /api/generate

```json
{ "prompt": "What is 2+2?", "max_tokens": 256 }
```

Response:

```json
{
  "text": "2+2 equals **4**.",
  "tokens": 42,
  "elapsed_ms": 3200.5,
  "tok_per_s": 13.1
}
```

### GET /api/health

```json
{ "status": "ok", "model": "qwen3.5-0.8b", "weights": "qwen3_5_0.8b_q8k.tl" }
```

## Requirements

- Node.js (any recent version)
- Pre-built `tomoul_qwen3_5-0.8b.exe` binary in `zig-out/bin/`
- Model artifacts in `artifacts/`
