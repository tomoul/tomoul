# Tomoul CI/CD

Automated build and release infrastructure for Tomoul models.

## Release Workflow

The [release.yml](workflows/release.yml) workflow triggers on:
- **Version tags**: `v*` (e.g., `v1.0.0`)
- **Manual dispatch**: Via GitHub Actions UI

## Build Matrix

For each model in `model_registry.zig`, we build:

| Platform | Architecture | Target | Output |
|----------|--------------|--------|--------|
| Web | wasm32 | `wasm32-freestanding` | `tomoul_{model}_web_wasm32_bundled.wasm` |
| Linux | x86_64 | `x86_64-linux` | `tomoul_linux_x86_64_bundled` |
| Linux | aarch64 | `aarch64-linux` | `tomoul_linux_aarch64_bundled` |
| macOS | x86_64 | `x86_64-macos` | `tomoul_mac_x86_64_bundled` |
| macOS | aarch64 | `aarch64-macos` | `tomoul_mac_aarch64_bundled` |

## Dynamic Model Discovery

The workflow **automatically discovers** models from `model_registry.zig`:

```yaml
# Job 0: discover-models
models=$(grep -oP '\.name = "\K[^"]+' model_registry.zig)
```

This means:
- **No manual matrix updates** when adding new models
- Just add to `model_registry.zig` and push a tag
- The workflow parses `hf_repo`, `weights_path`, and `description` automatically

## Current Models

From `model_registry.zig`:

| Model | Kind | HF Repo | Description |
|-------|------|---------|-------------|
| `silero_vad` | audio | [tomoul/silero-vad](https://huggingface.co/tomoul/silero-vad) | Voice Activity Detection |

## Artifact Naming Convention

```
tomoul_{model}_{platform}_{arch}_{variant}.{ext}
```

- **model**: Model name from registry (e.g., `silero_vad`)
- **platform**: `web`, `linux`, `mac`
- **arch**: `wasm32`, `x86_64`, `aarch64`
- **variant**: `bundled` (weights embedded) or `dynamic` (weights loaded at runtime)
- **ext**: `.wasm` for web, none for native executables

## Files Uploaded to Hugging Face

For each model, the workflow uploads:

```
{hf_repo}/
├── README.md                                    # Auto-generated
├── {model}.tl                                   # Raw model weights
└── bin/
    ├── tomoul_{model}_web_wasm32_bundled.wasm   # Browser WASM
    ├── tomoul_linux_x86_64_bundled              # Linux x64
    ├── tomoul_linux_aarch64_bundled             # Linux ARM64
    ├── tomoul_mac_x86_64_bundled                # macOS Intel
    └── tomoul_mac_aarch64_bundled               # macOS Apple Silicon
```

## Setup

### GitHub Secrets

Add `HF_TOKEN` to repository secrets:
1. Create token at https://huggingface.co/settings/tokens (write access)
2. Go to repo Settings → Secrets → Actions
3. Add `HF_TOKEN` with your token

### Adding New Models

1. Add model to `model_registry.zig`:
   ```zig
   .{
       .name = "new_model",
       .kind = .audio,  // or .text, .vision
       .wasm_binding = "src/bindings/wasm_new_model.zig",
       .model_module = "src/models/new_model.zig",
       .weights_path = "models/new_model.tl",
       .hf_repo = "tomoul/new-model",
       .description = "Model Description",
       .export_symbols = &.{ "init", "process", ... },
   },
   ```

2. Push a version tag - the workflow auto-discovers your new model!

3. The HF repo will be auto-created on first release.

## Local Release

Use `scripts/release.py` for local builds:

```bash
# Build and upload a specific model
python scripts/release.py --model silero_vad --version v1.0.0

# Build all models
python scripts/release.py --all --version v1.0.0

# Build only (no upload)
python scripts/release.py --model silero_vad --build-only

# WASM only
python scripts/release.py --model silero_vad --wasm-only --build-only
```
