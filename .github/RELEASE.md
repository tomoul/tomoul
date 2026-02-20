# Tomoul CI/CD

Automated build and release infrastructure for Tomoul models.

## Release Workflow

The [release.yml](workflows/release.yml) workflow triggers on:
- **All models**: `v*` tags (e.g., `v1.0.0`) - builds every model in registry
- **Single model**: `{model}-v*` tags (e.g., `silero_vad-v1.0.0`) - builds only that model
- **Manual dispatch**: Via GitHub Actions UI

### Tag-Based Release

```bash
# Release a single model (9 jobs instead of 9 × N models)
git tag silero_vad-v1.2.0
git push origin silero_vad-v1.2.0

# Release all models (rare, major releases)
git tag v2.0.0
git push origin v2.0.0
```

This prevents 900+ CI jobs when you have 100+ models.

## Build Matrix

For each model in `model_registry.zig`, we build:

### WebAssembly
| Platform | Arch | Target | Output |
|----------|------|--------|--------|
| Web | wasm32 | `wasm32-freestanding` | `tomoul_{model}_web_wasm32_bundled.wasm` |

### Executables
| Platform | Arch | Target | Output |
|----------|------|--------|--------|
| Linux | x86_64 | `x86_64-linux` | `tomoul_linux_x86_64_bundled` |
| Linux | aarch64 | `aarch64-linux` | `tomoul_linux_aarch64_bundled` |
| macOS | x86_64 | `x86_64-macos` | `tomoul_mac_x86_64_bundled` |
| macOS | aarch64 | `aarch64-macos` | `tomoul_mac_aarch64_bundled` |

### Static Libraries (.a)
| Platform | Arch | Target | Output |
|----------|------|--------|--------|
| Linux | x86_64 | `x86_64-linux` | `libtomoul_{model}_linux_x86_64.a` |
| Linux | aarch64 | `aarch64-linux` | `libtomoul_{model}_linux_aarch64.a` |
| macOS | x86_64 | `x86_64-macos` | `libtomoul_{model}_mac_x86_64.a` |
| macOS | aarch64 | `aarch64-macos` | `libtomoul_{model}_mac_aarch64.a` |
| iOS | aarch64 | `aarch64-ios` | `libtomoul_{model}_ios_aarch64.a` |
| Android | arm64-v8a | `aarch64-linux-android` | `libtomoul_{model}_android_arm64-v8a.a` |
| Android | x86_64 | `x86_64-linux-android` | `libtomoul_{model}_android_x86_64.a` |

### Shared Libraries
| Platform | Arch | Target | Output |
|----------|------|--------|--------|
| Linux | x86_64 | `x86_64-linux` | `libtomoul_{model}_linux_x86_64.so` |
| Linux | aarch64 | `aarch64-linux` | `libtomoul_{model}_linux_aarch64.so` |
| macOS | x86_64 | `x86_64-macos` | `libtomoul_{model}_mac_x86_64.dylib` |
| macOS | aarch64 | `aarch64-macos` | `libtomoul_{model}_mac_aarch64.dylib` |

### C Headers
| Output | Description |
|--------|-------------|
| `tomoul_{model}.h` | FFI header for C/C++ integration |

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

## Model Weight Generation

Model weights (`.tl` files) are **generated at build time**, not stored in git:

```yaml
# Job 1: generate-weights
python tools/export_{model}.py  # Downloads from torch.hub, exports to .tl
```

This approach:
- **No binary files in git** - keeps repo small
- **No storage costs** - no Git LFS needed
- **Always fresh** - uses latest upstream weights
- **Reproducible** - same export script, same output

## Files Uploaded to Hugging Face

For each model, the workflow uploads:

```
{hf_repo}/
├── README.md                                         # Auto-generated
├── {model}.tl                                        # Raw model weights
├── include/
│   └── tomoul_{model}.h                              # C header
├── bin/
│   ├── tomoul_{model}_web_wasm32_bundled.wasm        # Browser WASM
│   ├── tomoul_linux_x86_64_bundled                   # Linux x64 CLI
│   ├── tomoul_linux_aarch64_bundled                  # Linux ARM64 CLI
│   ├── tomoul_mac_x86_64_bundled                     # macOS Intel CLI
│   └── tomoul_mac_aarch64_bundled                    # macOS ARM64 CLI
└── lib/
    ├── libtomoul_{model}_linux_x86_64.a              # Linux x64 static
    ├── libtomoul_{model}_linux_x86_64.so             # Linux x64 shared
    ├── libtomoul_{model}_linux_aarch64.a             # Linux ARM64 static
    ├── libtomoul_{model}_linux_aarch64.so            # Linux ARM64 shared
    ├── libtomoul_{model}_mac_x86_64.a                # macOS Intel static
    ├── libtomoul_{model}_mac_x86_64.dylib            # macOS Intel shared
    ├── libtomoul_{model}_mac_aarch64.a               # macOS ARM64 static
    ├── libtomoul_{model}_mac_aarch64.dylib           # macOS ARM64 shared
    ├── libtomoul_{model}_ios_aarch64.a               # iOS ARM64 static
    ├── libtomoul_{model}_android_arm64-v8a.a         # Android ARM64 static
    └── libtomoul_{model}_android_x86_64.a            # Android x64 static
```

### Adding New Models

1. Create the model folder with all files:
   ```
   src/models/{model_name}/
   ├── model.zig   # Model implementation
   ├── wasm.zig    # WebAssembly binding (browser/WASM)
   └── c.zig       # C API binding (native libraries)
   ```

2. Add model to `src/models/registry.zig` (convention over configuration):
   ```zig
   .{
       .name = "new_model",         // Paths are derived from this name!
       .kind = .audio,              // or .text, .vision
       .description = "Model Description",
       .export_symbols = &.{ "init", "process", ... },
       .has_example = false,        // Set to true if you have examples/new-model/
   },
   ```

   **Paths are automatically derived:**
   - `src/models/new_model/wasm.zig` ← wasm binding
   - `src/models/new_model/c.zig` ← c binding
   - `src/models/new_model/model.zig` ← model implementation
   - `artifacts/new_model.tl` ← weights file
   - `tomoul/new-model` ← HuggingFace repo (underscores → dashes)

3. Create export script at `tools/export_new_model.py`:
   - Downloads model from upstream (e.g., torch.hub)
   - Exports weights to `artifacts/new_model.tl`
   - See `tools/export_silero_vad.py` as reference

4. Push a model-specific tag to release:
   ```bash
   git tag new_model-v1.0.0
   git push origin new_model-v1.0.0
   ```

5. The HF repo will be auto-created on first release.

### Source Structure

```
src/
├── artifacts/                 # Each model gets its own folder
│   ├── silero_vad/
│   │   ├── model.zig       # Model implementation (LSTM, layers, etc.)
│   │   ├── wasm.zig        # WASM binding for browser
│   │   └── c.zig           # C API for native libs
│   └── {new_model}/        # Future models follow same pattern
│       ├── model.zig
│       ├── wasm.zig
│       └── c.zig
└── core/                   # Shared inference engine
    ├── tensor.zig          # N-dimensional tensor
    ├── ops.zig             # Neural network operations
    └── loader.zig          # Model weight loading
```

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