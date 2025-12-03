# Tomoul CI/CD

Automated build and release infrastructure for Tomoul models.

## Release Workflow

The [release.yml](workflows/release.yml) workflow triggers on:
- **Version tags**: `v*` (e.g., `v1.0.0`)
- **Manual dispatch**: Via GitHub Actions UI

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

## Current Models

From `model_registry.zig`:

| Model | Kind | HF Repo | Description |
|-------|------|---------|-------------|
| `silero_vad` | audio | [tomoul/silero-vad](https://huggingface.co/tomoul/silero-vad) | Voice Activity Detection |

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

## Using the Artifacts

### Browser (WASM)
```javascript
const response = await fetch('https://huggingface.co/tomoul/silero-vad/resolve/main/bin/tomoul_silero_vad_web_wasm32_bundled.wasm');
const wasm = await WebAssembly.instantiate(await response.arrayBuffer());
wasm.instance.exports.init();
```

### C/C++ (Static Library)
```c
#include "tomoul_silero_vad.h"

int main() {
    tomoul_init();
    float* buf = tomoul_get_input_buffer_ptr();
    // Fill buf with audio samples
    float prob = tomoul_process_audio(512);
    return 0;
}
```

Compile with:
```bash
gcc -o myapp myapp.c -L. -ltomoul_silero_vad_linux_x86_64 -lm
```

### iOS (Static Library)
Link `libtomoul_silero_vad_ios_aarch64.a` in Xcode and include the header.

### Android (Static Library)
Add to your `CMakeLists.txt`:
```cmake
add_library(tomoul STATIC IMPORTED)
set_target_properties(tomoul PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libtomoul_silero_vad_android_${ANDROID_ABI}.a)
```
