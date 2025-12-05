#!/usr/bin/env python3
"""
Tomoul Release Script

Builds artifacts and uploads them to Hugging Face.

Usage:
    python scripts/release.py --model silero_vad --version v1.0.0
    python scripts/release.py --all --version v1.0.0
    python scripts/release.py --build-only  # Just build, don't upload

Requirements:
    pip install huggingface_hub

Setup:
    1. Create HF token: https://huggingface.co/settings/tokens
    2. Login: huggingface-cli login
    3. Create repo: huggingface-cli repo create tomoul/silero-vad --type model
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def derive_paths(model_name):
    """
    Derive all paths from model name using conventions.

    Given model name "silero_vad", derive:
      - wasm_binding  → src/models/silero_vad/wasm.zig
      - c_binding     → src/models/silero_vad/c.zig
      - model_module  → src/models/silero_vad/model.zig
      - cli_module    → src/models/silero_vad/cli.zig
      - weights_path  → artifacts/silero_vad.tl
      - example_dir   → examples/silero-vad (underscores → dashes)
      - hf_repo       → tomoul/silero-vad (underscores → dashes)
    """
    dashed_name = model_name.replace('_', '-')
    return {
        "wasm_binding": f"src/models/{model_name}/wasm.zig",
        "c_binding": f"src/models/{model_name}/c.zig",
        "model_module": f"src/models/{model_name}/model.zig",
        "cli_module": f"src/models/{model_name}/cli.zig",
        "weights_path": f"artifacts/{model_name}.tl",
        "example_dir": f"examples/{dashed_name}",
        "hf_repo": f"tomoul/{dashed_name}",
    }


def parse_model_registry():
    """
    Parse model_registry.zig to extract model configurations.
    This is the single source of truth for all model metadata.

    Paths are derived from model name using conventions (not stored in registry).
    """
    registry_path = Path(__file__).parent.parent / "src" / "models" / "registry.zig"
    content = registry_path.read_text()

    models = {}

    # Remove single-line comments first
    content_no_comments = re.sub(r'//[^\n]*', '', content)

    # Find the models array content
    models_match = re.search(r'pub const models = \[_\]ModelConfig\{(.*?)\};', content_no_comments, re.DOTALL)
    if not models_match:
        return models

    models_content = models_match.group(1)

    # Split into individual model blocks - match from .{ to },
    # Need to handle nested braces in export_symbols
    depth = 0
    current_block = ""
    blocks = []

    for char in models_content:
        if char == '{':
            depth += 1
            current_block += char
        elif char == '}':
            depth -= 1
            current_block += char
            if depth == 0 and current_block.strip():
                blocks.append(current_block)
                current_block = ""
        else:
            if depth > 0:
                current_block += char

    for block in blocks:
        name_match = re.search(r'\.name\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        name = name_match.group(1)

        # Parse fields from registry
        description_match = re.search(r'\.description\s*=\s*"([^"]*)"', block)
        description = description_match.group(1) if description_match else ""

        # export_symbols spans multiple lines with nested braces
        symbols_match = re.search(r'\.export_symbols\s*=\s*&\.{([^}]+)}', block, re.DOTALL)
        symbols = []
        if symbols_match:
            symbols = re.findall(r'"([^"]+)"', symbols_match.group(1))

        # Derive paths from model name (convention over configuration)
        paths = derive_paths(name)

        # Check for explicit path overrides in the registry
        for field in ['wasm_binding', 'c_binding', 'model_module', 'cli_module', 'weights_path', 'hf_repo']:
            override_match = re.search(rf'\.{field}\s*=\s*"([^"]+)"', block)
            if override_match:
                paths[field] = override_match.group(1)

        # Check for has_cli flag
        has_cli = bool(re.search(r'\.has_cli\s*=\s*true', block))

        # Check for supports_bundled flag (defaults to true)
        supports_bundled = not bool(re.search(r'\.supports_bundled\s*=\s*false', block))

        # Check for release_mode flag (defaults to small)
        release_mode = "small"
        if re.search(r'\.release_mode\s*=\s*\.fast', block):
            release_mode = "fast"

        models[name] = {
            "hf_repo": paths["hf_repo"],
            "weights": paths["weights_path"],
            "description": description,
            "export_symbols": symbols,
            "c_binding": paths["c_binding"],
            "cli_module": paths["cli_module"],
            "has_cli": has_cli,
            "supports_bundled": supports_bundled,
            "release_mode": release_mode,
        }

    return models


# Parse model registry from Zig source (single source of truth)
MODELS = parse_model_registry()

# Build targets for executables
TARGETS = [
    # (platform, arch, zig_target, output_suffix)
    ("web", "wasm32", "wasm32-freestanding", ".wasm"),
    ("linux", "x86_64", "x86_64-linux", ""),
    ("linux", "aarch64", "aarch64-linux", ""),
    ("mac", "x86_64", "x86_64-macos", ""),
    ("mac", "aarch64", "aarch64-macos", ""),
]

# Library targets (platform, arch, zig_target, static_ext, shared_ext)
LIB_TARGETS = [
    ("linux", "x86_64", "x86_64-linux", ".a", ".so"),
    ("linux", "aarch64", "aarch64-linux", ".a", ".so"),
    ("mac", "x86_64", "x86_64-macos", ".a", ".dylib"),
    ("mac", "aarch64", "aarch64-macos", ".a", ".dylib"),
]


def run(cmd, cwd=None):
    """Run a command and return output."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    return result.stdout.strip()


def build_wasm(model_name, output_dir):
    """Build WebAssembly artifact."""
    print(f"\n[BUILD] WASM for {model_name}")

    result = run(f"zig build wasm -Dmodel={model_name}")
    if result is None:
        return False

    src = Path(f"zig-out/bin/tomoul_{model_name}.wasm")
    if src.exists():
        dst = output_dir / f"tomoul_{model_name}_web_wasm32_bundled.wasm"
        shutil.copy(src, dst)
        print(f"  -> {dst}")
        return True
    else:
        print(f"  ERROR: {src} not found")
        return False


def build_native(model_name, platform, arch, zig_target, suffix, output_dir):
    """Build native artifact."""
    print(f"\n[BUILD] {platform}/{arch} for {model_name}")

    # Check if model supports bundled weights
    model_info = MODELS.get(model_name, {})
    supports_bundled = model_info.get('supports_bundled', True)
    variant = "bundled" if supports_bundled else "lite"

    # Use release_mode from model config (defaults to small)
    # small: ReleaseSmall - smaller binary, slightly slower (~15% for large models)
    # fast: ReleaseFast - larger binary, faster execution (recommended for large transformers)
    release_mode = model_info.get('release_mode', 'small')
    optimize = "ReleaseFast" if release_mode == "fast" else "ReleaseSmall"

    bundled_flag = "-Dbundled=true" if supports_bundled else ""
    result = run(f"zig build -Dmodel={model_name} -Dtarget={zig_target} -Doptimize={optimize} {bundled_flag}".strip())
    if result is None:
        return False

    # Check for model-specific executable
    src_specific = Path(f"zig-out/bin/tomoul_{model_name}")
    src_generic = Path("zig-out/bin/tomoul")

    src = src_specific if src_specific.exists() else (src_generic if src_generic.exists() else None)

    if src:
        dst = output_dir / f"tomoul_{model_name}_{platform}_{arch}_{variant}{suffix}"
        shutil.copy(src, dst)
        # Make executable
        if not suffix:
            os.chmod(dst, 0o755)
        print(f"  -> {dst}")
        return True
    else:
        print(f"  WARNING: No executable found (model may be library-only)")
        return False


def build_library(model_name, platform, arch, zig_target, static_ext, shared_ext, lib_dir, bundled=True):
    """Build static and shared libraries using the build system.

    Args:
        bundled: If True, embed model weights in the library (self-contained).
                 If False, build "lite" library (user must load weights separately).
    """
    mode = "bundled" if bundled else "lite"
    print(f"\n[BUILD] Libraries {platform}/{arch} for {model_name} ({mode})")

    # Use release_mode from model config (defaults to small)
    model_info = MODELS.get(model_name, {})
    release_mode = model_info.get('release_mode', 'small')
    optimize = "ReleaseFast" if release_mode == "fast" else "ReleaseSmall"

    bundled_flag = "-Dbundled=true" if bundled else ""
    result = run(f"zig build lib -Dmodel={model_name} -Dtarget={zig_target} -Doptimize={optimize} {bundled_flag}".strip())
    if result is None:
        print(f"  WARNING: Library build failed for {platform}/{arch}")
        return

    # Copy static library
    static_src = Path(f"zig-out/lib/libtomoul_{model_name}{static_ext}")
    if static_src.exists():
        static_dst = lib_dir / f"libtomoul_{model_name}_{platform}_{arch}{static_ext}"
        shutil.copy(static_src, static_dst)
        print(f"  -> {static_dst}")
    else:
        print(f"  WARNING: Static lib {static_src} not found")

    # Copy shared library
    shared_src = Path(f"zig-out/lib/libtomoul_{model_name}{shared_ext}")
    if shared_src.exists():
        shared_dst = lib_dir / f"libtomoul_{model_name}_{platform}_{arch}{shared_ext}"
        shutil.copy(shared_src, shared_dst)
        print(f"  -> {shared_dst}")
    else:
        print(f"  WARNING: Shared lib {shared_src} not found")


def parse_c_binding_functions(c_binding_path):
    """Parse export fn declarations from a C binding Zig file."""
    functions = []
    if not c_binding_path or not Path(c_binding_path).exists():
        return functions

    content = Path(c_binding_path).read_text()

    # Match: export fn name(params) return_type { or export fn name(params) void {
    # Also capture the doc comment above
    # Note: params can span multiple lines, return type can be complex like [*:0]const u8
    pattern = re.compile(
        r'((?:///[^\n]*\n)*)' +  # Optional doc comments
        r'export fn (\w+)\s*\(([^)]*(?:\n[^)]*)*)\)\s*([^\{]+?)\s*\{',
        re.MULTILINE | re.DOTALL
    )

    for match in pattern.finditer(content):
        doc_comment = match.group(1)
        func_name = match.group(2)
        params_str = match.group(3).strip()
        return_type = match.group(4)

        # Parse parameters
        params = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    # Zig params: name: type
                    # Split on first colon only (type may contain : like [*:0]const)
                    parts = param.split(':', 1)
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        param_type = parts[1].strip()
                        params.append((param_name, param_type))

        # Clean up doc comment
        doc_lines = []
        for line in doc_comment.strip().split('\n'):
            line = line.strip()
            if line.startswith('///'):
                doc_lines.append(line[3:].strip())

        functions.append({
            'name': func_name,
            'params': params,
            'return_type': return_type,
            'doc': ' '.join(doc_lines),
        })

    return functions


def zig_type_to_c(zig_type):
    """Convert Zig type to C type."""
    zig_type = zig_type.strip()

    type_map = {
        'bool': 'bool',
        'void': 'void',
        'f32': 'float',
        'f64': 'double',
        'u8': 'uint8_t',
        'u16': 'uint16_t',
        'u32': 'uint32_t',
        'u64': 'uint64_t',
        'i8': 'int8_t',
        'i16': 'int16_t',
        'i32': 'int32_t',
        'i64': 'int64_t',
        'usize': 'size_t',
        'isize': 'ptrdiff_t',
        'c_int': 'int',
    }

    # Handle null-terminated pointer: [*:0]const u8 -> const char*
    if zig_type.startswith('[*:0]const '):
        inner = zig_type[11:].strip()
        if inner == 'u8':
            return 'const char*'
        return f'const {zig_type_to_c(inner)}*'

    if zig_type.startswith('[*:0]'):
        inner = zig_type[5:].strip()
        if inner == 'u8':
            return 'char*'
        return f'{zig_type_to_c(inner)}*'

    # Handle many-item pointer: [*]const u8 -> const uint8_t*
    if zig_type.startswith('[*]const '):
        inner = zig_type[9:].strip()
        return f'const {zig_type_to_c(inner)}*'

    if zig_type.startswith('[*]'):
        inner = zig_type[3:].strip()
        return f'{zig_type_to_c(inner)}*'

    return type_map.get(zig_type, zig_type)


def generate_c_header(model_name, include_dir):
    """Generate C header file from C binding Zig file."""
    print(f"\n[HEADER] Generating C header for {model_name}")

    model_info = MODELS.get(model_name, {})
    c_binding_path = model_info.get('c_binding', '')

    # Get the project root
    project_root = Path(__file__).parent.parent
    if c_binding_path:
        c_binding_path = project_root / c_binding_path

    functions = parse_c_binding_functions(c_binding_path)

    # Build function declarations
    func_decls = []
    for func in functions:
        # Build C parameter list
        c_params = []
        for param_name, param_type in func['params']:
            c_type = zig_type_to_c(param_type)
            c_params.append(f'{c_type} {param_name}')

        params_str = ', '.join(c_params) if c_params else 'void'
        return_type = zig_type_to_c(func['return_type'])

        # Add doc comment if present
        if func['doc']:
            func_decls.append(f"/** {func['doc']} */")

        func_decls.append(f"{return_type} {func['name']}({params_str});")
        func_decls.append("")

    func_decls_str = '\n'.join(func_decls)

    # Convert model name to valid C identifier (replace dashes with underscores)
    guard_name = model_name.upper().replace('-', '_')

    header_content = f"""/**
 * Tomoul {model_name} - C API
 * Auto-generated from {c_binding_path.name if c_binding_path else 'model registry'}
 */

#ifndef TOMOUL_{guard_name}_H
#define TOMOUL_{guard_name}_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {{
#endif

{func_decls_str}
#ifdef __cplusplus
}}
#endif

#endif /* TOMOUL_{guard_name}_H */
"""

    header_path = include_dir / f"tomoul_{model_name}.h"
    header_path.write_text(header_content)
    print(f"  -> {header_path}")


def ensure_hf_repo_exists(repo):
    """Create HF repo if it doesn't exist."""
    from huggingface_hub import HfApi, repo_exists

    api = HfApi()
    if not repo_exists(repo, repo_type="model"):
        print(f"  Creating repo: {repo}")
        api.create_repo(repo, repo_type="model", exist_ok=True)
        return True
    return False


def upload_to_hf(model_name, version, upload_dir):
    """Upload artifacts to Hugging Face."""
    model_info = MODELS[model_name]
    repo = model_info["hf_repo"]

    print(f"\n[UPLOAD] Uploading to {repo}")

    # Check if logged in
    result = run("huggingface-cli whoami")
    if result is None or "Not logged in" in str(result):
        print("  ERROR: Not logged in to Hugging Face")
        print("  Run: huggingface-cli login")
        return False

    # Auto-create repo if it doesn't exist
    try:
        ensure_hf_repo_exists(repo)
    except Exception as e:
        print(f"  WARNING: Could not check/create repo: {e}")

    # Upload
    cmd = f'huggingface-cli upload {repo} {upload_dir}/ . --repo-type model --commit-message "Release {version}"'
    result = run(cmd)
    if result is None:
        return False

    print(f"  SUCCESS: https://huggingface.co/{repo}")
    return True


def generate_readme(model_name, output_dir):
    """Generate README.md for HF repo."""
    model_info = MODELS[model_name]
    export_symbols = model_info.get("export_symbols", [])

    # Build export symbols documentation from WASM binding
    symbols_doc = ""
    if export_symbols:
        symbols_doc = "\n## API Reference\n\nExported functions available in the WASM module:\n\n| Function | Description |\n|----------|-------------|\n"
        for sym in export_symbols:
            # Function descriptions are derived from the symbol name
            # Convert snake_case to readable description
            desc = sym.replace('_', ' ').title()
            symbols_doc += f"| `{sym}` | {desc} |\n"

    readme = f'''---
license: mit
tags:
  - voice-activity-detection
  - audio
  - zig
  - wasm
---

# Tomoul {model_info["description"]}

Built with [Tomoul](https://github.com/tomoul/tomoul) - the minimalist AI inference engine in Zig.

## Quick Start (CLI)

Download and run the bundled executable (2.2 MB, includes model weights):

```bash
# Linux x64
wget https://huggingface.co/{model_info["hf_repo"]}/resolve/main/bin/tomoul_{model_name}_linux_x86_64_bundled
chmod +x tomoul_{model_name}_linux_x86_64_bundled
./tomoul_{model_name}_linux_x86_64_bundled audio.wav

# macOS Apple Silicon
wget https://huggingface.co/{model_info["hf_repo"]}/resolve/main/bin/tomoul_{model_name}_mac_aarch64_bundled
chmod +x tomoul_{model_name}_mac_aarch64_bundled
./tomoul_{model_name}_mac_aarch64_bundled audio.wav
```

Supports WAV (16kHz mono 16-bit PCM) and RAW (16kHz mono 32-bit float) audio files.

## Quick Start (Browser)

```javascript
const response = await fetch('https://huggingface.co/{model_info["hf_repo"]}/resolve/main/bin/tomoul_{model_name}_web_wasm32_bundled.wasm');
const wasm = await WebAssembly.instantiate(await response.arrayBuffer());
wasm.instance.exports.init();
const prob = wasm.instance.exports.process_audio(512);
```
{symbols_doc}
## Files

| File | Description | Size |
|------|-------------|------|
| `{model_name}.tl` | Raw model weights | 2.1 MB |
| `bin/tomoul_{model_name}_web_wasm32_bundled.wasm` | Browser WASM (bundled) | ~2.2 MB |
| `bin/tomoul_{model_name}_linux_x86_64_bundled` | Linux x64 CLI (bundled) | 2.2 MB |
| `bin/tomoul_{model_name}_linux_aarch64_bundled` | Linux ARM64 CLI (bundled) | 2.2 MB |
| `bin/tomoul_{model_name}_mac_x86_64_bundled` | macOS Intel CLI (bundled) | 2.2 MB |
| `bin/tomoul_{model_name}_mac_aarch64_bundled` | macOS Apple Silicon CLI (bundled) | 2.2 MB |
| `lib/*` | C libraries (static .a and shared .so/.dylib) | ~2.2 MB |
| `include/tomoul_{model_name}.h` | C header file | <1 KB |

All binaries built with ReleaseSmall optimization for minimal size while maintaining excellent performance.

## License

MIT License
'''

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme)
    print(f"  -> {readme_path}")


def sha256_file(filepath):
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_manifest(model_name, version, output_dir):
    """Generate manifest.json with file hashes for programmatic access."""
    print(f"\n[MANIFEST] Generating manifest.json")

    model_info = MODELS[model_name]
    artifacts = {}

    # Scan all files in output directory and generate hashes
    for f in sorted(output_dir.rglob("*")):
        if f.is_file() and f.name != "manifest.json":
            rel_path = str(f.relative_to(output_dir))
            file_hash = sha256_file(f)
            file_size = f.stat().st_size

            # Categorize artifact
            if f.suffix == ".wasm":
                key = f"wasm_{f.stem.replace('tomoul_', '').replace('_bundled', '')}"
            elif f.suffix in [".so", ".dylib"]:
                key = f"lib_{f.stem.replace('libtomoul_', '').replace(model_name + '_', '')}_shared"
            elif f.suffix == ".a":
                key = f"lib_{f.stem.replace('libtomoul_', '').replace(model_name + '_', '')}_static"
            elif f.suffix == ".h":
                key = "header"
            elif f.suffix == ".tl":
                key = "weights"
            elif f.suffix == ".md":
                key = "readme"
            elif f.parent.name == "bin" and not f.suffix:
                key = f"bin_{f.stem.replace('tomoul_', '').replace('_bundled', '')}"
            else:
                key = f.stem

            artifacts[key] = {
                "file": rel_path,
                "sha256": file_hash,
                "size": file_size,
            }

    manifest = {
        "model": model_name,
        "version": version,
        "description": model_info.get("description", ""),
        "artifacts": artifacts,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Tomoul Release Script")
    parser.add_argument("--model", help="Model to build (e.g., silero_vad)")
    parser.add_argument("--all", action="store_true", help="Build all models")
    parser.add_argument("--version", default="dev", help="Version tag (e.g., v1.0.0)")
    parser.add_argument("--build-only", action="store_true", help="Build only, don't upload")
    parser.add_argument("--wasm-only", action="store_true", help="Build only WASM targets")
    parser.add_argument("--output", default="release", help="Output directory")
    args = parser.parse_args()

    # Determine which models to build
    if args.all:
        models = list(MODELS.keys())
    elif args.model:
        if args.model not in MODELS:
            print(f"ERROR: Unknown model '{args.model}'")
            print(f"Available: {', '.join(MODELS.keys())}")
            sys.exit(1)
        models = [args.model]
    else:
        print("ERROR: Specify --model or --all")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    bin_dir = output_dir / "bin"
    bin_dir.mkdir()
    lib_dir = output_dir / "lib"
    lib_dir.mkdir()
    include_dir = output_dir / "include"
    include_dir.mkdir()

    print(f"=== Tomoul Release {args.version} ===")
    print(f"Models: {', '.join(models)}")
    print(f"Output: {output_dir}")

    # Build each model
    for model_name in models:
        model_info = MODELS[model_name]

        # Generate C header
        generate_c_header(model_name, include_dir)

        # Build WASM
        build_wasm(model_name, bin_dir)

        # Build native (unless wasm-only)
        if not args.wasm_only:
            for platform, arch, zig_target, suffix in TARGETS:
                if platform != "web":
                    build_native(model_name, platform, arch, zig_target, suffix, bin_dir)

            # Build libraries
            for platform, arch, zig_target, static_ext, shared_ext in LIB_TARGETS:
                build_library(model_name, platform, arch, zig_target, static_ext, shared_ext, lib_dir)

        # Copy weights
        weights_src = Path(model_info["weights"])
        if weights_src.exists():
            weights_dst = output_dir / weights_src.name
            shutil.copy(weights_src, weights_dst)
            print(f"\n[COPY] {weights_src} -> {weights_dst}")

        # Generate README
        print(f"\n[README] Generating README.md")
        generate_readme(model_name, output_dir)

        # Generate manifest.json with SHA256 hashes
        generate_manifest(model_name, args.version, output_dir)

        # Upload (unless build-only)
        if not args.build_only:
            upload_to_hf(model_name, args.version, output_dir)

    print(f"\n=== Done ===")
    print(f"Artifacts in: {output_dir}/")

    # List artifacts
    print("\nArtifacts:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
            print(f"  {f.relative_to(output_dir)} ({size_str})")


if __name__ == "__main__":
    main()
