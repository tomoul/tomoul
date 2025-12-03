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
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Model registry (mirrors model_registry.zig)
MODELS = {
    "silero_vad": {
        "hf_repo": "tomoul/silero-vad",
        "weights": "models/silero_vad.tl",
        "description": "Voice Activity Detection",
    },
    # Add more models here as they're implemented
}

# Build targets
TARGETS = [
    # (platform, arch, zig_target, output_suffix)
    ("web", "wasm32", "wasm32-freestanding", ".wasm"),
    ("linux", "x86_64", "x86_64-linux", ""),
    ("linux", "aarch64", "aarch64-linux", ""),
    ("mac", "x86_64", "x86_64-macos", ""),
    ("mac", "aarch64", "aarch64-macos", ""),
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

    # For now, build the main executable (not model-specific)
    # In the future, we could have model-specific native builds
    result = run(f"zig build -Dtarget={zig_target} -Doptimize=ReleaseFast")
    if result is None:
        return False

    src = Path("zig-out/bin/tomoul")
    if src.exists():
        dst = output_dir / f"tomoul_{platform}_{arch}_bundled{suffix}"
        shutil.copy(src, dst)
        # Make executable
        if not suffix:
            os.chmod(dst, 0o755)
        print(f"  -> {dst}")
        return True
    else:
        print(f"  WARNING: {src} not found (native build may not be configured)")
        return False


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

## Quick Start (Browser)

```javascript
const response = await fetch('https://huggingface.co/{model_info["hf_repo"]}/resolve/main/bin/tomoul_{model_name}_web_wasm32_bundled.wasm');
const wasm = await WebAssembly.instantiate(await response.arrayBuffer());
wasm.instance.exports.init();
const prob = wasm.instance.exports.process_audio(512);
```

## Files

| File | Description |
|------|-------------|
| `{model_name}.tl` | Raw model weights |
| `bin/tomoul_{model_name}_web_wasm32_bundled.wasm` | Browser WASM |
| `bin/tomoul_linux_x86_64_bundled` | Linux x64 |
| `bin/tomoul_linux_aarch64_bundled` | Linux ARM64 |
| `bin/tomoul_mac_x86_64_bundled` | macOS Intel |
| `bin/tomoul_mac_aarch64_bundled` | macOS Apple Silicon |

## License

MIT License
'''

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme)
    print(f"  -> {readme_path}")


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

    print(f"=== Tomoul Release {args.version} ===")
    print(f"Models: {', '.join(models)}")
    print(f"Output: {output_dir}")

    # Build each model
    for model_name in models:
        model_info = MODELS[model_name]

        # Build WASM
        build_wasm(model_name, bin_dir)

        # Build native (unless wasm-only)
        if not args.wasm_only:
            for platform, arch, zig_target, suffix in TARGETS:
                if platform != "web":
                    build_native(model_name, platform, arch, zig_target, suffix, bin_dir)

        # Copy weights
        weights_src = Path(model_info["weights"])
        if weights_src.exists():
            weights_dst = output_dir / weights_src.name
            shutil.copy(weights_src, weights_dst)
            print(f"\n[COPY] {weights_src} -> {weights_dst}")

        # Generate README
        print(f"\n[README] Generating README.md")
        generate_readme(model_name, output_dir)

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
