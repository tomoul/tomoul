#!/usr/bin/env python3
"""
Tomoul Silero VAD Exporter
Exports Silero VAD model weights to .tl binary format for Zig inference engine.

Downloads the model from torch.hub and exports all parameters.
Also creates a validation file with test input/output for parity testing.

Binary Format (.tl):
- Header (16 bytes): Magic "TOUL", Version (u32), Tensor count (u32), Reserved (u32)
- Tensor Table: For each tensor - name, shape, data offset, data size
- Data Section: Raw f32 data (little-endian)
"""

import struct
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

MAGIC = b'TOUL'
VERSION = 1


class TomoulExporter:
    """Export PyTorch tensors to Tomoul binary format."""

    def __init__(self):
        self.tensors: Dict[str, torch.Tensor] = {}

    def add_tensor(self, name: str, tensor: torch.Tensor):
        """Add a tensor to be exported."""
        # Ensure float32 and contiguous memory layout
        t = tensor.detach().cpu().float().contiguous()
        self.tensors[name] = t
        print(f"  {name}: {list(t.shape)} ({t.numel()} params)")

    def export(self, output_path: str) -> Path:
        """Write all tensors to .tl file."""
        path = Path(output_path)
        tensor_count = len(self.tensors)

        print(f"\nExporting {tensor_count} tensors to {path}")

        with open(path, 'wb') as f:
            # Write header (16 bytes)
            f.write(MAGIC)                          # 4 bytes: Magic
            f.write(struct.pack('<I', VERSION))     # 4 bytes: Version
            f.write(struct.pack('<I', tensor_count)) # 4 bytes: Tensor count
            f.write(struct.pack('<I', 0))           # 4 bytes: Reserved

            # First pass: write tensor table with placeholder offsets
            tensor_info: List[Tuple[str, torch.Tensor, int, int]] = []

            for name, tensor in self.tensors.items():
                name_bytes = name.encode('utf-8')
                shape = list(tensor.shape)
                data_size = tensor.numel() * 4  # f32 = 4 bytes

                # Write name length and name
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)

                # Write number of dimensions and shape
                f.write(struct.pack('<I', len(shape)))
                for dim in shape:
                    f.write(struct.pack('<I', dim))

                # Record position for offset, write placeholder
                offset_pos = f.tell()
                f.write(struct.pack('<Q', 0))       # 8 bytes: Data offset (placeholder)
                f.write(struct.pack('<Q', data_size)) # 8 bytes: Data size

                tensor_info.append((name, tensor, offset_pos, data_size))

            # Second pass: write data and update offsets
            for name, tensor, offset_pos, data_size in tensor_info:
                current_offset = f.tell()

                # Go back and write the correct offset
                f.seek(offset_pos)
                f.write(struct.pack('<Q', current_offset))
                f.seek(0, 2)  # Return to end of file

                # Write tensor data as little-endian float32
                data = tensor.numpy().astype(np.float32).tobytes()
                f.write(data)

        # Verification
        file_size = path.stat().st_size
        print(f"Exported successfully: {file_size} bytes")

        return path


def export_silero_vad():
    """Export Silero VAD model."""
    print("=" * 60)
    print("Silero VAD Model Exporter")
    print("=" * 60)
    print("\nLoading Silero VAD model from torch.hub...")

    # Load model from torch hub
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )

    model.eval()
    print(f"Model loaded successfully")
    print(f"Model type: {type(model).__name__}")

    # Print model structure
    print("\n=== Model Structure ===")
    for name, module in model.named_modules():
        if name:
            print(f"  {name}: {type(module).__name__}")

    exporter = TomoulExporter()

    # Export state dict
    print("\n=== Extracting Weights ===")
    state_dict = model.state_dict()

    for name, param in state_dict.items():
        # Clean up names (remove _orig_mod. prefix if present)
        clean_name = name.replace("_orig_mod.", "")
        exporter.add_tensor(clean_name, param)

    # Export to models directory
    output_path = Path(__file__).parent.parent / "models" / "silero_vad.tl"
    output_path.parent.mkdir(exist_ok=True)
    exporter.export(str(output_path))

    # Print tensor summary for Zig implementation reference
    print("\n=== Tensor Reference for Zig ===")
    print("Use these names to load tensors:")
    for name, tensor in exporter.tensors.items():
        print(f"  loader.getTensor(\"{name}\")  // shape: {list(tensor.shape)}")

    return model, exporter


def export_validation_data(model):
    """Create validation data for parity testing with Zig."""
    print("\n" + "=" * 60)
    print("Creating Validation Data")
    print("=" * 60)

    validation_exporter = TomoulExporter()

    # Reset model states
    model.reset_states()

    # Test 1: Silence (zeros)
    print("\nTest 1: Silence input (zeros)")
    audio_silence = torch.zeros(1, 512)
    validation_exporter.add_tensor("test_silence_input", audio_silence)

    with torch.no_grad():
        prob_silence = model(audio_silence, 16000)
    validation_exporter.add_tensor("test_silence_output", prob_silence)
    print(f"  Silence probability: {prob_silence.item():.6f}")

    # Reset for next test
    model.reset_states()

    # Test 2: Sine wave (simulated speech-like tone)
    print("\nTest 2: Tone input (440Hz sine)")
    t = torch.linspace(0, 0.032, 512)  # 32ms at 16kHz
    audio_tone = torch.sin(2 * 3.14159265 * 440 * t).unsqueeze(0)
    validation_exporter.add_tensor("test_tone_input", audio_tone)

    with torch.no_grad():
        prob_tone = model(audio_tone, 16000)
    validation_exporter.add_tensor("test_tone_output", prob_tone)
    print(f"  Tone probability: {prob_tone.item():.6f}")

    # Reset for next test
    model.reset_states()

    # Test 3: Random noise
    print("\nTest 3: Random noise input")
    torch.manual_seed(42)  # Reproducible
    audio_noise = torch.randn(1, 512) * 0.1
    validation_exporter.add_tensor("test_noise_input", audio_noise)

    with torch.no_grad():
        prob_noise = model(audio_noise, 16000)
    validation_exporter.add_tensor("test_noise_output", prob_noise)
    print(f"  Noise probability: {prob_noise.item():.6f}")

    # Export validation file
    output_path = Path(__file__).parent.parent / "models" / "vad_validation.tl"
    validation_exporter.export(str(output_path))

    print("\n=== Validation Reference Values ===")
    print(f"Silence: {prob_silence.item():.6f}")
    print(f"Tone:    {prob_tone.item():.6f}")
    print(f"Noise:   {prob_noise.item():.6f}")

    return {
        'silence': prob_silence.item(),
        'tone': prob_tone.item(),
        'noise': prob_noise.item()
    }


def inspect_model_internals(model):
    """Detailed inspection of model for implementation reference."""
    print("\n" + "=" * 60)
    print("Model Internals (Implementation Reference)")
    print("=" * 60)

    state_dict = model.state_dict()

    # Categorize tensors
    encoder_tensors = []
    lstm_tensors = []
    decoder_tensors = []
    other_tensors = []

    for name in state_dict.keys():
        clean_name = name.replace("_orig_mod.", "")
        if 'encoder' in clean_name or 'stft' in clean_name or 'first' in clean_name:
            encoder_tensors.append(clean_name)
        elif 'lstm' in clean_name or 'rnn' in clean_name:
            lstm_tensors.append(clean_name)
        elif 'decoder' in clean_name or 'final' in clean_name or 'out' in clean_name:
            decoder_tensors.append(clean_name)
        else:
            other_tensors.append(clean_name)

    print("\n[Encoder Tensors]")
    for name in encoder_tensors:
        param = state_dict[name.replace("_orig_mod.", "") if "_orig_mod." not in name else name]
        print(f"  {name}: {list(param.shape)}")

    print("\n[LSTM Tensors]")
    for name in lstm_tensors:
        # Find the original name in state_dict
        for orig_name in state_dict.keys():
            if name in orig_name or orig_name.replace("_orig_mod.", "") == name:
                param = state_dict[orig_name]
                print(f"  {name}: {list(param.shape)}")
                break

    print("\n[Decoder Tensors]")
    for name in decoder_tensors:
        for orig_name in state_dict.keys():
            if name in orig_name or orig_name.replace("_orig_mod.", "") == name:
                param = state_dict[orig_name]
                print(f"  {name}: {list(param.shape)}")
                break

    if other_tensors:
        print("\n[Other Tensors]")
        for name in other_tensors:
            for orig_name in state_dict.keys():
                if name in orig_name or orig_name.replace("_orig_mod.", "") == name:
                    param = state_dict[orig_name]
                    print(f"  {name}: {list(param.shape)}")
                    break

    # LSTM specifics
    print("\n[LSTM Gate Layout]")
    print("PyTorch LSTM weights are packed as [input, forget, cell, output] gates")
    print("Each gate has hidden_size rows, so total is 4*hidden_size")

    for name in state_dict.keys():
        if 'weight_ih' in name:
            param = state_dict[name]
            hidden_size = param.shape[0] // 4
            input_size = param.shape[1]
            print(f"\n  {name}:")
            print(f"    Total shape: {list(param.shape)}")
            print(f"    hidden_size: {hidden_size}")
            print(f"    input_size: {input_size}")
            print(f"    Gates: i[0:{hidden_size}], f[{hidden_size}:{2*hidden_size}], "
                  f"g[{2*hidden_size}:{3*hidden_size}], o[{3*hidden_size}:{4*hidden_size}]")


def verify_export(model_path: Path):
    """Read and verify the exported file structure."""
    print("\n" + "=" * 60)
    print("File Verification")
    print("=" * 60)

    with open(model_path, 'rb') as f:
        # Read header
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<I', f.read(4))[0]
        reserved = struct.unpack('<I', f.read(4))[0]

        print(f"\nHeader:")
        print(f"  Magic: {magic} (expected: b'TOUL')")
        print(f"  Version: {version}")
        print(f"  Tensor count: {tensor_count}")

        # Read tensor table (just count and verify)
        print(f"\nTensor Table ({tensor_count} entries):")
        for i in range(min(5, tensor_count)):  # Show first 5
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            num_dims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(num_dims)]
            data_offset = struct.unpack('<Q', f.read(8))[0]
            data_size = struct.unpack('<Q', f.read(8))[0]
            print(f"  [{i}] {name}: shape={shape}, size={data_size} bytes")

        if tensor_count > 5:
            print(f"  ... and {tensor_count - 5} more tensors")


if __name__ == "__main__":
    # Export the model
    model, exporter = export_silero_vad()

    # Inspect internals for implementation
    inspect_model_internals(model)

    # Create validation data
    validation_results = export_validation_data(model)

    # Verify exported file
    model_path = Path(__file__).parent.parent / "models" / "silero_vad.tl"
    verify_export(model_path)

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - models/silero_vad.tl (model weights)")
    print(f"  - models/vad_validation.tl (test vectors)")
    print(f"\nNext steps:")
    print(f"  1. Implement LSTM cell in src/core/ops.zig")
    print(f"  2. Create SileroVAD struct in src/models/vad.zig")
    print(f"  3. Load weights and run inference")
