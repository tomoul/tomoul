#!/usr/bin/env python3
"""
Tomoul Model Exporter
Exports PyTorch models to .tl binary format for Zig inference engine.

Binary Format (.tl):
- Header (16 bytes): Magic "TOUL", Version (u32), Tensor count (u32), Reserved (u32)
- Tensor Table: For each tensor - name, shape, data offset, data size
- Data Section: Raw f32 data (little-endian)

Output:
- tests/fixtures/linear/model.tl - Model weights
- tests/fixtures/linear/validation.tl - Input/output test vectors
"""

import struct
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

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
        print(f"  Added: {name} {list(t.shape)}")

    def add_model(self, model: nn.Module, prefix: str = ""):
        """Add all parameters from a PyTorch model."""
        for name, param in model.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            self.add_tensor(full_name, param)

    def add_state_dict(self, state_dict: Dict[str, torch.Tensor], prefix: str = ""):
        """Add tensors from a state dict."""
        for name, tensor in state_dict.items():
            full_name = f"{prefix}.{name}" if prefix else name
            self.add_tensor(full_name, tensor)

    def export(self, output_path: str) -> Path:
        """Write all tensors to .tl file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
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


def export_linear_fixture():
    """
    Export a simple y = 2x + 1 linear model for testing.

    This creates a deterministic, simple fixture that can be committed to git
    and used for CI/CD testing without downloading large models.
    """
    print("=" * 60)
    print("Linear Fixture Generator")
    print("=" * 60)
    print("\nCreating simple linear model: y = 2x + 1")

    # Create a simple Linear(1, 1) model representing y = 2x + 1
    model = nn.Linear(1, 1, bias=True)

    with torch.no_grad():
        model.weight.fill_(2.0)  # y = 2 * x
        model.bias.fill_(1.0)    # + 1

    print("\nModel parameters:")
    print(f"  weight: {model.weight.item()}")
    print(f"  bias: {model.bias.item()}")

    # Export model weights
    exporter = TomoulExporter()
    exporter.add_tensor("weight", model.weight)
    exporter.add_tensor("bias", model.bias)

    fixture_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "linear"
    model_path = fixture_dir / "model.tl"
    exporter.export(str(model_path))

    # Create validation data
    print("\n" + "=" * 60)
    print("Creating validation data")
    print("=" * 60)

    validation_exporter = TomoulExporter()

    # Test case 1: x = 10.0 -> y = 2*10 + 1 = 21.0
    test_input = torch.tensor([[10.0]])
    with torch.no_grad():
        test_output = model(test_input)

    validation_exporter.add_tensor("input", test_input)
    validation_exporter.add_tensor("expected_output", test_output)

    print(f"\nTest case: input={test_input.item():.1f} -> expected={test_output.item():.1f}")

    # Test case 2: x = 0.0 -> y = 1.0
    test_input_2 = torch.tensor([[0.0]])
    with torch.no_grad():
        test_output_2 = model(test_input_2)

    validation_exporter.add_tensor("input_zero", test_input_2)
    validation_exporter.add_tensor("expected_output_zero", test_output_2)

    print(f"Test case: input={test_input_2.item():.1f} -> expected={test_output_2.item():.1f}")

    # Test case 3: x = -5.0 -> y = -9.0
    test_input_3 = torch.tensor([[-5.0]])
    with torch.no_grad():
        test_output_3 = model(test_input_3)

    validation_exporter.add_tensor("input_negative", test_input_3)
    validation_exporter.add_tensor("expected_output_negative", test_output_3)

    print(f"Test case: input={test_input_3.item():.1f} -> expected={test_output_3.item():.1f}")

    validation_path = fixture_dir / "validation.tl"
    validation_exporter.export(str(validation_path))

    print("\n" + "=" * 60)
    print("Fixture Export Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {model_path}")
    print(f"  - {validation_path}")
    print(f"\nThese files can be committed to git for CI/CD testing.")

    return model_path, validation_path


def verify_file(path: Path):
    """Read and verify the exported file structure."""
    print(f"\n{'=' * 50}")
    print(f"FILE VERIFICATION: {path.name}")
    print("=" * 50)

    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<I', f.read(4))[0]
        reserved = struct.unpack('<I', f.read(4))[0]

        print(f"\nHeader:")
        print(f"  Magic: {magic} (expected: b'TOUL')")
        print(f"  Version: {version}")
        print(f"  Tensor count: {tensor_count}")

        # Read tensor table
        print(f"\nTensors:")
        for i in range(tensor_count):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            num_dims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(num_dims)]
            data_offset = struct.unpack('<Q', f.read(8))[0]
            data_size = struct.unpack('<Q', f.read(8))[0]

            # Read first value
            current_pos = f.tell()
            f.seek(data_offset)
            first_val = struct.unpack('<f', f.read(4))[0]
            f.seek(current_pos)

            print(f"  [{i}] {name}: shape={shape}, first_val={first_val:.4f}")


if __name__ == "__main__":
    # Export fixtures for CI/CD testing
    model_path, validation_path = export_linear_fixture()
    verify_file(model_path)
    verify_file(validation_path)
