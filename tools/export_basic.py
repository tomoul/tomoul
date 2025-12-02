#!/usr/bin/env python3
"""
Tomoul Model Exporter
Exports PyTorch models to .tl binary format for Zig inference engine.

Binary Format (.tl):
- Header (16 bytes): Magic "TOUL", Version (u32), Tensor count (u32), Reserved (u32)
- Tensor Table: For each tensor - name, shape, data offset, data size
- Data Section: Raw f32 data (little-endian)
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


def export_simple_model():
    """Export a simple linear model for testing."""
    print("Creating simple test model...")
    print("Model: Linear(4, 8) -> ReLU -> Linear(8, 2)")

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    # Initialize with known values for testing
    with torch.no_grad():
        model[0].weight.fill_(0.1)
        model[0].bias.fill_(0.01)
        model[2].weight.fill_(0.2)
        model[2].bias.fill_(0.02)

    print("\nExtracting parameters:")
    exporter = TomoulExporter()
    exporter.add_model(model)

    # Export to models directory
    output_path = Path(__file__).parent.parent / "models" / "model.tl"
    output_path.parent.mkdir(exist_ok=True)
    exporter.export(str(output_path))

    # Print verification data
    print("\n" + "=" * 50)
    print("VERIFICATION DATA (compare with Zig loader):")
    print("=" * 50)
    print(f"\nTensor count: {len(exporter.tensors)}")

    for name, tensor in exporter.tensors.items():
        print(f"\n{name}:")
        print(f"  Shape: {list(tensor.shape)}")
        flat = tensor.flatten()
        print(f"  First 5 values: {flat[:5].tolist()}")

    return output_path


def verify_file(path: Path):
    """Read and verify the exported file structure."""
    print(f"\n{'=' * 50}")
    print("FILE VERIFICATION:")
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
        print(f"  Reserved: {reserved}")

        # Read tensor table
        print(f"\nTensor Table:")
        for i in range(tensor_count):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            num_dims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(num_dims)]
            data_offset = struct.unpack('<Q', f.read(8))[0]
            data_size = struct.unpack('<Q', f.read(8))[0]

            print(f"\n  [{i}] {name}:")
            print(f"      Shape: {shape}")
            print(f"      Data offset: {data_offset}")
            print(f"      Data size: {data_size} bytes")

            # Read and show first few values
            current_pos = f.tell()
            f.seek(data_offset)
            num_floats = min(5, data_size // 4)
            values = [struct.unpack('<f', f.read(4))[0] for _ in range(num_floats)]
            print(f"      First {num_floats} values: {values}")
            f.seek(current_pos)


if __name__ == "__main__":
    output_path = export_simple_model()
    verify_file(output_path)
