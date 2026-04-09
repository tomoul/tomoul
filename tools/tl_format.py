#!/usr/bin/env python3
"""
Tomoul .tl format utilities - shared across all exporters.

Supports:
- Float32 format (default)
- Q8_0 format (8-bit symmetric quantization)
- Q4_0 format (4-bit symmetric quantization)
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

MAGIC = b'TOUL'
VERSION = 1


class QuantFormat:
    """Quantization format identifiers (matches Zig loader)."""
    F32 = 0   # Float32 (no quantization)
    Q8_0 = 1  # Q8_0: symmetric 8-bit (per-tensor scale)
    Q4_0 = 2  # Q4_0: symmetric 4-bit (per-tensor scale)
    Q8_K = 3  # Q8_K: block-wise 8-bit (per-block scale, better accuracy)


# Block size for block-wise quantization (matches Zig BLOCK_SIZE)
BLOCK_SIZE = 32


def quantize_q8_0(tensor: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Quantize tensor to Q8_0 format (symmetric 8-bit).

    Args:
        tensor: Float32 numpy array

    Returns:
        (scale, quantized_data) where:
        - scale: float32 scale factor
        - quantized_data: int8 numpy array
    """
    data = tensor.flatten().astype(np.float32)

    # Find max absolute value
    max_abs = np.max(np.abs(data))

    if max_abs == 0:
        return 1.0, np.zeros(len(data), dtype=np.int8)

    # Symmetric quantization: scale = max_abs / 127
    scale = float(max_abs / 127.0)

    # Quantize: q = round(x / scale), clamp to [-127, 127]
    quantized = np.round(data / scale).astype(np.int32)
    quantized = np.clip(quantized, -127, 127).astype(np.int8)

    return scale, quantized


def dequantize_q8_0(scale: float, quantized: np.ndarray) -> np.ndarray:
    """Dequantize Q8_0 data back to float32 for verification."""
    return quantized.astype(np.float32) * scale


def quantize_q8_k(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize tensor to Q8_K format (block-wise 8-bit).

    Args:
        tensor: Float32 numpy array

    Returns:
        (scales, quantized_data) where:
        - scales: float32 numpy array (one per block)
        - quantized_data: int8 numpy array
    """
    data = tensor.flatten().astype(np.float32)
    n = len(data)
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    scales = np.zeros(num_blocks, dtype=np.float32)
    quantized = np.zeros(n, dtype=np.int8)

    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, n)
        block = data[start:end]

        # Find max absolute value in this block
        max_abs = np.max(np.abs(block))

        if max_abs == 0:
            scales[block_idx] = 1.0
            continue

        # Calculate scale for this block
        scales[block_idx] = float(max_abs / 127.0)
        inv_scale = 127.0 / max_abs

        # Quantize block elements
        q = np.round(block * inv_scale).astype(np.int32)
        q = np.clip(q, -127, 127).astype(np.int8)
        quantized[start:end] = q

    return scales, quantized


def dequantize_q8_k(scales: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    """Dequantize Q8_K data back to float32 for verification."""
    n = len(quantized)
    result = np.zeros(n, dtype=np.float32)

    for block_idx in range(len(scales)):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, n)
        scale = scales[block_idx]
        result[start:end] = quantized[start:end].astype(np.float32) * scale

    return result


def verify_quantization_q8_k(original: np.ndarray, scales: np.ndarray, quantized: np.ndarray) -> Dict:
    """Verify Q8_K quantization accuracy."""
    restored = dequantize_q8_k(scales, quantized)
    original_flat = original.flatten()

    mse = np.mean((original_flat - restored) ** 2)
    max_error = np.max(np.abs(original_flat - restored))
    max_val = np.max(np.abs(original_flat))
    rel_error = max_error / max_val if max_val > 0 else 0

    return {
        'mse': float(mse),
        'max_error': float(max_error),
        'rel_error': float(rel_error),
        'max_val': float(max_val),
    }


def quantize_q4_0(tensor: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Quantize tensor to Q4_0 format (symmetric 4-bit, packed).

    Args:
        tensor: Float32 numpy array

    Returns:
        (scale, packed_data) where:
        - scale: float32 scale factor
        - packed_data: uint8 numpy array (2 values per byte)
    """
    data = tensor.flatten().astype(np.float32)
    n = len(data)

    # Find max absolute value
    max_abs = np.max(np.abs(data))

    if max_abs == 0:
        packed_size = (n + 1) // 2
        return 1.0, np.zeros(packed_size, dtype=np.uint8)

    # Symmetric quantization: scale = max_abs / 7
    scale = float(max_abs / 7.0)

    # Quantize: q = round(x / scale), clamp to [-7, 7]
    quantized = np.round(data / scale).astype(np.int32)
    quantized = np.clip(quantized, -7, 7).astype(np.int8)

    # Pack two 4-bit values per byte (low nibble = even index, high nibble = odd)
    packed_size = (n + 1) // 2
    packed = np.zeros(packed_size, dtype=np.uint8)

    for i in range(n):
        byte_idx = i // 2
        # Convert signed 4-bit to unsigned nibble
        nibble = quantized[i] & 0x0F

        if i % 2 == 0:
            # Low nibble
            packed[byte_idx] = nibble
        else:
            # High nibble
            packed[byte_idx] |= (nibble << 4)

    return scale, packed


def dequantize_q4_0(scale: float, packed: np.ndarray, element_count: int) -> np.ndarray:
    """Dequantize Q4_0 packed data back to float32 for verification."""
    result = np.zeros(element_count, dtype=np.float32)

    for i in range(element_count):
        byte_idx = i // 2
        if i % 2 == 0:
            nibble = int(packed[byte_idx]) & 0x0F
        else:
            nibble = (int(packed[byte_idx]) >> 4) & 0x0F

        # Sign-extend from 4-bit
        if nibble & 0x08:
            value = nibble - 16
        else:
            value = nibble

        result[i] = float(value) * scale

    return result


def verify_quantization_q4(original: np.ndarray, scale: float, packed: np.ndarray) -> Dict:
    """Verify Q4_0 quantization accuracy."""
    element_count = original.size
    restored = dequantize_q4_0(scale, packed, element_count)
    original_flat = original.flatten()

    mse = np.mean((original_flat - restored) ** 2)
    max_error = np.max(np.abs(original_flat - restored))
    max_val = np.max(np.abs(original_flat))
    rel_error = max_error / max_val if max_val > 0 else 0

    return {
        'mse': float(mse),
        'max_error': float(max_error),
        'rel_error': float(rel_error),
        'max_val': float(max_val),
    }


def verify_quantization(original: np.ndarray, scale: float, quantized: np.ndarray) -> Dict:
    """Verify quantization accuracy."""
    restored = dequantize_q8_0(scale, quantized)
    original_flat = original.flatten()

    mse = np.mean((original_flat - restored) ** 2)
    max_error = np.max(np.abs(original_flat - restored))
    max_val = np.max(np.abs(original_flat))
    rel_error = max_error / max_val if max_val > 0 else 0

    return {
        'mse': float(mse),
        'max_error': float(max_error),
        'rel_error': float(rel_error),
        'max_val': float(max_val),
    }


def export_tensors(
    tensors: Dict[str, np.ndarray],
    output_path: str,
    quant_format: int = QuantFormat.F32,
    verify: bool = True,
):
    """
    Export tensors to .tl format.

    Args:
        tensors: Dict of tensor name -> numpy array
        output_path: Output file path
        quant_format: Quantization format (F32, Q8_0, etc.)
        verify: Whether to verify quantization accuracy
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_count = len(tensors)

    format_names = {QuantFormat.F32: 'f32', QuantFormat.Q8_0: 'q8_0', QuantFormat.Q4_0: 'q4_0'}
    print(f"Exporting {tensor_count} tensors as {format_names.get(quant_format, 'unknown')} to {path}")

    total_original = 0
    total_quantized = 0
    max_rel_error = 0.0

    with open(path, 'wb') as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))  # Version 1
        f.write(struct.pack('<I', tensor_count))

        # Reserved field: first byte is quant format
        f.write(struct.pack('<B', quant_format))
        f.write(b'\x00' * 3)  # Padding

        # Tensor table entries (we'll fill in offsets later)
        entries = []
        for name, tensor in tensors.items():
            name_bytes = name.encode('utf-8')
            shape = list(tensor.shape)

            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))

            # Placeholder for offset and size (will update later)
            offset_pos = f.tell()
            f.write(struct.pack('<Q', 0))  # offset
            f.write(struct.pack('<Q', 0))  # size
            entries.append((name, tensor, offset_pos))

        # Data section
        for name, tensor, offset_pos in entries:
            current_pos = f.tell()
            original_size = tensor.size * 4  # float32 bytes

            if quant_format == QuantFormat.F32:
                # No quantization
                data = tensor.astype(np.float32).tobytes()
                f.write(data)
                data_size = len(data)

            elif quant_format == QuantFormat.Q8_0:
                scale, quantized = quantize_q8_0(tensor)

                # Verify if requested
                if verify:
                    stats = verify_quantization(tensor, scale, quantized)
                    max_rel_error = max(max_rel_error, stats['rel_error'])

                # Write: 4-byte scale + int8 data
                f.write(struct.pack('<f', scale))
                f.write(quantized.tobytes())
                data_size = 4 + len(quantized)

            elif quant_format == QuantFormat.Q4_0:
                scale, packed = quantize_q4_0(tensor)

                # Verify if requested
                if verify:
                    stats = verify_quantization_q4(tensor, scale, packed)
                    max_rel_error = max(max_rel_error, stats['rel_error'])

                # Write: 4-byte scale + packed uint8 data
                f.write(struct.pack('<f', scale))
                f.write(packed.tobytes())
                data_size = 4 + len(packed)

            elif quant_format == QuantFormat.Q8_K:
                scales, quantized = quantize_q8_k(tensor)

                # Verify if requested
                if verify:
                    stats = verify_quantization_q8_k(tensor, scales, quantized)
                    max_rel_error = max(max_rel_error, stats['rel_error'])

                # Write: num_blocks (u32) + scales (f32 * num_blocks) + int8 data
                num_blocks = len(scales)
                f.write(struct.pack('<I', num_blocks))
                f.write(scales.tobytes())
                f.write(quantized.tobytes())
                data_size = 4 + num_blocks * 4 + len(quantized)

            else:
                raise ValueError(f"Unsupported quantization format: {quant_format}")

            # Update offset and size in header
            end_pos = f.tell()
            f.seek(offset_pos)
            f.write(struct.pack('<Q', current_pos))
            f.write(struct.pack('<Q', data_size))
            f.seek(end_pos)

            total_original += original_size
            total_quantized += data_size

    file_size = path.stat().st_size
    overall_compression = total_original / total_quantized if total_quantized > 0 else 0

    print(f"Exported: {file_size / 1024 / 1024:.2f} MB")

    if quant_format != QuantFormat.F32:
        print(f"Compression: {overall_compression:.2f}x "
              f"({total_original / 1024 / 1024:.1f}MB -> {total_quantized / 1024 / 1024:.1f}MB)")

    if verify and quant_format in (QuantFormat.Q8_0, QuantFormat.Q4_0, QuantFormat.Q8_K):
        print(f"Max relative error: {max_rel_error:.4%}")
        # Q4_0 has higher expected error due to fewer quantization levels
        if quant_format == QuantFormat.Q4_0:
            if max_rel_error < 0.08:
                print("Quantization accuracy: EXCELLENT (<8%)")
            elif max_rel_error < 0.15:
                print("Quantization accuracy: GOOD (<15%)")
            else:
                print("Quantization accuracy: WARNING (>15%)")
        else:  # Q8_0 and Q8_K
            if max_rel_error < 0.01:
                print("Quantization accuracy: EXCELLENT (<1%)")
            elif max_rel_error < 0.05:
                print("Quantization accuracy: GOOD (<5%)")
            else:
                print("Quantization accuracy: WARNING (>5%)")

    return path


def load_tl_file(path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load tensors from existing .tl file.

    Returns:
        (tensors dict, quant_format)
    """
    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version {version}, expected version 1")

        tensor_count = struct.unpack('<I', f.read(4))[0]
        reserved = f.read(4)
        quant_format = reserved[0]  # First byte is quant format

        # Read tensor table
        tensor_info = []
        for _ in range(tensor_count):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            num_dims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(num_dims)]

            offset = struct.unpack('<Q', f.read(8))[0]
            size = struct.unpack('<Q', f.read(8))[0]

            tensor_info.append((name, shape, offset, size))

        # Read tensor data
        tensors = {}
        for name, shape, offset, size in tensor_info:
            f.seek(offset)

            if quant_format == QuantFormat.F32:
                data = np.frombuffer(f.read(size), dtype=np.float32)
                tensors[name] = data.reshape(shape)
            elif quant_format == QuantFormat.Q8_0:
                scale = struct.unpack('<f', f.read(4))[0]
                quantized = np.frombuffer(f.read(size - 4), dtype=np.int8)
                # Dequantize for return
                tensors[name] = dequantize_q8_0(scale, quantized).reshape(shape)
            elif quant_format == QuantFormat.Q4_0:
                scale = struct.unpack('<f', f.read(4))[0]
                packed = np.frombuffer(f.read(size - 4), dtype=np.uint8)
                # Calculate element count from shape
                element_count = 1
                for dim in shape:
                    element_count *= dim
                # Dequantize for return
                tensors[name] = dequantize_q4_0(scale, packed, element_count).reshape(shape)
            elif quant_format == QuantFormat.Q8_K:
                # Read num_blocks first
                num_blocks = struct.unpack('<I', f.read(4))[0]
                # Read scales
                scales = np.frombuffer(f.read(num_blocks * 4), dtype=np.float32)
                # Read quantized data
                quantized = np.frombuffer(f.read(size - 4 - num_blocks * 4), dtype=np.int8)
                # Dequantize for return
                tensors[name] = dequantize_q8_k(scales, quantized).reshape(shape)
            else:
                raise ValueError(f"Unsupported quant format: {quant_format}")

        return tensors, quant_format


def add_quantize_args(parser):
    """Add standard quantization arguments to an argparse parser."""
    parser.add_argument(
        '--quantize', '-q',
        choices=['f32', 'q8_0', 'q4_0', 'q8_k'],
        default='f32',
        help='Output format: f32 (default), q8_0 (8-bit, ~4x), q4_0 (4-bit, ~8x), q8_k (block-wise 8-bit, best accuracy)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip quantization verification'
    )


def get_quant_format(format_str: str) -> int:
    """Convert format string to QuantFormat constant."""
    formats = {
        'f32': QuantFormat.F32,
        'q8_0': QuantFormat.Q8_0,
        'q4_0': QuantFormat.Q4_0,
        'q8_k': QuantFormat.Q8_K,
    }
    return formats.get(format_str, QuantFormat.F32)
