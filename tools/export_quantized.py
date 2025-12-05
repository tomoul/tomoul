#!/usr/bin/env python3
"""
Convert .tl models to quantized Q8_0 format

Supports weight-only quantization for inference optimization.
The exported model uses symmetric 8-bit quantization per tensor.

Usage:
    python export_quantized.py --input model.tl --output model_q8.tl
    python export_quantized.py --input model.tl --output model_q8.tl --no-verify

Note: For new exports, prefer using the --quantize flag directly in model exporters:
    python export_silero_vad.py --quantize q8_0
    python export_fullstop_punctuation_multilang_large.py -q q8_0

This tool is for converting existing float32 .tl files to quantized format.
"""

import argparse
from pathlib import Path

# Import shared format utilities
from tl_format import (
    QuantFormat, export_tensors, load_tl_file
)


def main():
    parser = argparse.ArgumentParser(
        description='Convert .tl models to quantized Q8_0 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert existing .tl file to quantized
  python export_quantized.py --input model.tl --output model_q8.tl

  # Export without verification (faster)
  python export_quantized.py --input model.tl --output model_q8.tl --no-verify

Note: For new exports, prefer using --quantize flag directly in model exporters.
"""
    )

    parser.add_argument('--input', '-i', type=str, required=True, help='Input .tl file to convert')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output .tl file path')
    parser.add_argument('--format', choices=['q8_0', 'f32'], default='q8_0',
                        help='Quantization format (default: q8_0)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip quantization verification')

    args = parser.parse_args()

    quant_format = QuantFormat.Q8_0 if args.format == 'q8_0' else QuantFormat.F32

    print(f"Loading: {args.input}")
    tensors, input_format = load_tl_file(args.input)

    format_names = {QuantFormat.F32: 'f32', QuantFormat.Q8_0: 'q8_0'}
    print(f"Input format: {format_names.get(input_format, 'unknown')}")
    print(f"Output format: {format_names.get(quant_format, 'unknown')}")
    print(f"Tensors: {len(tensors)}")

    export_tensors(tensors, args.output, quant_format, verify=not args.no_verify)


if __name__ == '__main__':
    main()
