#!/usr/bin/env python3
"""
Tomoul Silero VAD Exporter
Exports Silero VAD model weights to .tl binary format for Zig inference engine.

Usage:
    python export_vad.py           # Export full Silero VAD to artifacts/
  python export_vad.py --tiny    # Export tiny fixture to tests/fixtures/silero_vad/

Binary Format (.tl):
- Header (16 bytes): Magic "TOUL", Version (u32), Tensor count (u32), Reserved (u32)
- Tensor Table: For each tensor - name, shape, data offset, data size
- Data Section: Raw f32 data (little-endian)
"""

import argparse
import struct
import torch
import torch.nn as nn
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


class TinySileroVAD(nn.Module):
    """
    A tiny mock VAD model with the same architecture as Silero VAD but with small dimensions.
    Used for CI/CD testing without downloading the full 2MB model.

    Architecture:
      STFT basis -> Conv Encoder (4 layers) -> LSTM -> Decoder -> Sigmoid
    """

    def __init__(self, stft_bins=16, hidden_size=8, kernel_size=3):
        super().__init__()
        self.stft_bins = stft_bins
        self.hidden_size = hidden_size

        # STFT basis (mock - normally [258, 1, 256] for 16kHz)
        # Using smaller: [stft_bins*2, 1, stft_bins*2]
        self.stft_forward_basis = nn.Parameter(
            torch.randn(stft_bins * 2, 1, stft_bins * 2) * 0.1
        )

        # Encoder: 4 conv layers
        # Layer 0: [hidden_size, stft_bins+1, kernel_size]
        self.enc0_weight = nn.Parameter(torch.randn(hidden_size, stft_bins + 1, kernel_size) * 0.1)
        self.enc0_bias = nn.Parameter(torch.zeros(hidden_size))

        # Layer 1: [hidden_size//2, hidden_size, kernel_size]
        self.enc1_weight = nn.Parameter(torch.randn(hidden_size // 2, hidden_size, kernel_size) * 0.1)
        self.enc1_bias = nn.Parameter(torch.zeros(hidden_size // 2))

        # Layer 2: [hidden_size//2, hidden_size//2, kernel_size]
        self.enc2_weight = nn.Parameter(torch.randn(hidden_size // 2, hidden_size // 2, kernel_size) * 0.1)
        self.enc2_bias = nn.Parameter(torch.zeros(hidden_size // 2))

        # Layer 3: [hidden_size, hidden_size//2, kernel_size]
        self.enc3_weight = nn.Parameter(torch.randn(hidden_size, hidden_size // 2, kernel_size) * 0.1)
        self.enc3_bias = nn.Parameter(torch.zeros(hidden_size))

        # LSTM: [4*hidden_size, hidden_size] weights
        self.lstm_weight_ih = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.1)
        self.lstm_weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.1)
        self.lstm_bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.lstm_bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))

        # Decoder: [1, hidden_size, 1]
        self.dec_weight = nn.Parameter(torch.randn(1, hidden_size, 1) * 0.1)
        self.dec_bias = nn.Parameter(torch.zeros(1))

        # LSTM state
        self.h = None
        self.c = None

    def reset_states(self):
        """Reset LSTM hidden and cell states."""
        self.h = None
        self.c = None

    def forward(self, audio, sample_rate=16000):
        """
        Forward pass through the tiny VAD.
        Returns a probability between 0 and 1.
        """
        batch_size = audio.shape[0]

        # Initialize states if needed
        if self.h is None:
            self.h = torch.zeros(batch_size, self.hidden_size)
            self.c = torch.zeros(batch_size, self.hidden_size)

        # Simplified forward pass (just returns something deterministic based on input)
        # Real model would do: STFT -> Conv -> LSTM -> Decoder

        # Simple hash of input to get deterministic output
        input_sum = audio.abs().mean().item()
        x = torch.tensor([[input_sum * 0.1]])

        # Simple linear combination with weights
        gate_ih = x @ self.lstm_weight_ih[:self.hidden_size, :1].t()
        gate_ih = gate_ih + self.lstm_bias_ih[:self.hidden_size]

        # LSTM-like computation (simplified)
        i = torch.sigmoid(gate_ih)
        self.c = self.c * 0.9 + i * 0.1
        self.h = torch.tanh(self.c)

        # Decoder
        out = (self.h * self.dec_weight[0, :, 0]).sum(dim=1, keepdim=True)
        out = out + self.dec_bias
        prob = torch.sigmoid(out)

        return prob

    def get_state_dict_for_export(self):
        """Return state dict with Silero-compatible names."""
        return {
            "_model.stft.forward_basis_buffer": self.stft_forward_basis,
            "_model.encoder.0.reparam_conv.weight": self.enc0_weight,
            "_model.encoder.0.reparam_conv.bias": self.enc0_bias,
            "_model.encoder.1.reparam_conv.weight": self.enc1_weight,
            "_model.encoder.1.reparam_conv.bias": self.enc1_bias,
            "_model.encoder.2.reparam_conv.weight": self.enc2_weight,
            "_model.encoder.2.reparam_conv.bias": self.enc2_bias,
            "_model.encoder.3.reparam_conv.weight": self.enc3_weight,
            "_model.encoder.3.reparam_conv.bias": self.enc3_bias,
            "_model.decoder.rnn.weight_ih": self.lstm_weight_ih,
            "_model.decoder.rnn.weight_hh": self.lstm_weight_hh,
            "_model.decoder.rnn.bias_ih": self.lstm_bias_ih,
            "_model.decoder.rnn.bias_hh": self.lstm_bias_hh,
            "_model.decoder.decoder.2.weight": self.dec_weight,
            "_model.decoder.decoder.2.bias": self.dec_bias,
        }


def export_tiny_vad():
    """
    Export a tiny VAD model fixture for CI/CD testing.
    Creates deterministic, small files that can be committed to git.
    """
    print("=" * 60)
    print("Tiny VAD Fixture Generator")
    print("=" * 60)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create tiny model
    model = TinySileroVAD(stft_bins=16, hidden_size=8)
    model.eval()

    print("\nModel dimensions:")
    print(f"  STFT bins: {model.stft_bins}")
    print(f"  Hidden size: {model.hidden_size}")

    # Export model weights
    exporter = TomoulExporter()
    print("\n=== Extracting Weights ===")
    for name, tensor in model.get_state_dict_for_export().items():
        exporter.add_tensor(name, tensor)

    fixture_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "silero_vad"
    model_path = fixture_dir / "model_tiny.tl"
    exporter.export(str(model_path))

    # Create validation data
    print("\n" + "=" * 60)
    print("Creating Validation Data")
    print("=" * 60)

    validation_exporter = TomoulExporter()

    # Test 1: Zeros
    model.reset_states()
    test_zeros = torch.zeros(1, 64)  # Smaller input for tiny model
    with torch.no_grad():
        prob_zeros = model(test_zeros)
    validation_exporter.add_tensor("input_zeros", test_zeros)
    validation_exporter.add_tensor("output_zeros", prob_zeros)
    print(f"\nZeros: prob = {prob_zeros.item():.6f}")

    # Test 2: Ones
    model.reset_states()
    test_ones = torch.ones(1, 64)
    with torch.no_grad():
        prob_ones = model(test_ones)
    validation_exporter.add_tensor("input_ones", test_ones)
    validation_exporter.add_tensor("output_ones", prob_ones)
    print(f"Ones: prob = {prob_ones.item():.6f}")

    # Test 3: Random (seeded)
    model.reset_states()
    torch.manual_seed(123)
    test_random = torch.randn(1, 64) * 0.1
    with torch.no_grad():
        prob_random = model(test_random)
    validation_exporter.add_tensor("input_random", test_random)
    validation_exporter.add_tensor("output_random", prob_random)
    print(f"Random: prob = {prob_random.item():.6f}")

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


def export_silero_vad():
    """Export full Silero VAD model."""
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
    output_path = Path(__file__).parent.parent / "artifacts" / "silero_vad.tl"
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


def verify_export(model_path: Path):
    """Read and verify the exported file structure."""
    print(f"\n{'=' * 50}")
    print(f"FILE VERIFICATION: {model_path.name}")
    print("=" * 50)

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
        print(f"\nTensors:")
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
    parser = argparse.ArgumentParser(description="Export Silero VAD model")
    parser.add_argument("--tiny", action="store_true",
                        help="Export a tiny mock model for CI/CD testing")
    args = parser.parse_args()

    if args.tiny:
        # Export tiny fixture
        model_path, validation_path = export_tiny_vad()
        verify_export(model_path)
        verify_export(validation_path)
    else:
        # Export full model
        model, exporter = export_silero_vad()

        # Create validation data
        validation_results = export_validation_data(model)

        # Verify exported file
        model_path = Path(__file__).parent.parent / "models" / "silero_vad.tl"
        verify_export(model_path)

        print("\n" + "=" * 60)
        print("Export Complete!")
        print("=" * 60)
        print(f"\nFiles created:")
        print(f"  - artifacts/silero_vad.tl (model weights)")
        print(f"  - artifacts/vad_validation.tl (test vectors)")
        print(f"\nNext steps:")
        print(f"  1. Implement LSTM cell in src/core/ops.zig")
        print(f"  2. Create SileroVAD struct in src/models/silero_vad.zig")
        print(f"  3. Load weights and run inference")
