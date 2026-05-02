#!/usr/bin/env python3
"""
Validate Tomoul Qwen3.5-0.8B against PyTorch reference.

Creates reference logits from HuggingFace model and compares
against Tomoul CLI output.

Usage:
    python validate_qwen3_5.py --create-reference
    python validate_qwen3_5.py --validate
    python validate_qwen3_5.py --create-reference --validate
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_hf_model(model_path: str, device: str = "cpu"):
    """Load Qwen3.5-0.8B from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path} on {device}...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    elapsed = time.perf_counter() - t0
    print(f"Model loaded in {elapsed:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {next(model.parameters()).device}")

    return model, tokenizer


TEST_PROMPTS = [
    "Hello",
    "The capital of France is",
    "What is 2+2?",
    "Once upon a time",
]


def create_reference(model_path: str, output_path: str, device: str = "cpu"):
    """Generate reference logits from PyTorch model."""
    model, tokenizer = load_hf_model(model_path, device)

    references = {}

    for prompt in TEST_PROMPTS:
        print(f"\n--- Prompt: '{prompt}' ---")

        # Format same as Tomoul: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
        print(f"  Tokens ({input_ids.shape[1]}): {input_ids[0].tolist()[:20]}...")

        # Get logits at each prefill position + first decode step
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Save top-10 logits at last position (next-token prediction)
        last_logits = logits[-1].float().cpu().numpy()
        top_k = 10
        top_indices = np.argsort(last_logits)[::-1][:top_k]
        top_values = last_logits[top_indices]

        greedy_token = int(top_indices[0])
        greedy_text = tokenizer.decode([greedy_token])

        print(f"  Greedy next token: {greedy_token} = '{greedy_text}'")
        print(f"  Top-{top_k} logits:")
        for i in range(top_k):
            tok_text = tokenizer.decode([int(top_indices[i])])
            print(f"    [{int(top_indices[i]):>6}] {top_values[i]:>8.4f}  '{tok_text}'")

        # Generate a few tokens greedily
        gen_len = 20
        generated = model.generate(
            input_ids,
            max_new_tokens=gen_len,
            do_sample=False,
            temperature=1.0,
        )
        gen_ids = generated[0][input_ids.shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_ids)
        print(f"  Generated ({len(gen_ids)} tokens): '{gen_text}'")

        references[prompt] = {
            "input_ids": input_ids[0].tolist(),
            "top_k_indices": [int(x) for x in top_indices],
            "top_k_values": [float(x) for x in top_values],
            "greedy_token": greedy_token,
            "greedy_text": greedy_text,
            "generated_ids": gen_ids,
            "generated_text": gen_text,
            # Also save full logits for the last position (for detailed comparison)
            "last_logits_sample": {
                str(int(top_indices[i])): float(top_values[i])
                for i in range(top_k)
            },
        }

    # Save references
    with open(output_path, "w") as f:
        json.dump(references, f, indent=2)

    print(f"\nReference saved to {output_path}")
    return references


def validate_tomoul(
    reference_path: str,
    tomoul_exe: str,
    weights_path: str,
    tokenizer_path: str,
):
    """Validate Tomoul output against PyTorch reference."""
    with open(reference_path) as f:
        references = json.load(f)

    print(f"\nValidating Tomoul against PyTorch reference...")
    print(f"  Executable: {tomoul_exe}")
    print(f"  Weights: {weights_path}")
    print(f"  Tokenizer: {tokenizer_path}")

    passed = 0
    failed = 0

    for prompt, ref in references.items():
        print(f"\n--- Prompt: '{prompt}' ---")
        print(f"  PyTorch greedy: [{ref['greedy_token']}] '{ref['greedy_text']}'")
        print(f"  PyTorch generated: '{ref['generated_text'][:80]}'")

        # Run Tomoul
        cmd = [
            tomoul_exe, "generate", prompt,
            "--weights", weights_path,
            "--tokenizer", tokenizer_path,
            "--max-tokens", "20",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"  [FAIL] Tomoul returned error: {result.stderr[:200]}")
                failed += 1
                continue

            tomoul_output = result.stderr  # cli.zig uses std.debug.print → stderr
            print(f"  Tomoul output:\n    {tomoul_output.strip()[:200]}")

            # Check if generated text overlaps significantly
            ref_gen_tokens = ref["generated_ids"][:5]  # First 5 tokens
            # We check output text overlap since we can't easily get token IDs from CLI
            ref_text_start = ref["generated_text"][:30]

            # Simple heuristic: check if first few words match
            if ref_text_start[:10].lower() in tomoul_output.lower():
                print(f"  [PASS] Output matches reference start")
                passed += 1
            else:
                print(f"  [WARN] Output may differ (check manually)")
                print(f"    Expected start: '{ref_text_start}'")
                failed += 1

        except subprocess.TimeoutExpired:
            print(f"  [FAIL] Tomoul timed out after 120s")
            failed += 1
        except FileNotFoundError:
            print(f"  [FAIL] Tomoul executable not found: {tomoul_exe}")
            failed += 1
            break

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(references)}")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Validate Qwen3.5-0.8B implementation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                        help="HuggingFace model path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for PyTorch reference")
    parser.add_argument("--create-reference", action="store_true",
                        help="Create PyTorch reference logits")
    parser.add_argument("--validate", action="store_true",
                        help="Validate Tomoul against reference")
    parser.add_argument("--reference-path",
                        default="artifacts/qwen3_5_reference.json",
                        help="Path to reference JSON")
    parser.add_argument("--tomoul-exe",
                        default="zig-out/bin/tomoul_qwen3_5-0.8b.exe",
                        help="Path to Tomoul executable")
    parser.add_argument("--weights",
                        default="artifacts/qwen3_5_0.8b_f32.tl",
                        help="Path to .tl weights file")
    parser.add_argument("--tokenizer",
                        default="artifacts/qwen3_5_vocab.bin",
                        help="Path to tokenizer binary")
    args = parser.parse_args()

    if not args.create_reference and not args.validate:
        parser.print_help()
        return

    if args.create_reference:
        create_reference(args.model, args.reference_path, args.device)

    if args.validate:
        validate_tomoul(
            args.reference_path,
            args.tomoul_exe,
            args.weights,
            args.tokenizer,
        )


if __name__ == "__main__":
    main()
