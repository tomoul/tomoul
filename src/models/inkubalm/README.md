# InkubaLM-0.4B

African-languages LLM by Lelapa AI. Stock Llama-2-style decoder; everything
runs through `src/arch/llama.zig` with a `LlamaConfig` literal in `config.zig`.

| Field | Value |
|-------|-------|
| Hidden | 2048 |
| Layers | 8 |
| Heads / KV heads | 32 / 32 (MHA) |
| Head dim | 64 |
| FFN | 5632 (SwiGLU) |
| Vocab | 61,788 |
| Tied embeddings | yes |
| RoPE θ | 10,000 |
| Source | `lelapa/InkubaLM-0.4B` (HF) |

## Status

- [x] Config wired (`config.zig`)
- [x] Model wrapper with greedy generate (`model.zig`)
- [x] Token-id-driven CLI (`cli.zig`)
- [x] Registry entry
- [ ] **Tokenizer** — SentencePiece reader not implemented in Zig yet. Drive
      tokenization from Python, pipe ids in.
- [ ] Validation harness vs HF transformers
- [ ] Quantized weight variants (Q8_K)

## Running it

Until the SentencePiece reader lands, tokenize externally:

```bash
# 1) Get prompt token ids from HF transformers
ids="$(python -c 'from transformers import AutoTokenizer; \
  t = AutoTokenizer.from_pretrained("lelapa/InkubaLM-0.4B"); \
  print(" ".join(map(str, t("Habari").input_ids)))')"

# 2) Run Tomoul, get generated token ids
zig build -Dmodel=inkubalm-0.4b -Doptimize=ReleaseFast
./zig-out/bin/tomoul_inkubalm-0.4b artifacts/inkubalm_0.4b.safetensors --max-new 32 -- $ids

# 3) Decode back to text
python -c 'from transformers import AutoTokenizer; \
  t = AutoTokenizer.from_pretrained("lelapa/InkubaLM-0.4B"); \
  print(t.decode([... paste ids from stdout ...]))'
```

## Validation plan

Once the tokenizer lands, a Python harness will:

1. Load InkubaLM via HF transformers, capture logits for 5 prompts at
   each step.
2. Load InkubaLM via Tomoul, capture matching logits.
3. Assert max-abs error < 1e-3 in f32, < 1e-2 in bf16.
