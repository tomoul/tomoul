"""Compare PyTorch vs Zig first-token logits for Qwen3.5-0.8B."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3.5-0.8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float32
).to(device)
model.eval()

prompts = [
    "<|im_start|>user",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
]

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {repr(prompt[:60])}")
    print(f"Tokens: {input_ids.shape[1]}")

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids)
    elapsed = time.perf_counter() - t0

    logits = outputs.logits[0, -1]
    top_vals, top_ids = torch.topk(logits, 10)
    top_token = top_ids[0].item()
    top_logit = top_vals[0].item()
    top_text = tokenizer.decode([top_token])

    print(f"Forward pass: {elapsed*1000:.1f}ms")
    print(f"Top token: {top_token} ({repr(top_text)}) logit={top_logit:.4f}")
    print(f"Top-5: {[(tid.item(), f'{tv.item():.2f}', repr(tokenizer.decode([tid.item()]))) for tid, tv in zip(top_ids[:5], top_vals[:5])]}")
