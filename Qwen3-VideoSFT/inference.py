#!/usr/bin/env python
"""
Inference for the full-SFT Qwen3 checkpoint produced by training.py.

Setup: same torch (cu128) + transformers environments as training.py.

Usage:
    # single prompt
    python inference.py --ckpt ./ckpt-qwen3-fullsft --prompt "Describe this clip: a kid scores a goal"

    # interactive multi-turn chat
    python inference.py --ckpt ./ckpt-qwen3-fullsft --chat

    # knobs: temperature / top-p / length
    python inference.py --ckpt ./ckpt-qwen3-fullsft --prompt "..." --temperature 0.6 --max-new-tokens 300
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3 full-SFT inference")
    p.add_argument("--ckpt", required=True, help="output dir from training.py (or a base model id)")
    p.add_argument("--prompt")
    p.add_argument("--chat", action="store_true", help="interactive multi-turn chat")
    p.add_argument("--system", default="You are a helpful video understanding assistant.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--thinking", action="store_true",
                   help="off by default: Qwen3 thinking blocks disabled for short, direct answers")
    return p.parse_args()


def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    return model, tok


def make_input_ids(tok, system, history, enable_thinking):
    """Chat-template encode; tolerates templates without the enable_thinking knob."""
    messages = [{"role": "system", "content": system}] + history
    for attempt in (1, 2):
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            if attempt == 1 and not enable_thinking:
                continue  # template may not accept enable_thinking
            raise


def generate(model, tok, system, history, opts):
    input_ids = make_input_ids(tok, system, history, opts.thinking)
    inputs = torch.tensor([input_ids], device=model.device)
    gen_kwargs = dict(
        max_new_tokens=opts.max_new_tokens,
        do_sample=opts.temperature > 0,
        repetition_penalty=opts.repetition_penalty,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    if opts.temperature > 0:
        gen_kwargs.update(temperature=opts.temperature, top_p=opts.top_p)
    with torch.no_grad():
        out = model.generate(inputs, **gen_kwargs)
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main():
    opts = parse_args()
    model, tok = load_model(opts.ckpt)

    if opts.chat:
        history = []
        print("Qwen3 chat — type 'exit' to quit")
        while True:
            try:
                user = input("\nyou: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user or user.lower() in ("exit", "quit"):
                break
            history.append({"role": "user", "content": user})
            reply = generate(model, tok, opts.system, history, opts)
            history.append({"role": "assistant", "content": reply})
            print(f"\nbot: {reply}")
        return

    if not opts.prompt:
        raise SystemExit("pass --prompt or use --chat")

    reply = generate(model, tok, opts.system, [{"role": "user", "content": opts.prompt}], opts)
    print(reply)


if __name__ == "__main__":
    main()
