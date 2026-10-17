#!/usr/bin/env python
"""
Full-parameter SFT (fullfat) of a Qwen3 model on a video-annotated text dataset.
Run on: RTX 5070 Ti (16 GB VRAM, Blackwell sm_120 -> requires torch built for CUDA 12.8).

Dataset format (metadata.jsonl — one JSON object per line):
    {"video": "clips/001.mp4", "prompt": "Describe what happens.", "response": "A player scores..."}
Qwen3 is text-only: each video must be paired with its prompt(s) + target response(s).
The "video" field is kept purely as a provenance key.

Setup (Ubuntu box):
    pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
    pip install transformers accelerate bitsandbytes safetensors huggingface_hub

Example:
    python training.py --data ./dataset --out ./ckpt-qwen3-fullsft --epochs 3
"""

import argparse
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Full-parameter SFT of Qwen3")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B",
                   help="HF id or local path (Qwen3-0.6B / 1.7B / 4B; 4B needs bigger --grad-acc and --batch-size 1)")
    p.add_argument("--data", required=True, help="dir containing metadata.jsonl (or direct path to the jsonl)")
    p.add_argument("--out", required=True, help="output checkpoint dir")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=4, help="per-device batch (0.6B bf16 fits 8 on 16 GB)")
    p.add_argument("--grad-acc", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optim", default="adamw_bnb_8bit",
                   help="adamw_bnb_8bit is the VRAM-safe default; use adamw_torch if bnb unavailable")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --out")
    p.add_argument("--save-steps", type=int, default=200)
    args = p.parse_args()
    if os.path.isdir(args.data):
        args.data = os.path.join(args.data, "metadata.jsonl")
    return args


def load_examples(data_path):
    ex = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("question") or row.get("user")
            response = row.get("response") or row.get("answer") or row.get("assistant")
            if not prompt or not response:
                continue  # malformed rows are skipped, logged counters below
            ex.append({"video": row.get("video", ""), "prompt": prompt, "response": response})
    return ex


class VideoSFTDataset(Dataset):
    """Chat-formatted SFT samples with completion-only loss (prompt tokens masked)."""

    def __init__(self, examples, tokenizer, max_len=1024, system_prompt=None):
        self.tok = tokenizer
        self.max_len = max_len
        self.system = system_prompt or "You are a helpful video understanding assistant."
        self.rows = examples

    def __len__(self):
        return len(self.rows)

    def _enc(self, messages):
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return ids if isinstance(ids, list) else ids["input_ids"]

    def __getitem__(self, i):
        row = self.rows[i]
        # keep the video reference in the user turn so the model learns to ground answers
        user_text = f"[video: {row['video']}]\n{row['prompt']}" if row["video"] else row["prompt"]

        prefix_msgs = [{"role": "system", "content": self.system},
                       {"role": "user", "content": user_text}]
        full_msgs = prefix_msgs + [{"role": "assistant", "content": row["response"]}]

        prefix_ids = self.tok.apply_chat_template(prefix_msgs, tokenize=True, add_generation_prompt=True)
        full_ids = self._enc(full_msgs)

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]

        # truncate from the right (keep EOS)
        if len(full_ids) > self.max_len:
            full_ids = full_ids[-self.max_len:]
            labels = labels[-self.max_len:]

        return {"input_ids": full_ids, "labels": labels}


def make_collator(pad_id, pad_token):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
    return collate


def main():
    args = parse_args()
    set_seed(args.seed)

    examples = load_examples(args.data)
    n_valid = len(examples)
    if n_valid == 0:
        raise SystemExit("No valid prompt/response pairs found in the dataset file.")
    skipped = sum(1 for _ in open(args.data, encoding="utf-8")) - n_valid
    if skipped:
        print(f"[data] skipped {skipped} malformed row(s)")
    print(f"[data] {n_valid} SFT examples")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # flash-attn2 optional; sdpa is safe on Blackwell
        use_cache=False,              # required with gradient checkpointing
    )
    model.config.use_cache = False

    ds = VideoSFTDataset(examples, tok, max_len=args.max_len)
    collator = make_collator(tok.pad_token_id, tok.pad_token)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        warmup_ratio=args.warmup,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    train_kwargs = {}
    if args.resume and os.path.isdir(args.out) and any(
        d.startswith("checkpoint") for d in os.listdir(args.out)
    ):
        train_kwargs["resume_from_checkpoint"] = True

    trainer.train(**train_kwargs)

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] full SFT checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
