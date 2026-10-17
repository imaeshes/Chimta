#!/usr/bin/env python
"""
SafetyDecision — training.py

Supervised fine-tuning (full SFT) of **Qwen/Qwen3.5-0.8B** on CAMERA-FEED
decision data, in the style of Valen (Liuziyu77/Valen):

    one record  =  a state (video or image)  +  questions  +  target probabilities
    questions   =  {"choice": pick 1 of N, "noul": true/false, "score": grade 0..K-1}

The model learns to read the media + compiled question block and reply with a
STRICT compact JSON decision, e.g.:

    {"activity": "fall", "person_down": true, "risk": 2}

At inference the decision is turned back into probabilities over candidates
(see inference.py --probabilities). This keeps the training target short and
stable, while the Valen-style *interface* (state -> probabilities) survives.

File map (vs. the Valen repo, for your learning):
    prompt building here          ~ valen/data/compilers/qwen.py (compiler)
    dataset/validation here       ~ valen/data/schema.py
    training loop = HF Trainer    ~ valen/training/runner.py + sft.py
    answer extraction from labels ~ valen/data/group_questions.py

Dataset: metadata.jsonl next to the media, ONE record per line (collapsed!):

{"video": "clips/cam1_0142.mp4",
 "state": "Monitor this camera clip for incidents.",
 "questions": {
   "activity": {"type": "choice",
                "instructions": "Which activity is the main event?",
                "criteria": {"fall": "A person falls or is on the ground unexpectedly",
                              "fight": "Two or more people fighting",
                              "normal": "No unusual activity"}},
   "person_down": {"type": "noul", "instructions": "A person has fallen and does not get up."},
   "risk": {"type": "score", "instructions": "Rate the safety risk.",
            "criteria": ["Calm", "Unusual motion", "Fall happening", "Fight starting", "Emergency"]}},
 "targets": {
   "activity": {"fall": 1.0, "fight": 0.0, "normal": 0.0},
   "person_down": {"true": 1.0, "false": 0.0},
   "risk": {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0}}}

Image records replace "video" with "image": "frames/x.jpg".
Use "image" for fall/static events (cheap!), "video" only for fights.
Unlabeled = omit "targets" for that question id (its argmax answer is skipped).

Usage (Ubuntu box, 5070 Ti):
    python training.py --data ./dataset --out ./ckpt-safety-v1
"""

import argparse
import json
import os
import random
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
SYSTEM_PROMPT = (
    "You are a camera-safety decision model. Watch the provided media, read the "
    "questions, and answer with STRICT compact JSON: one key per question id. "
    "choice -> one option name as a string; noul -> true/false; score -> one "
    "integer index. Output the JSON object only, no other text."
)


# ----------------------------------------------------------------- args/dataset

def parse_args():
    p = argparse.ArgumentParser(description="Full SFT of Qwen3.5 on Valen-style decision data")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", required=True, help="dir with metadata.jsonl, or the .jsonl path")
    p.add_argument("--out", required=True, help="output checkpoint dir")
    p.add_argument("--frames", type=int, default=8, help="frames sampled per video record")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-acc", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024, help="token cap per sample")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optim", default="adamw_bnb_8bit", help="adamw_bnb_8bit needs bitsandbytes; else adamw_torch")
    p.add_argument("--no-freeze-vision", action="store_true",
                   help="vision tower is frozen by default (VRAM-safe); pass to train it too")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if os.path.isdir(args.data):
        args.data = os.path.join(args.data, "metadata.jsonl")
    return args


def load_records(path):
    """Load + validate records (Valen-style schema checks, simplified)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: not valid JSON ({e})")
            media = r.get("video") or r.get("image")
            if not media:
                raise SystemExit(f"{path}:{lineno}: needs a 'video' or 'image' path")
            if not r.get("state"):
                raise SystemExit(f"{path}:{lineno}: missing 'state'")
            qs = r.get("questions") or {}
            tg = r.get("targets") or {}
            if not qs:
                raise SystemExit(f"{path}:{lineno}: missing 'questions'")
            for qid, q in qs.items():
                if q.get("type") not in ("choice", "noul", "score"):
                    raise SystemExit(f"{path}:{lineno}: question '{qid}' needs type choice|noul|score")
                if qid not in tg:
                    raise SystemExit(f"{path}:{lineno}: question '{qid}' has no target "
                                     f"(delete it from questions, or add targets['{qid}'])")
                probs = tg[qid]
                if abs(sum(probs.values()) - 1.0) > 1e-3:
                    raise SystemExit(f"{path}:{lineno}: target '{qid}' probabilities must sum to 1.0")
            rows.append({"video": r.get("video"), "image": r.get("image"),
                         "state": r["state"], "questions": qs, "targets": tg})
    return rows


# ------------------------------------------------------------------ prompt text

def build_prompt_text(rec):
    """Compile state + questions into the request text.

    This render is TRAIN/INFERENCE-IDENTICAL — that is the Valen lesson: a
    decision model is mostly a *prompt contract*. Keep this function the single
    source of truth and import/reuse it in inference.py.
    """
    lines = [rec["state"], "", "Questions:"]
    for qid, q in rec["questions"].items():
        t = q["type"]
        lines.append(f'- "{qid}" ({t}): {q.get("instructions", "")}')
        if t == "choice":
            opts = ", ".join(f"{k}={v}" for k, v in q["criteria"].items())
            lines.append(f"  Options: {opts}")
        elif t == "noul":
            lines.append("  Answer true or false.")
        elif t == "score":
            grades = "; ".join(f"{i}={c}" for i, c in enumerate(q["criteria"]))
            lines.append(f"  Grades: {grades}")
    lines += ["", "Answer (JSON only):"]
    return "\n".join(lines)


def build_target_text(rec):
    """Compact JSON answer used as the supervised completion."""
    out = {}
    for qid, q in rec["questions"].items():
        prob = rec["targets"][qid]
        out[qid] = argmax_answer(q["type"], q, prob)
    return json.dumps(out, separators=(",", ":"))


def argmax_answer(qtype, q, prob):
    """Reduce a target distribution to the single trained answer (JSON value).

    choice -> winning option NAME (str), noul -> bool, score -> winning index (int).
    The answer types match the system prompt the model is trained to obey.
    """
    if qtype == "noul":
        return prob.get("true", 0.0) >= prob.get("false", 0.0)
    if qtype == "score":
        best = max(((int(k), v) for k, v in prob.items()), key=lambda kv: kv[1])
        return best[0]  # integer grade index, e.g. 2
    if q.get("criteria") and any(k not in q["criteria"] for k in prob):
        raise SystemExit(f"target has keys not in criteria: {sorted(prob)}")
    return max(prob.items(), key=lambda kv: kv[1])[0]


# ----------------------------------------------------------------- frames/media

def load_media(path, is_video, k):
    """Video -> k uniformly-sampled RGB frames; image -> single frame list."""
    if not is_video:
        return [Image.open(path).convert("RGB")]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total and total > 0:
        for idx in np.linspace(0, total - 1, k).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(Image.fromarray(f[:, :, ::-1]))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return frames


# ---------------------------------------------------------------------- dataset

class DecisionDataset(Dataset):
    """Media → processor-ready tensor dicts with completion-only labels."""

    def __init__(self, records, processor, root, frames, max_length):
        self.proc = processor
        self.root = root
        self.k = frames
        self.max_length = max_length
        self.rec = records

    def __len__(self):
        return len(self.rec)

    def __getitem__(self, i):
        r = self.rec[i]
        media_path = r["video"] or r["image"]
        if not os.path.isabs(media_path) and not os.path.exists(media_path):
            media_path = os.path.join(self.root, media_path)
        media = load_media(media_path, is_video=r["video"] is not None, k=self.k)

        if r["video"] is not None:
            media_content = [{"type": "video", "video": media, "max_frames": self.k}]
        else:
            media_content = [{"type": "image", "image": media[0]}]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": media_content + [{"type": "text", "text": build_prompt_text(r)}]},
            {"role": "assistant", "content": build_target_text(r)},
        ]
        batch = self._run(messages)
        plen = self._prompt_len(messages)
        labels = batch["input_ids"].clone()
        labels[:, : min(plen, batch["input_ids"].shape[1] - 1)] = -100  # mask prompt tokens

        if batch["input_ids"].shape[1] > self.max_length:  # keep the TAIL (answer side)
            batch["input_ids"] = batch["input_ids"][:, -self.max_length:]
            labels = labels[:, -self.max_length:]
            if "attention_mask" in batch:
                batch["attention_mask"] = batch["attention_mask"][:, -self.max_length:]
        batch["labels"] = labels
        return batch

    def _run_processor(self, messages):
        text = self.proc.apply_chat_template(messages, tokenize=False)
        videos, images = [], []
        for m in messages:
            for c in (m.get("content") or []):
                if isinstance(c, dict) and c.get("type") == "video":
                    videos.append(c["video"])
                if isinstance(c, dict) and c.get("type") == "image":
                    images.append(c["image"])
        enc = self.proc(text=[text], videos=videos, images=images,
                        return_tensors="pt", padding=False)
        return {k: (v if not hasattr(v, "dim") or v.dim() > 0 else v.unsqueeze(0))
                for k, v in enc.items()}

    def _prompt_len(self, messages):
        """Token length of prompt-only chat = generation-prompt boundary."""
        try:
            pre = [m for m in messages if m["role"] != "assistant"]
            text = self.proc.apply_chat_template(pre, tokenize=False, add_generation_prompt=True)
            videos, images = [], []
            for m in pre:
                for c in (m.get("content") or []):
                    if isinstance(c, dict) and c.get("type") == "video":
                        videos.append(c["video"])
                    if isinstance(c, dict) and c.get("type") == "image":
                        images.append(c["image"])
            enc = self.proc(text=[text], videos=videos, images=images, return_tensors="pt", padding=False)
            if "input_ids" in enc:
                return enc["input_ids"].shape[-1]
            return 0
        except Exception:
            return 0  # fallback: no masking (full-sequence loss; works, just less clean)


def make_collate(pad_id):
    def collate(batch):
        """Stack a list of per-sample processor dicts into one batch.

        Works for TEXT keys (input_ids/labels/attention_mask, shape [1, T]) and
        for visual processor outputs (pixel values etc.). No padding is applied:
        with the default per_device_train_batch_size=1 every sample trains
        un-padded, which is both simplest and VRAM-safest.
        """
        out = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            try:
                out[key] = torch.cat(vals, dim=0)
            except (RuntimeError, TypeError):
                out[key] = torch.stack(vals)
        return out
    return collate


# ------------------------------------------------------------------------- main

def main():
    args = parse_args()
    set_seed(args.seed)

    records = load_records(args.data)
    if not records:
        raise SystemExit("no valid records found")
    root = os.path.dirname(os.path.abspath(args.data))
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = int(len(records) * args.val_frac) if args.val_frac > 0 else 0
    val_rec, train_rec = records[:n_val], records[n_val:]
    print(f"[data] train={len(train_rec)} valid={n_val} frames/clip={args.frames}")

    processor = AutoProcessor.from_pretrained(args.model)
    if getattr(processor, "tokenizer", None) and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", use_cache=False)
    model.config.use_cache = False

    if not args.no_freeze_vision:
        vis = getattr(model, "visual", None) or getattr(model, "vision_tower", None)
        if vis is not None:
            n = sum(p.numel() for p in vis.parameters())
            for p in vis.parameters():
                p.requires_grad = False
            print(f"[model] vision tower frozen ({n/1e6:.0f}M params; "
                  f"pass --no-freeze-vision to train it)")

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
        eval_strategy="steps" if n_val else "no",
        eval_steps=args.eval_steps if n_val else None,
        load_best_model_at_end=bool(n_val),
        metric_for_best_model="eval_loss" if n_val else None,
        greater_is_better=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim=args.optim,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=DecisionDataset(train_rec, processor, root, args.frames, args.max_length),
                      eval_dataset=DecisionDataset(val_rec, processor, root, args.frames, args.max_length) if n_val else None,
                      data_collate=make_collate(processor.tokenizer.eos_token_id))

    kwargs = {}
    if args.resume and os.path.isdir(args.out) and any(d.startswith("checkpoint") for d in os.listdir(args.out)):
        kwargs["resume_from_checkpoint"] = True
    trainer.train(**kwargs)

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    if n_val:
        print("[val]", trainer.evaluate())
    print(f"[done] checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
