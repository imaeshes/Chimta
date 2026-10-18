#!/usr/bin/env python3
"""Single-GPU Valen training for a 16 GB GPU (RTX 5070 Ti).

--architecture qwen            Qwen3.5 + mixer/bilinear head (released 0.8B/2B path)
--architecture dual_encoder    ModernBERT + DINOv3 + fusion/reader (Valen-DE)

Two objectives share one loop:
  --mode sft   distribution cross-entropy (+ optional Brier) on labeled candidates
  --mode rlcd  Valen's RLCD over the candidate softmax (see valen/training/rlcd.py)

Data is Valen JSONL (docs/data-format.md). Dual-encoder records may include one image,
not video. Qwen records may include image and video.
"""
import argparse
from contextlib import nullcontext
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import zipfile

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = "Valen-local"
CONFIG_FILE = "valen_local.json"
WEIGHTS_FILE = "model.safetensors"
STATE_FILE = "trainer_state.pt"
REFERENCE_FILE = "reference.safetensors"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = REPO_ROOT / "data" / "hf"

# Official Valen JSONL + images (docs/technical.md). Media URLs are relative to train.jsonl.
VALEN_TRAIN_HUB = "Valen-Team/Valen-Training-General-100k"
VALEN_EVAL_HUB = "Valen-Team/Valen-Eval-General-5k"
VALEN_GAME_HUB = "Valen-Team/Valen-Eval-Game"
VALEN_SFT_COUNT = 70_000  # first 70k SFT, remaining 30k RLCD on the official 100k file
DATA_ALIASES = {
    "valen": VALEN_TRAIN_HUB,
    "general": VALEN_TRAIN_HUB,
    "train": VALEN_TRAIN_HUB,
    "eval": VALEN_EVAL_HUB,
    "game": VALEN_GAME_HUB,
    "smoke": str(REPO_ROOT / "data" / "smoke" / "train.jsonl"),
}

# Matches the released Valen checkpoints (config.json -> valen_media_kwargs).
DEFAULT_MEDIA_KWARGS = {
    "images_kwargs": {"size": {"longest_edge": 262144, "shortest_edge": 65536}},
    "videos_kwargs": {"fps": None, "num_frames": 16,
                      "size": {"longest_edge": 2097152, "shortest_edge": 131072}},
}
DEFAULT_HEAD = {"head_type": "mixer", "head_width": 256, "head_layers": 2,
                "head_token_hidden_dim": 16, "head_channel_hidden_dim": 1024}
DEFAULT_TEXT_ENCODER = "answerdotai/ModernBERT-base"
DEFAULT_VISION_ENCODER = "facebook/dinov3-vitb16-pretrain-lvd1689m"


# --------------------------------------------------------------------------- data

def candidates(question):
    kind = question.get("type")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError("instructions must be nonempty text")
    criteria = question.get("criteria")
    if kind == "noul":
        # Exact Valen candidate text; the released heads were trained on it.
        return [("true", "True"), ("false", "False")]
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise ValueError("Choice needs 1-255 named criteria")
        result = list(criteria.items())
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("Score criteria must be an ordered list of 2-10 descriptions")
        result = [(str(i), text) for i, text in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported question type: {kind}")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip() for k, v in result):
        raise ValueError("Candidate names and descriptions must be nonempty strings")
    return result


def target_distribution(target, keys):
    if target is None or target.get("probabilities") is None:
        return None
    probs = target["probabilities"]
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError(f"Probability keys must exactly match {keys}")
    values = [probs[k] for k in keys]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Probabilities must be finite nonnegative numbers")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Target probabilities must sum to 1")
    return [float(v) for v in values]


def resolve_dataset(spec):
    """Local JSONL, a directory that contains one, a Hub id, or an alias (`valen`, `eval`, `smoke`)."""
    if not spec:
        raise ValueError("Dataset spec is empty")
    spec = DATA_ALIASES.get(spec, spec)
    path = Path(spec)
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        for name in ("train.jsonl", "data.jsonl"):
            if (path / name).is_file():
                return (path / name).resolve()
        jsonl = sorted(path.glob("*.jsonl"))
        if len(jsonl) == 1:
            return jsonl[0].resolve()
        raise ValueError(f"{path}: need train.jsonl (or a single .jsonl)")
    if "/" not in spec or spec.startswith(".") or "\\" in spec:
        raise FileNotFoundError(spec)
    from huggingface_hub import snapshot_download
    dest = DATA_CACHE / spec.replace("/", "--")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Dataset {spec} → {dest}", flush=True)
    snapshot_download(repo_id=spec, repo_type="dataset", local_dir=str(dest))
    assets, archive = dest / "assets", dest / "assets.zip"
    if archive.is_file() and not assets.is_dir():
        print(f"Extracting {archive.name}", flush=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(dest)
    for name in ("train.jsonl", "test.jsonl", "val.jsonl"):
        if (dest / name).is_file():
            return (dest / name).resolve()
    jsonl = sorted(p for p in dest.glob("*.jsonl") if p.name != "dataset_info.json")
    if len(jsonl) == 1:
        return jsonl[0].resolve()
    raise FileNotFoundError(f"{spec}: no train.jsonl after download")


def slice_records(records, mode, data_slice, jsonl_path):
    """Valen's 100k recipe: SFT on the first 70k, RLCD on the last 30k. Other files are left whole."""
    official = VALEN_TRAIN_HUB.replace("/", "--") in Path(jsonl_path).as_posix()
    choice = data_slice
    if choice == "auto":
        choice = (mode if official else "all")
    if choice == "all" or not official:
        return records, choice
    if choice == "sft":
        return records[:VALEN_SFT_COUNT], choice
    if choice == "rlcd":
        return records[VALEN_SFT_COUNT:], choice
    raise ValueError(f"Unknown data slice: {data_slice}")


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                request = record["request"]
                if not isinstance(request["state"], (str, dict)):
                    raise ValueError("state must be text or a messages object")
                if not isinstance(request["questions"], dict) or not request["questions"]:
                    raise ValueError("questions must be a nonempty mapping")
                targets = record.get("targets") or {}
                if set(targets) - set(request["questions"]):
                    raise ValueError("Unknown target question IDs")
                for qid, question in request["questions"].items():
                    target_distribution(targets.get(qid), [k for k, _ in candidates(question)])
                records.append(record)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: empty dataset")
    return records


def is_labeled(record):
    targets = record.get("targets") or {}
    return any(t is not None and t.get("probabilities") is not None for t in targets.values())


@dataclass
class Branch:
    inputs: dict
    candidate_positions: list
    decision_position: int
    context_span: tuple
    instruction_span: tuple
    candidate_spans: list


@dataclass
class Question:
    qid: str
    kind: str
    keys: list
    descriptions: list
    branches: list
    target: list | None


@dataclass
class State:
    group_id: str
    questions: list
    tokens: int


class Compiler:
    """Shared state prompt, then one suffix per question (one per level for Score)."""

    def __init__(self, processor, media_root=".", max_length=8192, media_kwargs=None):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.media_root = Path(media_root)
        self.max_length = max_length
        self.media_kwargs = media_kwargs or {}
        self.special = [t for t in self.tokenizer.all_special_tokens if t]

    def _check_text(self, text):
        if any(t in text for t in self.special):
            raise ValueError("Reserved tokenizer control token in input text")

    def _messages(self, state):
        messages = ([{"role": "user", "content": state}] if isinstance(state, str)
                    else copy.deepcopy(state["messages"]))
        for message in messages:
            if message["role"] not in {"user", "assistant", "system"}:
                raise ValueError("Unsupported state role")
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            for item in message["content"]:
                kind = item["type"]
                if kind == "text":
                    self._check_text(item["text"])
                    continue
                if kind not in {"image_url", "video_url"}:
                    raise ValueError(f"Unsupported content type: {kind}")
                url = item[kind]["url"]
                if "://" in url:
                    raise ValueError("Media must be a local file path")
                path = Path(url)
                path = (path if path.is_absolute() else self.media_root / path).resolve(strict=True)
                item.clear()
                item.update(type=kind.removesuffix("_url"), path=str(path))
        return messages

    def compile(self, record, rng=None, labeled_only=False):
        targets = record.get("targets") or {}
        selected = []
        for qid, question in record["request"]["questions"].items():
            pairs = candidates(question)
            target = targets.get(qid)
            if labeled_only and target_distribution(target, [k for k, _ in pairs]) is None:
                continue
            if rng is not None and question["type"] == "choice":
                rng.shuffle(pairs)  # Labels follow their keys, so shuffling is label-safe.
            selected.append((qid, question, pairs, target))
        group_id = str(record.get("group_id", ""))
        if not selected:
            return State(group_id, [], 0)
        messages = self._messages(record["request"]["state"])
        base = dict(self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True,
            return_tensors="pt", processor_kwargs=dict(self.media_kwargs)))
        base.pop("video_metadata", None)
        base_length = base["input_ids"].shape[1]
        compiled, total = [], 0
        for qid, question, pairs, target in selected:
            for value in [question["instructions"]] + [v for pair in pairs for v in pair]:
                self._check_text(value)
            groups = [[pair] for pair in pairs] if question["type"] == "score" else [pairs]
            branches = []
            for group in groups:
                suffix = []

                def append(text, char_span=None):
                    start = base_length + len(suffix)
                    if char_span is None:
                        suffix.extend(self.tokenizer.encode(text, add_special_tokens=False))
                        return None
                    encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                    suffix.extend(encoded["input_ids"])
                    indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                               if a < char_span[1] and b > char_span[0]]
                    if not indices:
                        raise ValueError("Empty token span for decision role")
                    return (start + indices[0], start + indices[-1] + 1)

                suffix.extend(self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
                prefix = "Task: " + question["type"] + "\nQuestion: "
                instruction_span = append(prefix + question["instructions"] + "\nCandidates:\n",
                                          (len(prefix), len(prefix) + len(question["instructions"])))
                positions, candidate_spans = [], []
                for key, description in group:
                    prefix = key + ": " if question["type"] != "score" else ""
                    candidate_spans.append(append(prefix + description,
                                                  (len(prefix), len(prefix) + len(description))))
                    positions.append(base_length + len(suffix) - 1)
                    append("\n")
                append("Decision:")
                length = base_length + len(suffix)
                if length > self.max_length:
                    raise ValueError(f"{qid}: {length} tokens exceed max_length={self.max_length}")
                inputs = dict(base)
                inputs["input_ids"] = torch.cat(
                    [base["input_ids"], torch.tensor([suffix], dtype=torch.long)], dim=1)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                if "mm_token_type_ids" in base:
                    inputs["mm_token_type_ids"] = torch.cat(
                        [base["mm_token_type_ids"], torch.zeros((1, len(suffix)), dtype=torch.long)], dim=1)
                branches.append(Branch(inputs, positions, length - 1, (0, base_length),
                                       instruction_span, candidate_spans))
                total += length
            keys = [k for k, _ in pairs]
            compiled.append(Question(qid, question["type"], keys, [v for _, v in pairs], branches,
                                     target_distribution(target, keys)))
        return State(group_id, compiled, total)


# ------------------------------------------------ dual-encoder compiler (ModernBERT + DINOv3)

def parallel_candidates(question):
    """Valen dual-encoder candidate text (valen/data/compilers/dual_encoder.py)."""
    kind, criteria = question.get("type"), question.get("criteria")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError("instructions must be nonempty text")
    if kind == "noul":
        if criteria is None:
            criteria = {"true": "The stated condition is true.", "false": "The stated condition is false."}
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise ValueError("Noul criteria must define true and false")
        return [(k, criteria[k]) for k in ("true", "false")]
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise ValueError("Choice needs 1-255 named criteria")
        result = [(k, (v if isinstance(v, str) and v.strip() else k)) for k, v in criteria.items()]
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("Score criteria must be an ordered list of 2-10 descriptions")
        result = [(str(i), text) for i, text in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported question type: {kind}")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip() for k, v in result):
        raise ValueError("Candidate names and descriptions must be nonempty strings")
    return result


@dataclass
class ParallelState:
    group_id: str
    questions: list
    state_tokens: list
    instruction_tokens: list
    candidate_tokens: list
    candidate_offsets: list
    pixel_values: torch.Tensor | None
    visual_mask: torch.Tensor | None
    tokens: int
    media: list


class ParallelCompiler:
    """Independent state / instruction / candidate sequences. Pads only inside a text mini-batch."""

    def __init__(self, tokenizer, image_processor, media_root=".", image_size=384, patch_size=16,
                 state_max_length=1024, question_max_length=128, candidate_max_length=128, max_questions=256):
        self.tokenizer, self.image_processor = tokenizer, image_processor
        self.media_root = Path(media_root)
        self.image_size, self.patch_size = image_size, patch_size
        self.limits = (state_max_length, question_max_length, candidate_max_length)
        self.max_questions = max_questions
        if tokenizer.pad_token_id is None:
            raise ValueError("Text encoder tokenizer needs a pad token")

    def _tokens(self, text, limit, role):
        ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if not ids or len(ids) > limit:
            raise ValueError(f"{role}: {len(ids)} tokens exceed budget={limit} or empty sequence; no truncation")
        return ids

    def _state(self, state):
        messages = [{"role": "user", "content": state}] if isinstance(state, str) else state["messages"]
        text, media = [], []
        for message in messages:
            if message["role"] not in {"user", "assistant", "system"}:
                raise ValueError("Unsupported state role")
            text.append(f"{message['role']}:\n")
            content = message["content"]
            content = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for item in content:
                if item["type"] == "text":
                    text.append(item["text"])
                elif item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    if "://" in url:
                        raise ValueError("Media must be a local file path")
                    path = Path(url)
                    path = (path if path.is_absolute() else self.media_root / path).resolve(strict=True)
                    media.append(str(path))
                    text.append("\n[image]\n")
                else:
                    raise ValueError("Dual encoder supports text and a single image, not video")
            text.append("\n")
        if len(media) > 1:
            raise ValueError("Dual encoder supports at most one image per state")
        return "".join(text), media

    def _image(self, path):
        from PIL import Image, ImageOps
        import numpy as np
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            size, patch = self.image_size, self.patch_size
            ratio = size / max(image.size)
            width, height = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
            image = image.resize((width, height), resample=self.image_processor.resample)
            mean = np.array(self.image_processor.image_mean, dtype=np.float32)
            std = np.array(self.image_processor.image_std, dtype=np.float32)
            canvas = Image.new("RGB", (size, size), tuple(np.rint(mean * 255).astype(int)))
            left, top = (size - width) // 2, (size - height) // 2
            canvas.paste(image, (left, top))
            pixels = (np.asarray(canvas, dtype=np.float32) / 255.0 - mean) / std
        axis = torch.arange(size // patch) * patch
        valid_x = (axis < left + width) & (axis + patch > left)
        valid_y = (axis < top + height) & (axis + patch > top)
        valid = torch.cat([torch.ones(1, dtype=torch.bool), (valid_y[:, None] & valid_x[None, :]).flatten()])
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0), valid.unsqueeze(0)

    def compile(self, record, rng=None, labeled_only=False):
        items = record["request"]["questions"]
        if not isinstance(items, dict) or not 1 <= len(items) <= self.max_questions:
            raise ValueError(f"Expected 1-{self.max_questions} questions")
        selected = []
        for qid, question in items.items():
            pairs = parallel_candidates(question)
            target = (record.get("targets") or {}).get(qid)
            if labeled_only and target_distribution(target, [k for k, _ in pairs]) is None:
                continue
            if rng is not None and question["type"] == "choice":
                rng.shuffle(pairs)
            selected.append((qid, question, pairs, target))
        group_id = str(record.get("group_id", ""))
        if not selected:
            return ParallelState(group_id, [], [], [], [], [0], None, None, 0, [])
        state_text, media = self._state(record["request"]["state"])
        state_ids = self._tokens(state_text, self.limits[0], "state")
        questions, instructions, candidates, offsets = [], [], [], [0]
        for qid, q, pairs, target in selected:
            keys = [k for k, _ in pairs]
            instructions.append(self._tokens(q["instructions"], self.limits[1], "instructions"))
            for key, value in pairs:
                text = json.dumps({"name": key, "description": value}, ensure_ascii=False) if q["type"] == "choice" else value
                candidates.append(self._tokens(text, self.limits[2], "candidate"))
            offsets.append(len(candidates))
            questions.append(Question(qid, q["type"], keys, [v for _, v in pairs], [],
                                      target_distribution(target, keys)))
        pixels, mask = self._image(media[0]) if media else (None, None)
        tokens = len(state_ids) + sum(map(len, instructions)) + sum(map(len, candidates))
        if mask is not None:
            tokens += int(mask.numel())
        return ParallelState(group_id, questions, state_ids, instructions, candidates, offsets,
                             pixels, mask, tokens, media)


# ------------------------------------------------------------------------ heads

class BilinearHead(nn.Module):
    def __init__(self, hidden_size, projection_dim=256):
        super().__init__()
        self.decision = nn.Linear(hidden_size, projection_dim, bias=False)
        self.candidate = nn.Linear(hidden_size, projection_dim, bias=False)
        self.scale = math.sqrt(projection_dim)

    def extract_features(self, hidden, branch):
        return hidden[:, [*branch.candidate_positions, branch.decision_position], :]

    def score_features(self, features):
        d = self.decision(features[0, -1].float())
        c = self.candidate(features[0, :-1].float())
        return (c * d).sum(-1) / self.scale


class MixerBlock(nn.Module):
    def __init__(self, width, token_hidden, channel_hidden):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.token_mlp = nn.Sequential(nn.Linear(4, token_hidden), nn.GELU(), nn.Linear(token_hidden, 4))
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mlp = nn.Sequential(nn.Linear(width, channel_hidden), nn.GELU(),
                                         nn.Linear(channel_hidden, width))

    def forward(self, x):
        # Mix the four fixed roles of one candidate, never different candidates.
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class MixerHead(nn.Module):
    """Valen's released head: pooled (context, instruction, candidate, decision) roles."""

    def __init__(self, hidden_size, width=256, layers=2, token_hidden=16, channel_hidden=1024):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, width)
        self.roles = nn.Parameter(torch.zeros(1, 4, width))
        self.blocks = nn.Sequential(*(MixerBlock(width, token_hidden, channel_hidden) for _ in range(layers)))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def extract_features(self, hidden, branch):
        def pool(span):
            start, end = span
            if not 0 <= start < end <= hidden.shape[1]:
                raise ValueError(f"Invalid role span: {span}")
            return hidden[0, start:end].float().mean(0)

        context, instruction = pool(branch.context_span), pool(branch.instruction_span)
        decision = hidden[0, branch.decision_position].float()
        return torch.stack([torch.stack((context, instruction, pool(span), decision))
                            for span in branch.candidate_spans])

    def score_features(self, features):
        x = self.projection(self.input_norm(features.float())) + self.roles
        return self.output(self.blocks(x)[:, 2]).squeeze(-1)


def build_head(hidden_size, config):
    kind = config.get("head_type", "mixer")
    if kind == "bilinear":
        return BilinearHead(hidden_size, config.get("projection_dim", 256))
    if kind == "mixer":
        return MixerHead(hidden_size, config.get("head_width", 256), config.get("head_layers", 2),
                         config.get("head_token_hidden_dim", 16), config.get("head_channel_hidden_dim", 1024))
    raise ValueError(f"Unknown head_type: {kind!r}")


# ------------------------------------------------------------------------ model

class ValenModel(nn.Module):
    architecture = "qwen"

    def __init__(self, backbone, head, autocast_bf16=True):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.autocast_bf16 = autocast_bf16

    @property
    def device(self):
        return next(self.head.parameters()).device

    def forward(self, question):
        device = self.device
        logits = []
        for branch in question.branches:
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in branch.inputs.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=self.autocast_bf16 and device.type == "cuda"):
                hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            # The head runs outside autocast and stays FP32.
            logits.append(self.head.score_features(self.head.extract_features(hidden, branch)))
        return torch.cat(logits)


# -------------------------------------------- dual-encoder model (copied from valen/modeling/dual_encoder)

class _Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def _split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_width).transpose(1, 2)

    def project_memory(self, memory):
        return self._split(self.k(memory)), self._split(self.v(memory))

    def forward(self, query, memory=None, mask=None, kv=None):
        k, v = self.project_memory(memory) if kv is None else kv
        allowed = mask[:, None, None, :].bool() if mask is not None else None
        x = F.scaled_dot_product_attention(self._split(self.q(query)), k, v, attn_mask=allowed, dropout_p=0.0)
        return self.out(x.transpose(1, 2).reshape(query.shape))


class _FFN(nn.Sequential):
    def __init__(self, width, intermediate):
        super().__init__(nn.Linear(width, intermediate), nn.GELU(), nn.Linear(intermediate, width))


class _FusionBlock(nn.Module):
    def __init__(self, width, heads, intermediate):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.attention = _Attention(width, heads)
        self.ffn = _FFN(width, intermediate)

    def forward(self, x, mask):
        norm = self.norms[0](x)
        x = x + self.attention(norm, norm, mask)
        return x + self.ffn(self.norms[1](x))


class _QuestionBlock(nn.Module):
    def __init__(self, width, heads, intermediate):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.self_attention = _Attention(width, heads)
        self.instructions = _Attention(width, heads)
        self.memory = _Attention(width, heads)
        self.ffn = _FFN(width, intermediate)

    def forward(self, r, h, hmask, memory, mmask):
        n = self.norms[0](r)
        r = r + self.self_attention(n, n)
        r = r + self.instructions(self.norms[1](r), h, hmask)
        shape = r.shape
        r = r + self.memory(self.norms[2](r).reshape(1, -1, shape[-1]), memory, mmask).reshape(shape)
        return r + self.ffn(self.norms[3](r))


class _CandidateBlock(nn.Module):
    def __init__(self, width, heads, intermediate):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.question = _Attention(width, heads)
        self.memory = _Attention(width, heads)
        self.ffn = _FFN(width, intermediate)

    def forward(self, u, r, owners, memory, mmask, chunk_size):
        kv = self.memory.project_memory(memory)
        chunks = []
        for start in range(0, len(u), chunk_size):
            x = u[start:start + chunk_size].unsqueeze(1)
            evidence = r[owners[start:start + chunk_size]]
            x = x + self.question(self.norms[0](x), evidence)
            x = x + self.memory(self.norms[1](x).transpose(0, 1), mask=mmask, kv=kv).transpose(0, 1)
            chunks.append((x + self.ffn(self.norms[2](x))).squeeze(1))
        return torch.cat(chunks)


def _positions(length, width, device):
    pos = torch.arange(length, device=device).float()[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000.0) / width))
    features = pos * frequencies[None, :]
    return torch.stack((features.sin(), features.cos()), -1).flatten(-2)[:, :width]


def _load_encoder(path, config, unused_prefixes=(), device="cpu", dtype=torch.float32):
    from transformers import AutoModel
    local = Path(path).exists()
    device = torch.device(device)
    kwargs = dict(config=config, local_files_only=local, dtype=dtype,
                  attn_implementation="sdpa", output_loading_info=True, low_cpu_mem_usage=True)
    if device.type == "cuda":
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    encoder, info = AutoModel.from_pretrained(path, **kwargs)
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith(unused_prefixes)]
    if unexpected or any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete encoder load: {info}")
    return encoder


def _unfreeze_last(layers, norm, count, option):
    if type(count) is not int or not 1 <= count <= len(layers):
        raise ValueError(f"{option} out of range")
    for layer in layers[-count:]:
        layer.requires_grad_(True)
    norm.requires_grad_(True)


class ValenDualEncoder(nn.Module):
    """valen/modeling/dual_encoder/model.py — shared state memory, question/candidate readers."""

    architecture = "dual_encoder"
    model_name = "Valen-DE-ModernBERTbase-DINOv3B16-en-v1"

    def __init__(self, text_encoder, vision_encoder, config):
        super().__init__()
        self.config = dict(config)
        self.text_encoder, self.vision_encoder = text_encoder, vision_encoder
        d, heads, inner = config.get("hidden_size", 512), config.get("num_heads", 8), config.get("intermediate_size", 2048)
        self.width = d
        self.pad_token_id = text_encoder.config.pad_token_id
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, d)
        self.vision_projection = nn.Linear(vision_encoder.config.hidden_size, d)
        self.modality = nn.Embedding(2, d)
        self.coordinates = nn.Linear(2, d)
        self.visual_cls = nn.Parameter(torch.zeros(1, 1, d))
        self.fusion = nn.ModuleList([_FusionBlock(d, heads, inner) for _ in range(config.get("fusion_layers", 2))])
        self.question_queries = nn.Parameter(torch.randn(1, config.get("question_queries", 8), d) / math.sqrt(d))
        self.question_pool = _Attention(d, heads)
        self.question_reader = nn.ModuleList([_QuestionBlock(d, heads, inner) for _ in range(config.get("question_layers", 3))])
        self.candidate_query = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
        self.candidate_pool = _Attention(d, heads)
        self.task_type = nn.Embedding(3, d)
        self.candidate_init = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.candidate_reader = nn.ModuleList([_CandidateBlock(d, heads, inner) for _ in range(config.get("candidate_layers", 2))])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).float()
        self.text_batch_size = config.get("text_batch_size", 32)
        self.candidate_chunk_size = config.get("candidate_chunk_size", 256)

    @property
    def device(self):
        return next(self.head.parameters()).device

    def _amp(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda" and self.config.get("dtype", "bf16") == "bf16" else nullcontext())

    def _text_batch(self, sequences):
        # Pad to the longest sequence in this mini-batch only, not to state_max_length.
        device = self.device
        length = max(map(len, sequences))
        ids = torch.full((len(sequences), length), self.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = True
        context = nullcontext() if any(p.requires_grad for p in self.text_encoder.parameters()) else torch.no_grad()
        with context:
            hidden = self.text_encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        return self.text_projection(hidden), mask

    def _text_chunks(self, sequences, pool=None):
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        results = [None] * len(sequences)
        for start in range(0, len(order), self.text_batch_size):
            indices = order[start:start + self.text_batch_size]
            hidden, mask = self._text_batch([sequences[i] for i in indices])
            if pool is not None:
                hidden = pool(self.candidate_query.expand(len(indices), -1, -1), hidden, mask).squeeze(1)
            for row, i in enumerate(indices):
                results[i] = hidden[row] if pool is not None else hidden[row, :len(sequences[i])]
        if pool is not None:
            return torch.stack(results)
        max_length = max(map(len, sequences))
        hidden = torch.stack([F.pad(h, (0, 0, 0, max_length - h.shape[0])) for h in results])
        mask = torch.arange(max_length, device=hidden.device)[None, :] < torch.tensor(
            list(map(len, sequences)), device=hidden.device)[:, None]
        return hidden, mask

    def encode_state(self, state):
        with self._amp():
            text, mask = self._text_batch([state.state_tokens])
            text = text + self.modality.weight[0] + _positions(text.shape[1], self.width, text.device)
            if state.pixel_values is not None:
                pixels = state.pixel_values.to(device=text.device, dtype=next(self.vision_encoder.parameters()).dtype)
                context = nullcontext() if any(p.requires_grad for p in self.vision_encoder.parameters()) else torch.no_grad()
                with context:
                    hidden = self.vision_encoder(pixel_values=pixels, return_dict=True).last_hidden_state
                registers = getattr(self.vision_encoder.config, "num_register_tokens", 0) or 0
                hidden = torch.cat((hidden[:, :1], hidden[:, 1 + registers:]), dim=1)
                vision = self.vision_projection(hidden)
                patch = self.vision_encoder.config.patch_size
                rows, cols = pixels.shape[-2] // patch, pixels.shape[-1] // patch
                if vision.shape[1] != rows * cols + 1:
                    raise ValueError("Unexpected vision token layout")
                y, x = torch.meshgrid(torch.linspace(-1, 1, rows, device=text.device),
                                      torch.linspace(-1, 1, cols, device=text.device), indexing="ij")
                xy = self.coordinates(torch.stack((x, y), -1).reshape(1, -1, 2))
                vision = vision + torch.cat((self.visual_cls, xy), 1) + self.modality.weight[1]
                text = torch.cat((text, vision), dim=1)
                mask = torch.cat((mask, state.visual_mask.to(text.device)), dim=1)
            for layer in self.fusion:
                text = layer(text, mask)
            return text, mask

    def encode_questions(self, state, memory):
        with self._amp():
            m, mmask = memory
            h, hmask = self._text_chunks(state.instruction_tokens)
            r = self.question_pool(self.question_queries.expand(len(state.questions), -1, -1), h, hmask)
            for layer in self.question_reader:
                r = layer(r, h, hmask, m, mmask)
            return r

    def score_questions(self, state, memory, evidence):
        with self._amp():
            m, mmask = memory
            candidates = self._text_chunks(state.candidate_tokens, self.candidate_pool)
            counts = torch.tensor([b - a for a, b in zip(state.candidate_offsets, state.candidate_offsets[1:])],
                                  device=m.device)
            owners = torch.arange(len(counts), device=m.device).repeat_interleave(counts)
            kinds = torch.tensor([{"choice": 0, "noul": 1, "score": 2}[q.kind] for q in state.questions], device=m.device)
            u = self.candidate_init(torch.cat((candidates, evidence.mean(1)[owners], self.task_type(kinds)[owners]), -1))
            for layer in self.candidate_reader:
                u = layer(u, evidence, owners, m, mmask, self.candidate_chunk_size)
        with torch.autocast(device_type=u.device.type, enabled=False):
            logits = self.head(u.float()).squeeze(-1)
        return list(logits.split(counts.tolist()))

    def forward(self, state):
        if not state.questions:
            return []
        memory = self.encode_state(state)
        return self.score_questions(state, memory, self.encode_questions(state, memory))

    forward_state = forward


class DualProcessor:
    """ModernBERT tokenizer + DINOv3 image processor, saved beside model.safetensors."""

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(directory)
        self.image_processor.save_pretrained(directory)


def is_dual(model):
    return getattr(model, "architecture", "qwen") == "dual_encoder"


def question_logits(model, state):
    """Pairs (question, logits[K]). Dual encoder: one shared forward. Qwen: one forward per question."""
    if is_dual(model):
        return list(zip(state.questions, model(state)))
    return [(question, model(question)) for question in state.questions]


def build_dual_encoder(args, device):
    from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
    device = torch.device(device)
    text_path, vision_path = args.text_model, args.vision_model
    text_cfg = AutoConfig.from_pretrained(text_path, local_files_only=Path(text_path).exists())
    if text_cfg.model_type != "modernbert":
        raise ValueError("Dual encoder text tower must be ModernBERT")
    for name in ("attention_dropout", "embedding_dropout", "mlp_dropout", "classifier_dropout"):
        setattr(text_cfg, name, 0.0)
    vision_cfg = AutoConfig.from_pretrained(vision_path, local_files_only=Path(vision_path).exists())
    if vision_cfg.model_type != "dinov3_vit":
        raise ValueError("Dual encoder vision tower must be DINOv3 ViT")
    for name in ("pos_embed_shift", "pos_embed_jitter", "pos_embed_rescale"):
        setattr(vision_cfg, name, None)
    vision_cfg.attention_dropout = vision_cfg.drop_path_rate = 0.0
    load_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    text = _load_encoder(text_path, text_cfg, ("head.", "decoder."), device, load_dtype)
    vision = _load_encoder(vision_path, vision_cfg, (), device, load_dtype)
    de_config = {
        "dtype": "bf16",
        "hidden_size": args.de_hidden, "num_heads": args.de_heads, "intermediate_size": args.de_intermediate,
        "fusion_layers": 2, "question_layers": 3, "candidate_layers": 2, "question_queries": 8,
        "text_batch_size": 32, "candidate_chunk_size": 256,
    }
    model = ValenDualEncoder(text, vision, de_config)
    apply_dual_stage(model, args.stage, args.text_unfreeze_layers, args.vision_unfreeze_layers,
                     args.no_gradient_checkpointing)
    tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=Path(text_path).exists())
    image_processor = AutoImageProcessor.from_pretrained(vision_path, local_files_only=Path(vision_path).exists())
    model = model.to(device)
    if args.precision == "bf16":
        model.to(dtype=torch.bfloat16)
    model.head.float()
    disable_dropout(model)
    return model, DualProcessor(tokenizer, image_processor)


def build_compiler(config, processor, media_root):
    args = config.get("args") or {}
    if config.get("architecture", args.get("architecture", "qwen")) == "dual_encoder":
        from transformers import AutoConfig
        vision = config.get("vision_model") or args.get("vision_model") or DEFAULT_VISION_ENCODER
        local = Path(str(vision)).exists()
        patch = AutoConfig.from_pretrained(vision, local_files_only=local).patch_size
        return ParallelCompiler(
            processor.tokenizer, processor.image_processor, media_root,
            image_size=int(args.get("image_size", 384)), patch_size=patch,
            state_max_length=int(args.get("state_max_length", 1024)),
            question_max_length=int(args.get("question_max_length", 128)),
            candidate_max_length=int(args.get("candidate_max_length", 128)),
            max_questions=int(args.get("max_questions", 256)))
    return Compiler(processor, media_root, config.get("max_length", 8192), config.get("media_kwargs"))


def apply_dual_stage(model, stage, text_layers, vision_layers, no_checkpointing):
    model.text_encoder.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    if stage not in {"warmup", "text", "joint", "vision_top"}:
        raise ValueError(f"Unknown dual-encoder stage: {stage}")
    if stage != "warmup":
        _unfreeze_last(model.text_encoder.layers, model.text_encoder.final_norm, text_layers, "text_unfreeze_layers")
    if stage == "vision_top":
        _unfreeze_last(model.vision_encoder.model.layer, model.vision_encoder.norm, vision_layers,
                       "vision_unfreeze_layers")
    if not no_checkpointing:
        for encoder in (model.text_encoder, model.vision_encoder):
            if any(p.requires_grad for p in encoder.parameters()):
                encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def load_backbone(model_path, attn_implementation="eager", dtype=torch.float32, device="cpu"):
    from transformers import Qwen3_5ForConditionalGeneration
    # Load the published container and keep only its backbone. Loading Qwen3_5Model
    # directly can silently re-initialize the language tower in Transformers 5.4.
    device = torch.device(device)
    kwargs = dict(dtype=dtype, attn_implementation=attn_implementation,
                  output_loading_info=True, low_cpu_mem_usage=True)
    if device.type == "cuda":
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    container, info = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **kwargs)
    problems = {key: [k for k in info.get(key, []) if not k.startswith("lm_head")]
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    if any(problems.values()) or info.get("error_msgs"):
        raise ValueError(f"Incomplete pretrained model load: {problems} {info.get('error_msgs')}")
    backbone = container.model
    del container  # The generation head is never used.
    return backbone


def disable_dropout(module):
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if hasattr(m, "attention_dropout") and isinstance(m.attention_dropout, float):
            m.attention_dropout = 0.0


def save_weights(model, path, backbone_dtype=torch.float32):
    from safetensors.torch import save_file
    tensors = {}
    encoder_prefix = ("backbone.", "text_encoder.", "vision_encoder.")
    for name, value in model.state_dict().items():
        value = value.detach()
        if value.is_floating_point():
            if name.startswith("head."):
                value = value.float()
            elif name.startswith(encoder_prefix) or (is_dual(model) and name != "head"):
                value = value.to(backbone_dtype)
        tensors[name] = value.cpu().contiguous()
    tmp = Path(str(path) + ".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)


def load_weights(model, path):
    from safetensors.torch import load_file
    state = load_file(str(path))
    own = model.state_dict()
    converted = {k: v.to(own[k].dtype) if k in own else v for k, v in state.items()}
    model.load_state_dict(converted, strict=True)


def build_model(base_model, head_config, device, attn="eager", gradient_checkpointing=True,
                backbone_dtype=torch.float32):
    backbone = load_backbone(base_model, attn, backbone_dtype, device)
    head = build_head(backbone.config.text_config.hidden_size, head_config)
    model = ValenModel(backbone, head.float())
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    disable_dropout(model)
    return model.to(device)


def load_for_inference(checkpoint, device="cuda", base_model=None, attn="eager"):
    """Rebuild a trained checkpoint directory for prediction (BF16 backbone on CUDA)."""
    from transformers import AutoProcessor
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / CONFIG_FILE).read_text(encoding="utf-8"))
    target = torch.device(device)
    saved = config.get("args") or {}
    architecture = config.get("architecture", saved.get("architecture", "qwen"))
    if architecture == "dual_encoder":
        args = argparse.Namespace(
            text_model=base_model or config.get("text_model") or saved.get("text_model") or DEFAULT_TEXT_ENCODER,
            vision_model=config.get("vision_model") or saved.get("vision_model") or DEFAULT_VISION_ENCODER,
            precision="bf16" if target.type == "cuda" else "mixed",
            de_hidden=int(saved.get("de_hidden", 512)),
            de_heads=int(saved.get("de_heads", 8)),
            de_intermediate=int(saved.get("de_intermediate", 2048)),
            stage="warmup",
            text_unfreeze_layers=int(saved.get("text_unfreeze_layers", 6)),
            vision_unfreeze_layers=int(saved.get("vision_unfreeze_layers", 2)),
            no_gradient_checkpointing=True,
        )
        model, processor = build_dual_encoder(args, target)
        if (checkpoint / "tokenizer.json").exists() or (checkpoint / "tokenizer_config.json").exists():
            from transformers import AutoImageProcessor, AutoTokenizer
            processor = DualProcessor(AutoTokenizer.from_pretrained(checkpoint),
                                      AutoImageProcessor.from_pretrained(checkpoint))
    else:
        processor = AutoProcessor.from_pretrained(checkpoint)
        dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
        model = build_model(base_model or config["base_model"], config["head"], target, attn,
                            gradient_checkpointing=False, backbone_dtype=dtype)
    load_weights(model, checkpoint / WEIGHTS_FILE)
    model.head.float()
    return model.eval(), processor, config


def answer(question, logits, temperature=1.0):
    """Valen response format (valen responses.answer)."""
    probs = (logits.float() / temperature).softmax(-1).detach().cpu().tolist()
    result = {"type": question.kind}
    if question.kind == "noul":
        result["noul"] = probs[question.keys.index("true")]
        return result
    result["probabilities"] = dict(zip(question.keys, probs))
    mode = max(range(len(probs)), key=probs.__getitem__)
    if question.kind == "choice":
        result["choice"] = question.keys[mode]
        k = len(probs)
        result["confidence"] = 1.0 if k == 1 else max(0.0, (max(probs) - 1 / k) / (1 - 1 / k))
    else:
        result["score"] = sum(i * p for i, p in enumerate(probs))
        result["legend"] = dict(zip(question.keys, question.descriptions))
        distance = sum(p * abs(i - mode) for i, p in enumerate(probs))
        uniform = sum(abs(i - (len(probs) - 1) / 2) for i in range(len(probs))) / len(probs)
        result["confidence"] = max(0.0, 1.0 - distance / uniform)
    return result


# -------------------------------------------------------------------- objectives

def sft_loss(logits, target, brier_weight=0.0):
    log_probs = logits.float().log_softmax(-1)
    target = torch.as_tensor(target, device=log_probs.device, dtype=log_probs.dtype)
    ce = -(target * log_probs).sum()
    brier = (log_probs.exp() - target).square().sum()
    return ce + brier_weight * brier, {"ce": ce.detach(), "brier": brier.detach()}


def decision_reward(probabilities, target, actions, correctness_weight, confidence_weight):
    confidence = probabilities.detach()[actions]
    correctness = target.detach()[actions]
    error = correctness * (1 - confidence).square() + (1 - correctness) * confidence.square()
    return correctness_weight * correctness - confidence_weight * error, correctness, confidence, error


def group_advantages(rewards, epsilon):
    rewards = rewards.detach()
    if torch.all(rewards == rewards[0]):
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)


@dataclass
class Rollout:
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    correctness: torch.Tensor
    confidence: torch.Tensor
    reference_log_probs: torch.Tensor | None


@torch.no_grad()
def rlcd_rollouts(model, reference, states, args):
    """Sample candidate actions from the current (old) policy; frozen for all updates."""
    result = []
    for state in states:
        group = []
        pairs = question_logits(model, state)
        refs = question_logits(reference, state) if reference is not None else [(None, None)] * len(pairs)
        for (question, logits), (_, ref_logits) in zip(pairs, refs):
            logits = logits.float()
            if not torch.isfinite(logits).all():
                raise ValueError(f"Non-finite logits for {state.group_id}/{question.qid}")
            log_probs = logits.log_softmax(-1)
            target = torch.as_tensor(question.target, device=logits.device, dtype=logits.dtype)
            actions = torch.multinomial(log_probs.exp(), args.group_size, replacement=True)
            reward, correctness, confidence, _ = decision_reward(
                log_probs.exp(), target, actions, args.correctness_weight, args.confidence_weight)
            ref = None if ref_logits is None else ref_logits.float().log_softmax(-1).to(logits.device)
            group.append(Rollout(actions, log_probs[actions], group_advantages(reward, args.advantage_epsilon),
                                 reward, correctness, confidence, ref))
        result.append(group)
    return result


def rlcd_loss(logits, question, rollout, args):
    log_probs = logits.float().log_softmax(-1)
    ratio = (log_probs[rollout.actions] - rollout.old_log_probs).exp()
    unclipped = ratio * rollout.advantages
    clipped = ratio.clamp(1 - args.clip_epsilon, 1 + args.clip_epsilon) * rollout.advantages
    policy = -torch.minimum(unclipped, clipped).mean()
    probs = log_probs.exp()
    target = torch.as_tensor(question.target, device=probs.device, dtype=probs.dtype)
    brier = (probs - target).square().sum()
    kl = ((probs * (log_probs - rollout.reference_log_probs)).sum()
          if rollout.reference_log_probs is not None else probs.new_zeros(()))
    loss = policy + args.beta * kl + args.brier_weight * brier
    return loss, {"policy_loss": policy.detach(), "kl": kl.detach(), "brier": brier.detach(),
                  "reward": rollout.rewards.mean(), "sample_correctness": rollout.correctness.mean(),
                  "sample_confidence": rollout.confidence.mean(),
                  "clip_fraction": (clipped < unclipped).float().mean().detach()}


def hard_correct(logits, target):
    if target is None or max(target) < 1.0:
        return None  # Soft labels do not count toward accuracy.
    return float(int(logits.argmax()) == target.index(max(target)))


@torch.no_grad()
def evaluate(model, compiler, records, limit=None):
    totals = {"questions": 0, "hard": 0, "correct": 0.0, "brier": 0.0, "ce": 0.0}
    for record in records[:limit] if limit else records:
        try:
            state = compiler.compile(record, labeled_only=True)
        except (ValueError, FileNotFoundError) as exc:
            print(f"[eval] skip {record.get('group_id')}: {exc}", flush=True)
            continue
        for question, logits in question_logits(model, state):
            logits = logits.float()
            _, terms = sft_loss(logits, question.target)
            totals["questions"] += 1
            totals["ce"] += float(terms["ce"])
            totals["brier"] += float(terms["brier"])
            correct = hard_correct(logits, question.target)
            if correct is not None:
                totals["hard"] += 1
                totals["correct"] += correct
    n = max(totals["questions"], 1)
    return {"questions": totals["questions"], "accuracy": totals["correct"] / max(totals["hard"], 1),
            "ce": totals["ce"] / n, "brier": totals["brier"] / n}


# --------------------------------------------------------------------- training

def configure_trainable(model, args):
    if is_dual(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.1f}%)  "
              f"stage={args.stage}", flush=True)
        return
    model.requires_grad_(True)
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    visual = getattr(model.backbone, "visual", None)
    if visual is not None:
        if args.freeze_vision:
            visual.requires_grad_(False)
        if args.freeze_merger:
            visual.merger.requires_grad_(False)
        elif args.train_merger:
            # Valen "joint": frozen vision encoder, trainable vision-language adaptor.
            visual.merger.requires_grad_(True)
    if args.freeze_embeddings:
        model.backbone.get_input_embeddings().requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)", flush=True)
    if not any(p.requires_grad for p in model.backbone.parameters()):
        # No backbone gradients: checkpointing would only add recomputation.
        model.backbone.gradient_checkpointing_disable()


def stochastic_round_(target, source):
    """Write FP32 `source` into BF16 `target`, rounding up with probability equal to the
    discarded fraction. Round-to-nearest would drop the ~1e-5 relative updates of full
    fine-tuning, which are smaller than BF16's ~4e-3 relative spacing."""
    bits = source.view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    target.copy_(((bits + noise) & -65536).view(torch.float32))  # Exact: low 16 bits are zero.


class BF16AdamW(torch.optim.Optimizer):
    """AdamW whose moments share each parameter's dtype. BF16 weights are updated in FP32
    and written back with stochastic rounding; FP32 parameters (the head) update normally."""

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, chunk_size=1 << 24):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.chunk_size = chunk_size  # Bounds FP32 temporaries for the 248k-token embedding table.

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, decay = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.contiguous_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.contiguous_format)
                state["step"] += 1
                bias1, bias2 = 1 - beta1 ** state["step"], 1 - beta2 ** state["step"]
                flat = [t.view(-1) for t in (p.data, p.grad.contiguous(), state["exp_avg"], state["exp_avg_sq"])]
                for start in range(0, flat[0].numel(), self.chunk_size):
                    w_, g_, m_, v_ = (t[start:start + self.chunk_size] for t in flat)
                    # For FP32 tensors .float() aliases the storage, so only the moment and
                    # weight tensors themselves are updated in place.
                    g = g_.float()
                    m = m_.float().mul_(beta1).add_(g, alpha=1 - beta1)
                    v = v_.float().mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    m_.copy_(m)
                    v_.copy_(v)
                    w = w_.float()
                    if decay:
                        w.mul_(1 - lr * decay)
                    w.addcdiv_(m, (v / bias2).sqrt_().add_(eps), value=-lr / bias1)
                    if w_.dtype == torch.bfloat16:
                        stochastic_round_(w_, w)
                    else:
                        w_.copy_(w)
        return loss


def make_optimizer(model, args):
    rates = {"head": args.lr_head, "merger": args.lr_merger, "vision": args.lr_vision, "language": args.lr_backbone}
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("head.", "fusion", "question_", "candidate_", "task_type", "modality",
                            "coordinates", "visual_cls", "text_projection", "vision_projection")):
            part = "head"
        elif name.startswith("text_encoder."):
            part = "language"
        elif name.startswith("vision_encoder.") or (".visual." in name and ".visual.merger." not in name):
            part = "vision"
        elif ".visual.merger." in name:
            part = "merger"
        elif name.startswith("head."):
            part = "head"
        else:
            part = "language"
        decay = parameter.ndim >= 2 and "embed" not in name and "norm" not in name
        groups.setdefault((part, decay), []).append(parameter)
    params = [{"params": ps, "lr": rates[part], "base_lr": rates[part],
               "weight_decay": args.weight_decay if decay else 0.0, "name": f"{part}{'' if decay else '_nodecay'}"}
              for (part, decay), ps in groups.items()]
    if args.optim == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            sys.exit("bitsandbytes is not installed: `pip install bitsandbytes` or pass --optim adafactor")
        return bnb.optim.AdamW8bit(params, betas=(0.9, 0.95), eps=1e-8)
    if args.optim == "adamw_bf16":
        return BF16AdamW(params, betas=(0.9, 0.95), eps=1e-8)
    if args.optim == "adafactor":
        return torch.optim.Adafactor(params)
    return torch.optim.AdamW(params, betas=(0.9, 0.95), eps=1e-8, fused=model.device.type == "cuda")


def set_learning_rate(optimizer, step, args):
    if step < args.warmup_steps:
        factor = (step + 1) / args.warmup_steps
    elif args.schedule == "cosine":
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        factor = args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    else:
        factor = 1.0
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor
    return factor


def make_reference(model, device, path):
    """Frozen copy of the starting policy; BF16 on CUDA to save memory."""
    reference = copy.deepcopy(model)
    reference.requires_grad_(False)
    if is_dual(reference):
        for encoder in (reference.text_encoder, reference.vision_encoder):
            if hasattr(encoder, "gradient_checkpointing_disable"):
                encoder.gradient_checkpointing_disable()
    else:
        reference.backbone.gradient_checkpointing_disable()
    if path.exists():
        load_weights(reference, path)
    else:
        save_weights(model, path, backbone_dtype=torch.bfloat16)
    if torch.device(device).type == "cuda" and not is_dual(reference):
        reference.backbone.to(torch.bfloat16)
    reference.head.float()
    return reference.to(device).eval()


def save_checkpoint(directory, model, optimizer, processor, args, progress):
    directory.mkdir(parents=True, exist_ok=True)
    bf16 = args.save_bf16 or args.precision == "bf16"  # BF16 weights are saved exactly.
    save_weights(model, directory / WEIGHTS_FILE, torch.bfloat16 if bf16 else torch.float32)
    config = {"model": getattr(model, "model_name", MODEL_NAME),
              "architecture": "dual_encoder" if is_dual(model) else "qwen",
              "base_model": args.base_model, "text_model": args.text_model, "vision_model": args.vision_model,
              "head": args.head, "max_length": args.max_length, "media_kwargs": args.media_kwargs,
              "mode": args.mode, "progress": progress, "args": vars(args)}
    (directory / CONFIG_FILE).write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    processor.save_pretrained(directory)
    if optimizer is not None:
        state = {"optimizer": optimizer.state_dict(), "progress": progress,
                 "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(state, directory / (STATE_FILE + ".tmp"))
        os.replace(directory / (STATE_FILE + ".tmp"), directory / STATE_FILE)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("sft", "rlcd"), required=True)
    p.add_argument("--architecture", choices=("qwen", "dual_encoder"), default="dual_encoder",
                   help="dual_encoder: ModernBERT + DINOv3 (Valen-DE). qwen: Qwen3.5 + mixer/bilinear head")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B", help="HF id or local Qwen3.5 directory")
    p.add_argument("--text-model", default=DEFAULT_TEXT_ENCODER, help="ModernBERT id or local directory")
    p.add_argument("--vision-model", default=DEFAULT_VISION_ENCODER, help="DINOv3 ViT id or local directory")
    p.add_argument("--stage", choices=("warmup", "text", "joint", "vision_top"), default="warmup",
                   help="Dual-encoder freeze schedule: warmup freezes both towers")
    p.add_argument("--text-unfreeze-layers", type=int, default=6)
    p.add_argument("--vision-unfreeze-layers", type=int, default=2)
    p.add_argument("--de-hidden", type=int, default=512)
    p.add_argument("--de-heads", type=int, default=8)
    p.add_argument("--de-intermediate", type=int, default=2048)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--state-max-length", type=int, default=1024)
    p.add_argument("--question-max-length", type=int, default=128)
    p.add_argument("--candidate-max-length", type=int, default=128)
    p.add_argument("--max-questions", type=int, default=256)
    p.add_argument("--data", default=VALEN_TRAIN_HUB,
                   help="Valen JSONL, directory, Hub id, or alias (valen/eval/smoke). "
                        f"Default: {VALEN_TRAIN_HUB}")
    p.add_argument("--eval-data", default=VALEN_EVAL_HUB,
                   help=f"Held-out JSONL/Hub id/alias. Default: {VALEN_EVAL_HUB}. Pass empty to skip.")
    p.add_argument("--data-slice", choices=("auto", "all", "sft", "rlcd"), default="auto",
                   help="On the official 100k file: auto follows --mode (sft=first 70k, rlcd=last 30k)")
    p.add_argument("--output", required=True)
    p.add_argument("--init", help="Start a new run from a checkpoint directory (e.g. SFT -> RLCD)")
    p.add_argument("--resume", help="Continue a run (weights, optimizer, data cursor) from its checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    # Schedule
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1000, help="Rollout/optimizer batches")
    p.add_argument("--states-per-step", type=int, default=8, help="Records per batch (gradient accumulation)")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--schedule", choices=("constant", "cosine"), default="cosine")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    # Optimizer
    p.add_argument("--precision", choices=("mixed", "bf16"), default="bf16",
                   help="bf16: load and train the backbone in BF16 on CUDA (default). "
                        "mixed: FP32 weights + BF16 autocast (Valen's own setup). Head stays FP32")
    p.add_argument("--optim", choices=("adamw8bit", "adamw_bf16", "adafactor", "adamw"),
                   help="Default: adamw8bit (mixed), adamw_bf16 (bf16)")
    p.add_argument("--lr-backbone", type=float, help="Default: 1e-5 (sft), 1e-6 (rlcd)")
    p.add_argument("--lr-vision", type=float, help="Vision encoder. Default: 2e-6 (sft), 2e-7 (rlcd)")
    p.add_argument("--lr-merger", type=float, help="Vision adaptor (visual.merger). Default: 1e-5 (sft), 1e-6 (rlcd)")
    p.add_argument("--lr-head", type=float, help="Default: 2e-4 (sft), 1e-5 (rlcd)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # What trains
    p.add_argument("--freeze-backbone", action="store_true", help="Head-only warmup")
    p.add_argument("--freeze-vision", action="store_true",
                   help="Freeze the whole vision tower, including the adaptor unless --train-merger")
    p.add_argument("--freeze-merger", action="store_true", help="Freeze only the vision adaptor (visual.merger)")
    p.add_argument("--train-merger", action="store_true",
                   help="With --freeze-vision or --freeze-backbone, keep the vision adaptor trainable")
    p.add_argument("--freeze-embeddings", action="store_true", help="Keep token embeddings frozen")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--attn", default="eager", help="attn_implementation passed to Transformers")
    # Model and inputs
    p.add_argument("--head-type", choices=("mixer", "bilinear"), default="mixer")
    p.add_argument("--max-length", type=int, default=8192, help="Longer prompts are skipped, never truncated")
    p.add_argument("--media-kwargs", type=json.loads, default=DEFAULT_MEDIA_KWARGS,
                   help="JSON passed to the Qwen processor (image/video resolution and frames)")
    # SFT
    p.add_argument("--sft-brier-weight", type=float, default=0.0)
    # RLCD (defaults from configs/train/qwen/rlcd_*.json)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--num-iterations", type=int, default=2, help="Policy updates per rollout batch")
    p.add_argument("--correctness-weight", type=float, default=1.0)
    p.add_argument("--confidence-weight", type=float, default=1.0)
    p.add_argument("--brier-weight", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.02, help="KL weight to the frozen reference policy")
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--advantage-epsilon", type=float, default=1e-6)
    p.add_argument("--reference-device", help="Default: --device. Use cpu if the GPU is short on memory")
    # Saving and evaluation
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--save-bf16", action="store_true", help="Smaller checkpoints; resume loses FP32 precision")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-limit", type=int, help="Evaluate only the first N held-out records")
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args(argv)
    sft = args.mode == "sft"
    args.lr_backbone = args.lr_backbone if args.lr_backbone is not None else (1e-5 if sft else 1e-6)
    args.lr_vision = args.lr_vision if args.lr_vision is not None else (2e-6 if sft else 2e-7)
    args.lr_merger = args.lr_merger if args.lr_merger is not None else (1e-5 if sft else 1e-6)
    args.lr_head = args.lr_head if args.lr_head is not None else (2e-4 if sft else 1e-5)
    if args.freeze_merger and args.train_merger:
        p.error("--freeze-merger and --train-merger are contradictory")
    if args.optim is None:
        args.optim = "adamw_bf16" if args.precision == "bf16" else "adamw8bit"
    if args.precision == "bf16" and args.optim != "adamw_bf16":
        p.error("--precision bf16 needs --optim adamw_bf16; other optimizers round BF16 updates to zero")
    if args.init and args.resume:
        p.error("--init starts a new run and --resume continues one; choose one")
    if args.group_size < 2 or args.num_iterations < 1 or not 0 < args.clip_epsilon < 1:
        p.error("RLCD requires group_size >= 2, num_iterations >= 1 and 0 < clip_epsilon < 1")
    if args.correctness_weight == args.confidence_weight == 0:
        p.error("At least one RLCD reward weight must be positive")
    if args.states_per_step < 1 or args.max_steps < 1:
        p.error("--states-per-step and --max-steps must be positive")
    args.head = dict(DEFAULT_HEAD) if args.head_type == "mixer" else {"head_type": "bilinear", "projection_dim": 256}
    if args.eval_data in ("", "none", "off"):
        args.eval_data = None
    return args


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoProcessor

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            sys.exit("CUDA is not available to this PyTorch build; see README (RTX 50xx needs CUDA 12.8 wheels)")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
        print(f"CUDA: {torch.cuda.get_device_name(device)}  capability={torch.cuda.get_device_capability(device)}  "
              f"bf16={torch.cuda.is_bf16_supported()}", flush=True)

    source = Path(args.resume or args.init) if (args.resume or args.init) else None
    if source is not None:
        saved = json.loads((source / CONFIG_FILE).read_text(encoding="utf-8"))
        saved_args = saved.get("args") or {}
        saved_arch = saved.get("architecture", saved_args.get("architecture", "qwen"))
        if saved_arch != args.architecture:
            sys.exit(f"Checkpoint architecture is {saved_arch!r}; pass --architecture {saved_arch}")
        if saved.get("base_model") and saved["base_model"] != args.base_model:
            print(f"Using checkpoint base model {saved['base_model']!r}", flush=True)
            args.base_model = saved["base_model"]
        for key in ("text_model", "vision_model", "de_hidden", "de_heads", "de_intermediate"):
            if saved.get(key) is not None:
                setattr(args, key, saved[key])
            elif saved_args.get(key) is not None:
                setattr(args, key, saved_args[key])
        if "head" in saved:
            args.head = saved["head"]
        if args.resume:
            for key in ("stage", "text_unfreeze_layers", "vision_unfreeze_layers"):
                if saved_args.get(key) is not None:
                    setattr(args, key, saved_args[key])
        saved_precision = saved_args.get("precision", "mixed")
        if args.resume and saved_precision != args.precision:
            sys.exit(f"Checkpoint was trained with --precision {saved_precision}; resume with the same "
                     "value, or start a new run with --init")

    if args.architecture == "dual_encoder":
        model, processor = build_dual_encoder(args, device)
        print(f"architecture=dual_encoder  text={args.text_model}  vision={args.vision_model}  "
              f"stage={args.stage}", flush=True)
    else:
        processor = AutoProcessor.from_pretrained(args.base_model)
        backbone_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
        model = build_model(args.base_model, args.head, device, args.attn, not args.no_gradient_checkpointing,
                            backbone_dtype)
        model.head.float()
        print(f"architecture=qwen  base={args.base_model}", flush=True)
    if source is not None:
        load_weights(model, source / WEIGHTS_FILE)
        print(f"Loaded weights from {source}", flush=True)
    elif args.mode == "rlcd":
        print("WARNING: RLCD from a random decision head. Run --mode sft first and pass --init.", flush=True)

    reference = None
    if args.mode == "rlcd" and args.beta > 0:
        # Frozen starting policy. Resumed runs reuse the saved original reference.
        reference = make_reference(model, args.reference_device or args.device, output / REFERENCE_FILE)
    configure_trainable(model, args)
    model.train()

    optimizer = make_optimizer(model, args)
    progress = {"step": 0, "optimizer_steps": 0, "epoch": 0, "cursor": 0, "best_accuracy": None}
    if args.resume:
        state = torch.load(source / STATE_FILE, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        progress = state["progress"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resumed at step {progress['step']} (epoch {progress['epoch']}, cursor {progress['cursor']})", flush=True)

    args.data = str(resolve_dataset(args.data))
    if args.eval_data:
        args.eval_data = str(resolve_dataset(args.eval_data))
    loaded = [r for r in read_jsonl(args.data) if is_labeled(r)]
    records, used_slice = slice_records(loaded, args.mode, args.data_slice, args.data)
    eval_records = [r for r in read_jsonl(args.eval_data) if is_labeled(r)] if args.eval_data else None
    compiler_config = {"architecture": args.architecture, "max_length": args.max_length,
                       "media_kwargs": args.media_kwargs, "args": vars(args),
                       "vision_model": args.vision_model}
    compiler = build_compiler(compiler_config, processor, Path(args.data).parent)
    eval_compiler = (build_compiler(compiler_config, processor, Path(args.eval_data).parent)
                     if args.eval_data else None)
    print(f"{len(records)} labeled training records (slice={used_slice} of {len(loaded)}); "
          f"data={args.data}; architecture={args.architecture}; mode={args.mode}; precision={args.precision}; "
          f"optimizer={args.optim}; device={device}", flush=True)

    def epoch_order(epoch):
        order = list(range(len(records)))
        random.Random(f"{args.seed}-{epoch}").shuffle(order)
        return order

    order = epoch_order(progress["epoch"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    metrics_file = (output / "metrics.jsonl").open("a", encoding="utf-8")
    skipped = 0

    def next_batch():
        nonlocal order, skipped
        batch = []
        while len(batch) < args.states_per_step:
            if progress["cursor"] >= len(order):
                progress["epoch"] += 1
                progress["cursor"] = 0
                if progress["epoch"] >= args.epochs:
                    break
                order = epoch_order(progress["epoch"])
            index = order[progress["cursor"]]
            progress["cursor"] += 1
            rng = random.Random(f"{args.seed}-{progress['epoch']}-{index}")
            try:
                state = compiler.compile(records[index], rng=rng, labeled_only=True)
            except (ValueError, FileNotFoundError, OSError) as exc:
                skipped += 1
                print(f"skip {records[index].get('group_id')}: {exc}", flush=True)
                continue
            if state.questions:
                batch.append(state)
        return batch

    def backward_question(loss, n_questions, n_states):
        # Valen weighting: mean over questions within a state, then over states.
        (loss / (n_questions * n_states)).backward()

    while progress["step"] < args.max_steps and progress["epoch"] < args.epochs:
        started = time.perf_counter()
        batch = next_batch()
        if not batch:
            break
        lr_factor = set_learning_rate(optimizer, progress["step"], args)
        sums, count, correct, hard = {}, 0, 0.0, 0

        def accumulate(terms):
            nonlocal count
            count += 1
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + float(value)

        if args.mode == "sft":
            optimizer.zero_grad(set_to_none=True)
            for state in batch:
                losses = []
                for question, logits in question_logits(model, state):
                    loss, terms = sft_loss(logits, question.target, args.sft_brier_weight)
                    accumulate(terms | {"loss": loss.detach()})
                    c = hard_correct(logits.detach(), question.target)
                    if c is not None:
                        hard, correct = hard + 1, correct + c
                    if is_dual(model):
                        losses.append(loss)
                    else:
                        backward_question(loss, len(state.questions), len(batch))
                if is_dual(model):
                    combined = losses[0] if len(losses) == 1 else torch.stack(losses).sum()
                    backward_question(combined, len(state.questions), len(batch))
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            progress["optimizer_steps"] += 1
        else:
            rollouts = rlcd_rollouts(model, reference, batch, args)
            for _ in range(args.num_iterations):
                optimizer.zero_grad(set_to_none=True)
                for state, group in zip(batch, rollouts):
                    losses = []
                    for (question, logits), rollout in zip(question_logits(model, state), group):
                        loss, terms = rlcd_loss(logits, question, rollout, args)
                        accumulate(terms | {"loss": loss.detach()})
                        c = hard_correct(logits.detach(), question.target)
                        if c is not None:
                            hard, correct = hard + 1, correct + c
                        if is_dual(model):
                            losses.append(loss)
                        else:
                            backward_question(loss, len(state.questions), len(batch))
                    if is_dual(model):
                        combined = losses[0] if len(losses) == 1 else torch.stack(losses).sum()
                        backward_question(combined, len(state.questions), len(batch))
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                progress["optimizer_steps"] += 1

        progress["step"] += 1
        row = {"step": progress["step"], "epoch": progress["epoch"], "optimizer_steps": progress["optimizer_steps"],
               "states": len(batch), "tokens": sum(s.tokens for s in batch),
               "lr_factor": lr_factor, "grad_norm": float(grad_norm), "skipped_records": skipped,
               "accuracy": correct / hard if hard else None,
               "seconds": round(time.perf_counter() - started, 3)}
        row |= {key: value / count for key, value in sums.items()}
        if torch.cuda.is_available() and model.device.type == "cuda":
            row["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        if progress["step"] % args.log_every == 0:
            shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            print(json.dumps(shown), flush=True)

        if eval_records and progress["step"] % args.eval_every == 0:
            result = evaluate(model, eval_compiler, eval_records, args.eval_limit)
            print(json.dumps({"eval_step": progress["step"], **result}), flush=True)
            metrics_file.write(json.dumps({"eval_step": progress["step"], **result}) + "\n")
            if progress["best_accuracy"] is None or result["accuracy"] > progress["best_accuracy"]:
                progress["best_accuracy"] = result["accuracy"]
                save_checkpoint(output / "best", model, None, processor, args, dict(progress))
        if progress["step"] % args.save_every == 0:
            save_checkpoint(output / "latest", model, optimizer, processor, args, dict(progress))

    save_checkpoint(output / "latest", model, optimizer, processor, args, dict(progress))
    metrics_file.close()
    print(f"Finished at step {progress['step']}; checkpoint: {output / 'latest'}", flush=True)


if __name__ == "__main__":
    main()
