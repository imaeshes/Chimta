#!/usr/bin/env python3
"""Single-GPU full fine-tuning of a Valen decision model: Qwen3.5 backbone + decision head.

Two objectives share one loop:
  --mode sft   distribution cross-entropy (+ optional Brier) on labeled candidates
  --mode rlcd  Valen's RLCD: GRPO over the candidate distribution, rewarded for
               correctness and calibrated confidence, with exact KL to a frozen
               reference policy and a Brier auxiliary term (see valen/training/rlcd.py)

Unlike `python -m valen.train` (LoRA on the language tower), every backbone weight is
trainable here unless frozen explicitly. Data is the Valen JSONL format (docs/data-format.md).
The prompt compiler and decision heads reproduce valen/data/compilers/qwen.py and
valen/modeling/qwen/heads.py so that readout positions match the released Valen models.
"""
import argparse
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

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import nn

MODEL_NAME = "Valen-local"
CONFIG_FILE = "valen_local.json"
WEIGHTS_FILE = "model.safetensors"
STATE_FILE = "trainer_state.pt"
REFERENCE_FILE = "reference.safetensors"

# Matches the released Valen checkpoints (config.json -> valen_media_kwargs).
DEFAULT_MEDIA_KWARGS = {
    "images_kwargs": {"size": {"longest_edge": 262144, "shortest_edge": 65536}},
    "videos_kwargs": {"fps": None, "num_frames": 16,
                      "size": {"longest_edge": 2097152, "shortest_edge": 131072}},
}
DEFAULT_HEAD = {"head_type": "mixer", "head_width": 256, "head_layers": 2,
                "head_token_hidden_dim": 16, "head_channel_hidden_dim": 1024}


# --------------------------------------------------------------------------- data

def candidates(question):
    kind = question.get("type")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError("instructions must be nonempty text")
    criteria = question.get("criteria")
    if kind == "noul":
        # Exact Valen candidate text; the released heads were trained on it.
        return [("true", "True / 是：满足问题中的条件。"), ("false", "False / 否：不满足问题中的条件。")]
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


def load_backbone(model_path, attn_implementation="eager"):
    from transformers import Qwen3_5ForConditionalGeneration
    # Load the published container and keep only its backbone. Loading Qwen3_5Model
    # directly can silently re-initialize the language tower in Transformers 5.4.
    container, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation=attn_implementation, output_loading_info=True)
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
    for name, value in model.state_dict().items():
        value = value.detach()
        if name.startswith("backbone.") and value.is_floating_point():
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


def build_model(base_model, head_config, device, attn="eager", gradient_checkpointing=True):
    backbone = load_backbone(base_model, attn)
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
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = build_model(base_model or config["base_model"], config["head"], "cpu", attn, gradient_checkpointing=False)
    load_weights(model, checkpoint / WEIGHTS_FILE)
    if torch.device(device).type == "cuda":
        model.backbone.to(torch.bfloat16)
    model.head.float()
    model.to(device).eval()
    return model, processor, config


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
        for question in state.questions:
            logits = model(question).float()
            if not torch.isfinite(logits).all():
                raise ValueError(f"Non-finite logits for {state.group_id}/{question.qid}")
            log_probs = logits.log_softmax(-1)
            target = torch.as_tensor(question.target, device=logits.device, dtype=logits.dtype)
            actions = torch.multinomial(log_probs.exp(), args.group_size, replacement=True)
            reward, correctness, confidence, _ = decision_reward(
                log_probs.exp(), target, actions, args.correctness_weight, args.confidence_weight)
            ref = None
            if reference is not None:
                ref = reference(question).float().log_softmax(-1).to(logits.device)
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
        for question in state.questions:
            logits = model(question).float()
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


def make_optimizer(model, args):
    rates = {"head": args.lr_head, "merger": args.lr_merger, "vision": args.lr_vision, "language": args.lr_backbone}
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        part = ("head" if name.startswith("head.") else "merger" if ".visual.merger." in name
                else "vision" if ".visual." in name else "language")
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
    reference.backbone.gradient_checkpointing_disable()
    if path.exists():
        load_weights(reference, path)
    else:
        save_weights(model, path, backbone_dtype=torch.bfloat16)
    if torch.device(device).type == "cuda":
        reference.backbone.to(torch.bfloat16)
    reference.head.float()
    return reference.to(device).eval()


def save_checkpoint(directory, model, optimizer, processor, args, progress):
    directory.mkdir(parents=True, exist_ok=True)
    save_weights(model, directory / WEIGHTS_FILE, torch.bfloat16 if args.save_bf16 else torch.float32)
    config = {"model": MODEL_NAME, "base_model": args.base_model, "head": args.head,
              "max_length": args.max_length, "media_kwargs": args.media_kwargs,
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
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B", help="HF id or local Qwen3.5 directory")
    p.add_argument("--data", required=True, help="Valen JSONL; media paths are relative to its directory")
    p.add_argument("--eval-data", help="Optional held-out Valen JSONL")
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
    p.add_argument("--optim", choices=("adamw8bit", "adafactor", "adamw"), default="adamw8bit")
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
    if args.init and args.resume:
        p.error("--init starts a new run and --resume continues one; choose one")
    if args.group_size < 2 or args.num_iterations < 1 or not 0 < args.clip_epsilon < 1:
        p.error("RLCD requires group_size >= 2, num_iterations >= 1 and 0 < clip_epsilon < 1")
    if args.correctness_weight == args.confidence_weight == 0:
        p.error("At least one RLCD reward weight must be positive")
    if args.states_per_step < 1 or args.max_steps < 1:
        p.error("--states-per-step and --max-steps must be positive")
    args.head = dict(DEFAULT_HEAD) if args.head_type == "mixer" else {"head_type": "bilinear", "projection_dim": 256}
    return args


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoProcessor

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        if not torch.cuda.is_available():
            sys.exit("CUDA is not available to this PyTorch build; see README (RTX 50xx needs CUDA 12.8 wheels)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    source = Path(args.resume or args.init) if (args.resume or args.init) else None
    if source is not None:
        saved = json.loads((source / CONFIG_FILE).read_text(encoding="utf-8"))
        if saved["base_model"] != args.base_model:
            print(f"Using checkpoint base model {saved['base_model']!r}", flush=True)
            args.base_model = saved["base_model"]
        args.head = saved["head"]

    processor = AutoProcessor.from_pretrained(args.base_model)
    model = build_model(args.base_model, args.head, "cpu", args.attn, not args.no_gradient_checkpointing)
    if source is not None:
        load_weights(model, source / WEIGHTS_FILE)
        print(f"Loaded weights from {source}", flush=True)
    elif args.mode == "rlcd":
        print("WARNING: RLCD from a random decision head. Run --mode sft first and pass --init.", flush=True)

    reference = None
    if args.mode == "rlcd" and args.beta > 0:
        # Copied on CPU to avoid a second FP32 model on the GPU. Resumed runs reuse
        # the saved original reference, not the current weights.
        reference = make_reference(model, args.reference_device or args.device, output / REFERENCE_FILE)
    model.to(args.device)
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

    records = [r for r in read_jsonl(args.data) if is_labeled(r)]
    eval_records = [r for r in read_jsonl(args.eval_data) if is_labeled(r)] if args.eval_data else None
    compiler = Compiler(processor, Path(args.data).parent, args.max_length, args.media_kwargs)
    eval_compiler = (Compiler(processor, Path(args.eval_data).parent, args.max_length, args.media_kwargs)
                     if args.eval_data else None)
    print(f"{len(records)} labeled training records; mode={args.mode}; optimizer={args.optim}", flush=True)

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
                for question in state.questions:
                    logits = model(question)
                    loss, terms = sft_loss(logits, question.target, args.sft_brier_weight)
                    backward_question(loss, len(state.questions), len(batch))
                    accumulate(terms | {"loss": loss.detach()})
                    c = hard_correct(logits.detach(), question.target)
                    if c is not None:
                        hard, correct = hard + 1, correct + c
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            progress["optimizer_steps"] += 1
        else:
            rollouts = rlcd_rollouts(model, reference, batch, args)
            for _ in range(args.num_iterations):
                optimizer.zero_grad(set_to_none=True)
                for state, group in zip(batch, rollouts):
                    for question, rollout in zip(state.questions, group):
                        logits = model(question)
                        loss, terms = rlcd_loss(logits, question, rollout, args)
                        backward_question(loss, len(state.questions), len(batch))
                        accumulate(terms | {"loss": loss.detach()})
                        c = hard_correct(logits.detach(), question.target)
                        if c is not None:
                            hard, correct = hard + 1, correct + c
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
