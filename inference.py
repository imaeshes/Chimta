#!/usr/bin/env python
"""
SafetyDecision — inference.py

Runs the decision checkpoint produced by training.py on videos/images and
returns Valen-style answers: probabilities over the candidate set.

Two answer mechanisms:
  1. generation (always on)      — the model emits its trained compact JSON
  2. --probabilities (optional)  — for every question, each candidate value is
     teacher-forced and scored by its true sequence log-prob, then softmaxed.
     This is the generative twin of Valen's softmax decision head: same
     interface, different plumbing.

Valen file map, for your learning:
    answer scoring here        ~ valen/modeling/qwen/heads.py (decision heads)
    output record shape here   ~ valen/evaluation/inference.py
    question preset below      ~ request.questions in docs/data-format.md

Usage:
    # one video, plain decision
    python inference.py --ckpt ./ckpt-safety-v1 --video clips/cam1_0142.mp4

    # with real probabilities per question
    python inference.py --ckpt ./ckpt-safety-v1 --image frames/fall_0042.jpg --probabilities

    # batch a JSONL of records -> predictions.jsonl (like valen.inference)
    python inference.py --ckpt ./ckpt-safety-v1 --data ./dataset/eval.jsonl \
        --pred-out predictions.jsonl
"""

import argparse
import json
import math

import cv2
import numpy as np
import torch
from PIL import Image

from training import SYSTEM_PROMPT, build_prompt_text  # prompt contract: single source of truth

# Default question preset = the training task. Override with --questions file
# (same shape: {"qid": {"type":..., "instructions":..., "criteria":...}}).
DEFAULT_QUESTIONS = {
    "activity": {"type": "choice", "instructions": "Which activity is the main event?",
                 "criteria": {"fall": "A person falls or is on the ground unexpectedly",
                              "fight": "Two or more people fighting",
                              "normal": "No unusual activity"}},
    "person_down": {"type": "noul", "instructions": "A person has fallen and does not get up."},
    "risk": {"type": "score", "instructions": "Rate the safety risk.",
             "criteria": ["Calm", "Unusual motion", "Fall happening", "Fight starting",
                          "Emergency"]},
}

ALERT_THRESHOLD = 0.5  # alarm when p(fall)+p(fight) over this, or p(person_down) over this


# ------------------------------------------------------------------ args & data

def parse_args():
    p = argparse.ArgumentParser(description="Safety decision inference")
    p.add_argument("--ckpt", required=True, help="training.py --out dir (or a base model id to compare)")
    p.add_argument("--video")
    p.add_argument("--image")
    p.add_argument("--state", default="Monitor this camera feed for incidents.")
    p.add_argument("--questions", help="JSON file overriding the question preset")
    p.add_argument("--data", help="JSONL batch of records (one video/image per line)")
    p.add_argument("--pred-out", help="write predictions JSONL (pairs with --data)")
    p.add_argument("--probabilities", action="store_true",
                   help="logit-score every candidate (choice/noul/score)")
    p.add_argument("--frames", type=int, default=8, help="frames sampled per video")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.questions:
        with open(args.questions, "r", encoding="utf-8") as f:
            args.questions = json.load(f)
    else:
        args.questions = dict(DEFAULT_QUESTIONS)
    if not (args.video or args.image or args.data):
        p.error("pass --video/--image or --data")
    if args.video and args.image:
        p.error("pass either --video or --image, not both")
    return args


def load_media(path, is_video, k):
    """Video -> k uniformly sampled RGB frames; image -> single frame."""
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


# ---------------------------------------------------------------------- loading

def load_model(path, device):
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    return model, proc, tok


# ------------------------------------------------------------------- inference

def encode_record(proc, rec, questions, k, device):
    """One record -> processor tensors on `device` (batch of 1).

    `rec` is dict with keys: state, optional id, and exactly one of
    video/image (path str). `questions` shape = default preset.
    """
    media = rec.get("video") or rec.get("image")
    is_video = rec.get("video") is not None
    media_abs = media
    frames = load_media(media_abs, is_video, k)
    content = ([{"type": "video", "video": frames, "max_frames": k}]
               if is_video else [{"type": "image", "image": frames[0]}])
    qrec = {"state": rec.get("state", ""), "questions": questions,
            "video": media if is_video else None,
            "image": None if is_video else media}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content + [{"type": "text", "text": build_prompt_text(qrec)}]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], videos=frames if is_video else [],
               images=[] if is_video else frames,
               return_tensors="pt", padding=False)
    enc = {key: v.to(device) if hasattr(v, "to") else v for key, v in enc.items()}
    return enc


def generate_decision(model, tok, enc, max_new_tokens=64):
    """Greedy generation -> parsed decision dict (+ raw text if unparseable)."""
    eos = tok.eos_token_id
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}, text
    try:
        return json.loads(text[start:end + 1]), text
    except json.JSONDecodeError:
        return {}, text


def candidates_for(qid, q):
    t = q["type"]
    if t == "choice":
        return list(q["criteria"].keys())
    if t == "noul":
        return ["true", "false"]
    return [str(i) for i in range(len(q["criteria"]))]


def prefill_for(qid, q):
    """Assistant prefix placing the model exactly before the answer value.

    Training output shape is {"activity": "fall", "person_down": true, "risk": 2}
    -> a choice value sits behind a quote, noul/score values sit bare.
    """
    if q["type"] == "choice":
        return "{" + json.dumps(qid) + ': "'    # -> {"activity": "
    return "{" + json.dumps(qid) + ": "         # -> {"person_down":


@torch.inference_mode()
def score_candidates(model, tok, enc, qid, q):
    """P(candidate | prompt + prefill), softmaxed across candidates.

    One short forward per candidate (cheap — candidates are 1–2 tokens).
    """
    cands = candidates_for(qid, q)
    pre = tok(prefill_for(qid, q), add_special_tokens=False,
              return_tensors="pt")["input_ids"]
    base = torch.cat([enc["input_ids"], pre.to(enc["input_ids"].device)], dim=1)
    scores = {}
    for cand in cands:
        cids = tok(cand, add_special_tokens=False, return_tensors="pt")["input_ids"]
        inp = torch.cat([base, cids.to(base.device)], dim=1)
        logits = model(input_ids=inp[:, :-1]).logits          # rows predict token i+1
        lp = torch.log_softmax(logits[0, -cids.shape[1]:, :].float(), dim=-1)
        tok_lp = lp.gather(-1, cids[0].to(lp.device).unsqueeze(-1)).sum().item()
        scores[cand] = tok_lp
    m = max(scores.values())                                   # stable softmax
    exp = {c: math.exp(v - m) for c, v in scores.items()}
    z = sum(exp.values())
    return {c: e / z for c, e in exp.items()}


def summarize_alarm(probs):
    """Deployment-flavored summary line (fall/fight alert)."""
    if "activity" not in probs:
        return None
    act = probs["activity"]
    p_fall, p_fight = act.get("fall", 0.0), act.get("fight", 0.0)
    risk = probs.get("risk", {})
    risk_ev = sum(float(k) * v for k, v in risk.items()) if risk else None
    pdown = None
    if "person_down" in probs:
        pdown = probs["person_down"].get("true", 0.0)
    alarms = []
    if p_fall >= ALERT_THRESHOLD:
        alarms.append("FALL")
    if p_fight >= ALERT_THRESHOLD:
        alarms.append("FIGHT")
    if pdown is not None and pdown >= ALERT_THRESHOLD:
        alarms.append("PERSON_DOWN")
    line = f"p_fall={p_fall:.2f} p_fight={p_fight:.2f}"
    if pdown is not None:
        line += f" p_down={pdown:.2f}"
    if risk_ev is not None:
        line += f" risk_ev={risk_ev:.2f}"
    return {"alarms": alarms, "summary": line}


# ------------------------------------------------------------------------ main

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, proc, tok = load_model(args.ckpt, args.device if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    def run_one(rec, questions, line_no=None):
        enc = encode_record(proc, rec, questions, args.frames, device)
        dec, raw = generate_decision(model, tok, enc)
        out = {"id": rec.get("id"),
               "source": rec.get("video") or rec.get("image"),
               "decision": dec}
        if not dec:
            out["raw"] = raw
        if args.probabilities:
            probs = {qid: score_candidates(model, tok, enc, qid, q)
                     for qid, q in questions.items()}
            out["probabilities"] = probs
            alarm = summarize_alarm(probs)
            if alarm is not None:
                out["alarm"] = alarm
        if line_no is not None:
            tag = f"line {line_no}: " if line_no else ""
            print(f"[{tag}] {json.dumps(out, ensure_ascii=False)}")
        return out

    if args.video or args.image:
        rec = {"state": args.state, "video": args.video, "image": args.image}
        result = run_one(rec, args.questions, line_no="")
        if result["decision"]:
            print("decision:", json.dumps(result["decision"], ensure_ascii=False))
        if args.probabilities:
            for qid, dist in result["probabilities"].items():
                pretty = ", ".join(f"{c}={v:.3f}"
                                   for c, v in sorted(dist.items(), key=lambda kv: -kv[1]))
                print(f"  {qid}: {pretty}")
            if result.get("alarm"):
                print("ALARM:", ", ".join(result["alarm"]["alarms"]) or "none",
                      "|", result["alarm"]["summary"])
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # batch mode over --data JSONL
    preds = []
    with open(args.data, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[skip] line {lineno}: bad JSON ({e})")
                continue
            media = rec.get("video") or rec.get("image")
            if not media:
                print(f"[skip] line {lineno}: no video/image")
                continue
            preds.append(run_one(rec, rec.get("questions") or args.questions,
                                 line_no=lineno))
    if args.pred_out:
        with open(args.pred_out, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[done] wrote {args.pred_out}")


if __name__ == "__main__":
    main()
