#!/usr/bin/env python3
"""Run a checkpoint from training.py on Valen JSONL, or on one ad-hoc question.

Batch:   python inference.py --checkpoint out/rlcd/latest --data eval.jsonl --output preds.jsonl
Single:  python inference.py --checkpoint out/rlcd/latest --text "The light is red." \
             --type choice --instructions "Which color?" --criteria '{"red": "Red", "green": "Green"}'
Labeled records also produce an accuracy / Brier summary.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training import (MODEL_NAME, Compiler, answer, hard_correct, load_for_inference,  # noqa: E402
                      read_jsonl, sft_loss)


def single_record(args):
    content = [{"type": "text", "text": args.text}] if args.text else []
    content += [{"type": "image_url", "image_url": {"url": str(Path(p).resolve())}} for p in args.image]
    content += [{"type": "video_url", "video_url": {"url": str(Path(p).resolve())}} for p in args.video]
    if not content:
        sys.exit("Provide --text, --image or --video")
    question = {"type": args.type, "instructions": args.instructions}
    if args.type != "noul":
        if not args.criteria:
            sys.exit("--criteria is required for choice (JSON object) and score (JSON list)")
        question["criteria"] = json.loads(args.criteria)
    return {"group_id": "cli", "request": {"state": {"messages": [{"role": "user", "content": content}]},
                                           "questions": {"q": question}}}


@torch.inference_mode()
def predict(model, compiler, record, temperature):
    started = time.perf_counter()
    state = compiler.compile(record)
    answers, labels = {}, {}
    for question in state.questions:
        logits = model(question).float()
        answers[question.qid] = answer(question, logits, temperature)
        if question.target is not None:
            _, terms = sft_loss(logits, question.target)
            labels[question.qid] = {"correct": hard_correct(logits, question.target),
                                    "brier": float(terms["brier"])}
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    response = {"model": MODEL_NAME, "group_id": record.get("group_id"), "answers": answers,
                "usage": {"compute_tokens": state.tokens, "output_tokens": 0},
                "latency_ms": round(1000 * (time.perf_counter() - started), 1)}
    return response, labels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Directory written by training.py (latest/ or best/)")
    p.add_argument("--base-model", help="Override the base model path recorded in the checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--data", help="Valen JSONL; targets are optional")
    p.add_argument("--output", help="Prediction JSONL (default: stdout)")
    p.add_argument("--text")
    p.add_argument("--image", action="append", default=[])
    p.add_argument("--video", action="append", default=[])
    p.add_argument("--type", choices=("choice", "noul", "score"), default="choice")
    p.add_argument("--instructions", default="Choose the best answer.")
    p.add_argument("--criteria", help='choice: {"key": "description"}; score: ["level 0", "level 1", ...]')
    args = p.parse_args()
    if args.temperature <= 0:
        p.error("--temperature must be positive")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available to this PyTorch build; see README (RTX 50xx needs CUDA 12.8 wheels)")

    model, processor, config = load_for_inference(args.checkpoint, args.device, args.base_model)
    if args.data:
        records, root = read_jsonl(args.data), Path(args.data).parent
    else:
        records, root = [single_record(args)], Path(".")
    compiler = Compiler(processor, root, config["max_length"], config["media_kwargs"])

    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    correct = hard = 0
    brier, labeled, latencies = 0.0, 0, []
    try:
        for record in records:
            response, labels = predict(model, compiler, record, args.temperature)
            latencies.append(response["latency_ms"])
            out.write(json.dumps(response, ensure_ascii=False) + "\n")
            for label in labels.values():
                labeled += 1
                brier += label["brier"]
                if label["correct"] is not None:
                    hard += 1
                    correct += label["correct"]
    finally:
        if out is not sys.stdout:
            out.close()
    summary = {"records": len(records), "mean_latency_ms": round(sum(latencies) / len(latencies), 1)}
    if labeled:
        summary |= {"labeled_questions": labeled, "hard_label_accuracy": correct / hard if hard else None,
                    "mean_brier": brier / labeled}
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
