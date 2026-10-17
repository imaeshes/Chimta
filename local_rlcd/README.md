# Valen full fine-tuning (SFT + RLCD) on one 16 GB GPU

Two scripts that train and run a Valen decision model, Qwen3.5-0.8B backbone plus decision head, on a single RTX 5070 Ti (16 GB):

- `training.py` trains every weight (language model, vision tower and decision head) with `--mode sft` or `--mode rlcd`.
- `inference.py` runs a trained checkpoint on a JSONL file or on one question from the command line.

The scripts don't depend on the `valen` package. They read the same JSONL format ([docs/data-format.md](../docs/data-format.md)) and copy Valen's prompt compiler and decision heads. The compiler output was checked token for token against `valen/data/compilers/qwen.py` on the smoke data.

How this differs from `python -m valen.train`:

| | `valen.train` | `local_rlcd/training.py` |
| --- | --- | --- |
| Backbone update | LoRA on the language tower (plus merger / top vision blocks) | **Full fine-tuning** of all weights, or freeze parts with flags |
| Hardware | Multi-GPU data parallel | One GPU |
| Checkpoint | Trainable deltas only; needs the base model | Full weights in `model.safetensors` (`backbone.*`, `head.*`) |

## What "RLCD" means here

RLCD is the reinforcement learning stage that Valen implements in [valen/training/rlcd.py](../valen/training/rlcd.py). It is not text generation RL. For each question:

1. The current policy, which is the softmax over the decision head's candidate logits, samples `group_size` candidate actions with replacement.
2. Each sampled action \(a_i\) with probability \(p_i\) and label \(y_i\) gets the reward
   \(r_i = w_c\,y_i - w_f\,[\,y_i(1-p_i)^2 + (1-y_i)p_i^2\,]\).
   A correct answer is rewarded, and so is confidence that matches correctness.
3. Group-normalised advantages are used in a PPO-style clipped ratio loss.
4. The loss adds an exact categorical KL to a frozen reference policy (the model as it was when RLCD started) and a Brier term:
   \(L = L_\text{clip} + \beta\,\mathrm{KL}(\pi_\theta\Vert\pi_\text{ref}) + w_B\sum_k(\pi_\theta(k)-y_k)^2\).

Rollouts are fixed for `--num-iterations` optimizer updates. The defaults match `configs/train/qwen/rlcd_*.json`. RLCD needs labels and a head that already works, so **train SFT first**.

## Install (RTX 5070 Ti / Blackwell)

The 5070 Ti is compute capability sm_120 and needs **PyTorch ≥ 2.7 built for CUDA 12.8**. The repository's pinned `torch==2.6.0` does not run on it.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.4.0" safetensors accelerate "av==16.1.0" pillow numpy
pip install bitsandbytes                                   # 8-bit AdamW (default optimizer)
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Optional speed-up: Qwen3.5 uses Gated DeltaNet layers. Without `flash-linear-attention` and `causal-conv1d`, Transformers prints "fast path is not available" and uses a slower PyTorch fallback. Both install most easily on Linux or WSL2:

```bash
pip install flash-linear-attention causal-conv1d
```

Download the base model (the script also accepts the Hub id directly):

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

## Data

Use Valen JSONL. One record per line with `request.state`, `request.questions` (`choice`, `noul` or `score`) and `targets`. Media paths are relative to the JSONL file. Only labeled questions are trained. Records longer than `--max-length` tokens are skipped with a message, never truncated. Keep the train and eval files split by `group_id`.

```json
{"group_id": "light-1", "request": {"state": {"messages": [{"role": "user", "content": [
  {"type": "text", "text": "Is crossing allowed?"}, {"type": "image_url", "image_url": {"url": "img/1.png"}}]}]},
  "questions": {"color": {"type": "choice", "instructions": "Light color?",
    "criteria": {"red": "Red light", "green": "Green light", "unknown": "Not visible"}}}},
 "targets": {"color": {"probabilities": {"red": 0.0, "green": 1.0, "unknown": 0.0}}}}
```

## Recommended recipe

Run from the repository root. Each stage writes to its own output directory.

```bash
# 1. Head warmup: backbone frozen, head learns to read Qwen features (fast).
python local_rlcd/training.py --mode sft --freeze-backbone \
  --base-model models/Qwen3.5-0.8B --data data/train.jsonl --eval-data data/eval.jsonl \
  --output output/local/sft_head --max-steps 300 --lr-head 2e-4

# 2. Full SFT from the warm head.
python local_rlcd/training.py --mode sft --init output/local/sft_head/latest \
  --data data/train.jsonl --eval-data data/eval.jsonl \
  --output output/local/sft_full --max-steps 2000 --epochs 3

# 3. Full RLCD from the SFT checkpoint (best/ = highest eval accuracy).
python local_rlcd/training.py --mode rlcd --init output/local/sft_full/best \
  --data data/rl.jsonl --eval-data data/eval.jsonl \
  --output output/local/rlcd_full --max-steps 1000

# Continue an interrupted run (same flags, weights + optimizer + data cursor restored):
python local_rlcd/training.py --mode rlcd --resume output/local/rlcd_full/latest \
  --data data/rl.jsonl --output output/local/rlcd_full --max-steps 1000
```

When `--init` or `--resume` is set, the base model and head architecture are read from the checkpoint. Valen's own experiments split the data, for example 70k SFT and 30k RLCD records out of 100k.

Default learning rates (override with `--lr-backbone`, `--lr-vision`, `--lr-head`):

| | backbone (language) | vision tower | head |
| --- | --- | --- | --- |
| SFT | 1e-5 | 2e-6 | 2e-4 |
| RLCD | 1e-6 | 2e-7 | 1e-5 |

Schedule: linear warmup (`--warmup-steps 20`), then cosine to 10% (`--schedule constant` keeps Valen's constant rate). One step is `--states-per-step` records, with gradients accumulated per question. Valen's weighting applies: mean over questions within a record, then over records.

## Memory on 16 GB

Rough arithmetic for about 0.85B parameters (estimates, not measurements):

| Item | Size |
| --- | --- |
| FP32 master weights (BF16 autocast compute) | ~3.4 GB |
| FP32 gradients | ~3.4 GB |
| 8-bit AdamW states (`--optim adamw8bit`, default) | ~1.7 GB |
| RLCD frozen reference, BF16 (only when `--beta > 0`) | ~1.7 GB |
| Activations (gradient checkpointing on by default) | grows with tokens / media |

That is about 9–10 GB before activations, which leaves room for text and moderate image prompts. Plain `--optim adamw` adds about 5 GB of optimizer state and will usually not fit with RLCD. If you run out of memory, try these in order:

1. Lower image and video resolution or frames, for example `--media-kwargs '{"images_kwargs":{"size":{"longest_edge":200704,"shortest_edge":3136}},"videos_kwargs":{"fps":null,"num_frames":8,"size":{"longest_edge":1048576,"shortest_edge":3136}}}'`
2. `--reference-device cpu` (RLCD only; slower rollouts)
3. `--freeze-embeddings` (the 248k-token embedding table is a large share of the parameters and adds little for a decision head)
4. `--freeze-vision`
5. `--optim adafactor`, or a lower `--max-length`

Memory does not depend on `--states-per-step`, because each question is backpropagated before the next one is run. Check `peak_gpu_gb` in the logs.

## Outputs

```text
<output>/metrics.jsonl          one row per step (+ eval rows): loss, accuracy, reward, KL, clip fraction, peak_gpu_gb
<output>/latest/                model.safetensors, valen_local.json, processor files, trainer_state.pt (resume)
<output>/best/                  best eval accuracy (weights only, use with --init or inference)
<output>/reference.safetensors  RLCD frozen reference policy (reused on --resume)
```

`model.safetensors` uses the same `backbone.*` / `head.*` key names as Valen's merged releases. `--save-bf16` halves checkpoint size, but resumed runs then lose FP32 precision.

## Inference

```bash
# JSONL in, Valen response JSONL out (prints accuracy/Brier if targets exist)
python local_rlcd/inference.py --checkpoint output/local/rlcd_full/best \
  --data data/eval.jsonl --output output/local/preds.jsonl

# One question
python local_rlcd/inference.py --checkpoint output/local/rlcd_full/best \
  --image photo.jpg --text "Look at the traffic light." --type choice \
  --instructions "Which color is the light?" --criteria '{"red": "Red light", "green": "Green light"}'
```

The response format is the same as Valen's. `choice` returns `probabilities`, `choice` and `confidence`. `noul` returns `P(true)`. `score` returns the expected level, `probabilities` and `legend`. No answer tokens are generated (`output_tokens: 0`). Inference runs the backbone in BF16 on CUDA.

## What was verified

These checks ran on CPU, using a tiny random Qwen3.5 with the real architecture (DeltaNet + full attention, vision tower) and the real Qwen3.5 tokenizer and processor:

- SFT, RLCD from `--init`, RLCD `--resume`, eval, `best/` / `latest/` saving, and both inference modes all completed on `data/smoke/train.jsonl` (text, image and video records).
- The compiler's input tensors, candidate and decision positions, and role spans are identical to Valen's compiler for all 69 smoke branches, including shuffled choice orders.

Not yet verified: training on a real RTX 5070 Ti with Qwen3.5-0.8B weights, the real memory peak, throughput, and model quality. Run a short job first, for example `--max-steps 5`, and check `peak_gpu_gb`.
