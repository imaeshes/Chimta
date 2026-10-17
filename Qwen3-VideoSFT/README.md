# Qwen3-VideoSFT

Full-parameter (fullfat) supervised fine-tuning of **Qwen3-0.6B** on a video-annotated
text dataset, targeted at an **RTX 5070 Ti (16 GB)** on the Ubuntu box.

```
Qwen3-VideoSFT/
├── training.py      # full SFT trainer (bf16 + 8-bit AdamW)
├── inference.py     # single-prompt or chat inference on the trained checkpoint
├── dataset/         # (you create) videos + metadata.jsonl
└── ckpt-qwen3-fullsft/   # (training creates) final model + tokenizer
```

## 1. Who trains where

- **Train on the Ubuntu PC at home** (that's where the 5070 Ti lives).
- **Run everything via SSH from the work PC:**
  ```powershell
  ssh home
  ```
  (already configured on this machine via Tailscale)

## 2. Environment setup (Ubuntu)

The 5070 Ti is a Blackwell GPU (sm_120) → needs torch built for CUDA 12.8.

```bash
# python 3.10+ assumed; a venv is recommended
python -m venv .venv && source .venv/bin/activate

pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate bitsandbytes safetensors huggingface_hub

# sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Dataset

Create a `dataset/` folder with your videos and one `metadata.jsonl` inside —
one JSON object per line:

```json
{"video": "clips/001.mp4", "prompt": "What happens in this clip?", "response": "A player kicks the ball and scores."}
```

Rules:
- `prompt` + `response` are required (aliases `question/answer` or `user/assistant` also work).
- `video` is a provenance key — **Qwen3 is text-only**, so each video must already
  have its description written. Malformed rows are skipped automatically.

Copy `training.py` + `inference.py` and the dataset folder to the Ubuntu machine, e.g.:

```powershell
# from the work PC (uses the same SSH config)
scp -r Qwen3-VideoSFT home:~/videosft
```

## 4. Train

```bash
cd ~/videosft
python training.py --data ./dataset --out ./ckpt-qwen3-fullsft --epochs 3
```

Useful knobs:

| Flag | Default | Note |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-1.7B` also fits; 4B does **not** fit full-SFT on 16 GB |
| `--batch-size` | 4 | raise to 8 on 0.6B if VRAM allows |
| `--grad-acc` | 4 | effective batch = batch-size × grad-acc |
| `--optim` | `adamw_bnb_8bit` | VRAM-safe; swap to `adamw_torch` if bitsandbytes is missing |
| `--resume` | off | continue from the latest checkpoint in `--out` |
| `--max-len` | 1024 | token cap per sample |

Expected VRAM for 0.6B full SFT: **~5–6 GB** → plenty of headroom on the 16 GB card.
Long runs: use `tmux`/`screen` so an SSH drop doesn't kill training.

## 5. Inference

```bash
# single prompt
python inference.py --ckpt ./ckpt-qwen3-fullsft --prompt "Describe this clip: a kid scores a goal"

# interactive chat
python inference.py --ckpt ./ckpt-qwen3-fullsft --chat

# colder/safer sampling
python inference.py --ckpt ./ckpt-qwen3-fullsft --prompt "..." --temperature 0.6 --top-p 0.8
```

Thinking mode is **off by default** (short, direct answers). Add `--thinking` to enable
Qwen3's reasoning blocks.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `cuda` unavailable / sm_120 error | wrong torch build → reinstall from the cu128 index above |
| CUDA OOM | lower `--batch-size`, keep `--gradient-checkpointing`, raise `--grad-acc` |
| `bitsandbytes` install error | `pip install bitsandbytes>=0.43` or switch `--optim adamw_torch` |
| Training stops after SSH disconnect | run inside `tmux`; reconnect later (`tmux attach`) |
| Inference loads base-looking model | make sure `--ckpt` points to the `--out` dir of training, not the base model |

## 7. Reality check

Qwen3 processes **text**. Videos enter this pipeline only as their captions/dialogues,
never as pixels. If you later want actual frame-level video understanding, the right tool
is a VLM — e.g. **Qwen2.5-VL** (3B/7B) — with its own training pipeline (available on request).
