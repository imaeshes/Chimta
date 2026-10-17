# SafetyDecision

Fall & fight detection from camera feeds, in the **Valen decision-model style**:
media + questions in → decision probabilities out. Full-parameter SFT of
**Qwen/Qwen3.5-0.8B**, sized for the **RTX 5070 Ti (16 GB)** box.

> This is "Valen, your-size edition": the repo's architecture (decision head +
> softmax over candidates) is replaced by a generative twin — the model emits a
> strict decision JSON, and inference.py converts it into true probabilities by
> logit-scoring every candidate. Same interface, honest trade-off (see §8).

```
(project root)/
├── README.md          ← you are here
├── training.py        # decision-SFT trainer  (~ valen/train.py + sft.py)
├── inference.py       # decision + probabilities (~ valen/inference.py)
├── dataset/           # (you create) media + metadata.jsonl
└── ckpt-safety-v1/    # (training creates) final checkpoint
```

The prompt contract lives in ONE function (`training.build_prompt_text`) and is
imported by inference.py — train/infer always compile identical requests. That
mirrors valen/data/compilers/qwen.py being the single source of truth.

## 1. Who runs where

- **Train on the Ubuntu box** (5070 Ti): `ssh home`
- **Inference anywhere** with the checkpoint (same venv)
- Jetson Orin deployment notes at the bottom (§9) — inference.py runs there too
  once the Jetson venv is set up (torch from the jetson-ai-lab index).

## 2. Environment (Ubuntu, 5070 Ti = Blackwell sm_120 → cu128 torch)

```bash
ssh home
python -m venv .venv && source .venv/bin/activate

pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate pillow opencv-python-headless
pip install bitsandbytes            # for --optim adamw_bnb_8bit (default)

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Copy the folder over:

```powershell
# from the work PC (-- exclude nothing, the whole folder is the project)
scp -r "Default Project" home:~/safetydec
```

## 3. Dataset format (Valen-style)

`dataset/metadata.jsonl` — one record per line (single line, not pretty-printed).
Copy one of these blocks as-is:

```json
{"video": "clips/cam1_0142.mp4", "id": "cam1-0142",
 "state": "Monitor this camera clip for incidents.",
 "questions": {
   "activity": {"type": "choice", "instructions": "Which activity is the main event?",
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
```

Image records: replace `"video"` with `"image": "frames/x.jpg"`.

Rules (validated at load, mirrors valen/data/schema.py):

- One question key per id; `targets` must cover every question id.
- `target` probabilities must sum to 1.0.
- choice target keys ∈ criteria keys; noul ∈ {"true","false"}; score ∈ {"0".."K-1"}.
- Soft labels OK — training uses the argmax as the supervised answer
  (`distribution → hard answer`, generative-SFT simplification; real Valen
  optimizes the full distribution with cross-entropy).

Data hygiene (applies later, when converting real footage):

- Use **images** for falls/static events (cheap tokens); **video** for fights
  (temporal). Aim ≥ 2:1 normal : incident.
- Same camera-day → share an id prefix; split train/val **by camera or day**,
  never shuffle clips blindly.

## 4. Train

```bash
cd ~/safetydec && source .venv/bin/activate

# dry sanity first: tiny run, watch loss go down
python training.py --data ./dataset --out ./ckpt-safety-smoke --epochs 0.05

# real run
python training.py --data ./dataset --out ./ckpt-safety-v1 \
    --epochs 2 --lr 1e-5 --frames 8
```

Knobs:

| Flag | Default | Note |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-0.8B` | 2B also fits full-SFT (tighter, lower `--frames`) |
| `--frames` | 8 | frames per video; falls spike briefly keep 8–12 |
| `--optim` | `adamw_bnb_8bit` | 8-bit Adam = the only reason 0.8B–2B *full* SFT fits 16 GB; else `adamw_torch` |
| `--no-freeze-vision` | off | vision tower frozen by default; unfreeze only with plenty of data |
| `--resume` | off | continue latest checkpoint in `--out` |

Expected VRAM (bf16 weights + 8-bit Adam + grad checkpointing, batch 1):
0.8B ≈ **5–8 GB**, 2B ≈ **13–15 GB** (tight: keep batch 1, frames ≤ 8).
Finance-note vs Valen: Valen keeps FP32 master weights (4+4+4+2 = 14–16 B/param
× 16 GB card OOMs at 2B full). 8-bit Adam is this project's memory budget hack.

Run inside `tmux` so SSH drops don't kill epochs.

## 5. Inference

```bash
# decision JSON for one clip
python inference.py --ckpt ./ckpt-safety-v1 --video clips/cam1_0142.mp4

# true probabilities per question + alarm line
python inference.py --ckpt ./ckpt-safety-v1 --image frames/fall_0042.jpg --probabilities

# batch: eval your held-out cameras -> predictions.jsonl (valen.inference style)
python inference.py --ckpt ./ckpt-safety-v1 --data ./dataset/eval.jsonl \
    --pred-out predictions.jsonl --probabilities
```

`--probabilities` output: per question, a distribution over candidates, e.g.
`activity: fall=0.91, normal=0.07, fight=0.02`. Alarm line fires at ≥0.5 on
fall / fight / person_down. Compare base vs trained:

```bash
python inference.py --ckpt Qwen/Qwen3.5-0.8B --video ... --probabilities   # base
python inference.py --ckpt ./ckpt-safety-v1 --video ... ... --probabilities # yours
```

## 6. Why this teaches you "just like Valen"

| Valen piece | here |
|---|---|
| `data-format.md` JSONL | §3 schema (same three question types) |
| `compilers/qwen.py` | `training.build_prompt_text` — one function, the whole prompt contract |
| `heads.py` (softmax head) | `inference.score_candidates` (logit-scored candidates) |
| `runner.py`/`sft.py` (loop + CE) | HF Trainer (same CE loss, generic loop) |
| checkpoint (`latest/`) | `ckpt-*/` HF-saveable dir |

Read the valen files and diff against these — you'll recognize what we
did and what Valen does better (checkpoints, token budgets, RL).

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| sm_120 kernel error | torch not built for Blackwell → reinstall from the cu128 index |
| CUDA OOM | batch 1 + lower `--frames`, or drop `--lr` epochs; 2B needs `--frames 8/4` |
| `bitsandbytes` fails | `pip install bitsandbytes` (Ubuntu) or `--optim adamw_torch` (more VRAM) |
| labels sum ≠ 1 errors | format the JSONL targets as 0.0 fractions |
| Model rambles instead of JSON | train longer; verify eval uses the trained `--ckpt`, not base |
| SSH killed the run | run inside tmux: `tmux new -s train` before launching |

## 8. What this setup does NOT have (vs. real Valen — deliberate)

- No learned decision head — probabilities come from answer-token logits.
- No distribution-level CE loss — hard-label only (soft labels→argmax).
- No RLCD stage, token-budget packing, multi-GPU sync, checkpoints with
  full optimizer state. Growth path: port the §3 JSONL 1:1 into valen's
  `docs/data-format.md` schema and graduate to the real training pipeline
  when the dataset size justifies it.

## 9. Deployment sketch (later milestone: Jetson AGX Orin 64GB)

- Export/deploy the checkpoint; inference.py is the single entry point.
- Edge venv: torch from `pypi.jetson-ai-lab.io/jp6/cu126` (aarch64), `sdpa`,
  bf16 OK on Orin; feed it snapshots (8 s interval) from an ffmpeg RTSP ring:
  `ffmpeg -rtsp_transport tcp -i rtsp://CAM -vf fps=1/8 -update 1 snap.jpg`
- Keep `ALERT_THRESHOLD` there; raise person_down threshold ≥ 0.7 before
  paging humans (false alarms are the #1 killer of monitoring systems).
