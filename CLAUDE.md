# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Prodigy** is an early-stage Python/PyTorch project for working with drum MIDI data. The dataset used is the [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) (stored under `data/groove/`), which contains MIDI recordings from human drummers across various styles (funk, jazz, rock, soul, hip-hop, Latin, etc.).

The Python package lives at `src/prodigy/` (src-layout), using `hatchling` as the build backend.

## Development Environment

This project uses Nix flakes for a reproducible dev environment with CUDA support.

```bash
nix develop          # Enter the dev shell (Python 3.11, PyTorch+CUDA, miditok, music21, …)
pip install -e .     # Install the package in editable mode (run once inside the shell)
```

## Common Commands

```bash
python -m prodigy.train                              # Train from scratch
python -m prodigy.generate \
    --checkpoint checkpoints/epoch_050.pt \
    --out generated.mid \
    --temperature 1.0 --top_k 50                     # Generate a drum MIDI file
```

## Architecture

The goal is a GPT-style autoregressive model that generates drum token sequences.

**Data pipeline** (`src/prodigy/data/`):
- `tokenizer.py` — wraps miditok's `REMI` tokenizer configured for 4/4 drum-only patterns (no chords/programs, 32 velocity buckets, 32nd-note grid). Saved to `tokenizer/` after the first run.
- `dataset.py` — converts MIDI files to flat token-id sequences via `symusic.Score`, then produces `(input, target)` pairs using a sliding window (50% overlap by default). Target is input shifted left by 1 — the next-token prediction objective.

**Model** (`src/prodigy/model/transformer.py`):
- Decoder-only Transformer (`DrumTransformer`): token embedding + learned positional embedding → N × `TransformerBlock` (pre-norm, causal self-attention, GELU feed-forward) → LayerNorm → linear head.
- Causal mask is a boolean upper-triangular buffer; weight tying links the output head to the token embedding.
- Default size: 6 layers, d_model=256, 8 heads, d_ff=1024 (~10 M params).

**Training** (`src/prodigy/train.py`):
- AdamW (β₂=0.95) + cosine LR decay with linear warm-up.
- Checkpoints saved to `checkpoints/epoch_NNN.pt` containing `model_cfg` and `model_state`.

**Generation** (`src/prodigy/generate.py`):
- Autoregressive sampling seeded with a `BAR_None` token.
- Temperature + top-k filtering; decodes token ids back to MIDI via `TokSequence` → `tokenizer.tokens_to_score`.

**Config** (`src/prodigy/config.py`):
- `ModelConfig` and `TrainConfig` dataclasses. Change `device = "cpu"` in `TrainConfig` for smoke-testing without a GPU.

## Data

- `data/groove/` — Groove MIDI Dataset, organized as `drummer{N}/session{N}/*.mid`
- `data/groove-v1.0.0-midionly.zip` — source archive
- MIDI filenames encode metadata: `{index}_{style}_{bpm}_{type}_4-4.mid` (type is `beat` or `fill`)
