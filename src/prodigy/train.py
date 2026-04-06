"""
Training script.

From the repo root (inside `nix develop`):

    pip install -e .          # installs the package in editable mode (once)
    python -m prodigy.train

Checkpoints are written to checkpoints/epoch_NNN.pt.
To resume training, load the checkpoint in your own script:

    ckpt = torch.load("checkpoints/epoch_010.pt")
    model.load_state_dict(ckpt["model_state"])
    # Optionally restore optimizer state if you saved it too.
"""

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from prodigy.config import ModelConfig, TrainConfig
from prodigy.data.dataset import DrumDataset, collect_midi_paths, tokenize_files
from prodigy.data.tokenizer import build_tokenizer, save_tokenizer
from prodigy.model.transformer import DrumTransformer


def cosine_lr_with_warmup(step: int, cfg: TrainConfig, total_steps: int) -> float:
    """
    Linear warm-up for the first `warmup_steps` steps, then cosine decay to 0.

    Warm-up prevents large gradient updates early in training when the weights
    are still random. Cosine decay avoids an abrupt LR drop at the end.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def train() -> None:
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    midi_paths = collect_midi_paths(train_cfg.data_dir)
    print(f"Found {len(midi_paths)} MIDI files in {train_cfg.data_dir}")

    tokenizer = build_tokenizer()
    save_tokenizer(tokenizer, train_cfg.tokenizer_dir)

    model_cfg.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {model_cfg.vocab_size}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Tokenizing files…")
    sequences = tokenize_files(tokenizer, midi_paths)
    print(f"Successfully tokenized {len(sequences)} files")

    dataset = DrumDataset(sequences, seq_len=model_cfg.seq_len)
    print(f"Dataset: {len(dataset)} windows × {model_cfg.seq_len} tokens")

    n_val = max(1, int(0.1 * len(dataset)))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(
        train_set, batch_size=train_cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DrumTransformer(model_cfg).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    # AdamW with β₂=0.95 (slightly tighter than the default 0.999).
    # GPT papers found this works better for language/sequence modelling.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total_steps = len(train_loader) * train_cfg.max_epochs
    step = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Manually update LR each step (no scheduler object needed)
            lr = cosine_lr_with_warmup(step, train_cfg, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            _, loss = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping: prevents a single bad batch from destroying weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % train_cfg.log_every == 0:
                print(f"  epoch {epoch:3d} | step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e}")

        avg_train = running_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, targets=y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"epoch {epoch:3d} | train {avg_train:.4f} | val {val_loss:.4f}")

        if epoch % train_cfg.save_every == 0:
            ckpt = train_cfg.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model_cfg": model_cfg, "model_state": model.state_dict()}, ckpt)
            print(f"  → saved {ckpt}")


if __name__ == "__main__":
    train()
