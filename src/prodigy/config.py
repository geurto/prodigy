from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 512       # Overwritten after the tokenizer is built
    seq_len: int = 512          # Context window in tokens (covers ~4–8 bars of drums)
    d_model: int = 256          # Embedding / attention dimension
    n_heads: int = 8            # Must divide d_model evenly
    n_layers: int = 6           # Number of stacked Transformer blocks
    d_ff: int = 1024            # Feed-forward inner dimension (4 × d_model is typical)
    dropout: float = 0.1


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/groove")
    checkpoint_dir: Path = Path("checkpoints")
    tokenizer_dir: Path = Path("tokenizer")  # directory for save_pretrained

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    warmup_steps: int = 500     # Steps over which LR linearly ramps up
    grad_clip: float = 1.0      # Max gradient norm (prevents exploding gradients)

    device: str = "cuda"        # Switch to "cpu" for smoke-testing without a GPU
    log_every: int = 50         # Print loss every N optimizer steps
    save_every: int = 5         # Save a checkpoint every N epochs
