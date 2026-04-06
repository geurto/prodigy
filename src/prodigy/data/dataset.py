"""
PyTorch Dataset for tokenized drum MIDI sequences.

The training objective is next-token prediction: given tokens [0 … T-1],
predict tokens [1 … T]. This is identical to how GPT is trained on text.

To build many training samples from relatively few MIDI files we use a
sliding window over the concatenated token stream. A 50% overlap (stride =
seq_len // 2) roughly doubles the number of samples at the cost of correlation
between adjacent windows — acceptable for this dataset size.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from miditok import REMI
from miditok.classes import TokSequence
from symusic import Score


def collect_midi_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.mid"))


def tokenize_files(tokenizer: REMI, paths: list[Path]) -> list[list[int]]:
    """
    Tokenize each MIDI file and return a list of integer-id sequences.

    Files that fail to parse (corrupted, unsupported format) are skipped with
    a warning rather than crashing the whole pipeline.
    """
    sequences: list[list[int]] = []
    for p in paths:
        try:
            score = Score(str(p))
            result = tokenizer(score)
            # tokenizer() returns a list of TokSequence (one per track)
            tok_seq: TokSequence = result[0] if isinstance(result, list) else result
            if len(tok_seq.ids) > 1:
                sequences.append(tok_seq.ids)
        except Exception as exc:
            print(f"  skipping {p.name}: {exc}")
    return sequences


class DrumDataset(Dataset):
    """
    Sliding-window dataset over a concatenated token stream.

    Args:
        sequences: List of token-id lists, one per MIDI file.
        seq_len:   Number of tokens per training sample.
        stride:    Step size for the sliding window. Defaults to seq_len // 2.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        seq_len: int,
        stride: int | None = None,
    ) -> None:
        self.seq_len = seq_len
        stride = stride or seq_len // 2

        # Concatenate all token sequences into a single flat stream.
        # A more sophisticated approach would insert special BOS/EOS tokens at
        # file boundaries to let the model learn where pieces start and end.
        flat: list[int] = []
        for seq in sequences:
            flat.extend(seq)

        # Each window is seq_len + 1 tokens: input = [0:seq_len], target = [1:seq_len+1]
        self.windows: list[list[int]] = [
            flat[i : i + seq_len + 1]
            for i in range(0, len(flat) - seq_len, stride)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.tensor(self.windows[idx], dtype=torch.long)
        return w[:-1], w[1:]  # (input, target) — target is input shifted left by 1
