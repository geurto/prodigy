"""
MIDI tokenization using miditok (REMI scheme).

miditok turns a MIDI file into a flat sequence of integer tokens — the same
idea as tokenizing text before feeding it to a language model.

REMI (Revamped MIDI) uses these event types:
  BAR       — marks the start of each measure
  POSITION  — beat/sub-beat offset within the bar (quantized grid)
  PITCH     — note number (for drums: the GM drum-map instrument, e.g. 36=kick)
  VELOCITY  — bucketed into N bins (32 here)
  DURATION  — how long the note lasts

All Groove files are 4/4 drum tracks, so we disable chords, programs, and
time-signature tokens to keep the vocabulary small (~300–500 tokens).
"""

from pathlib import Path

from miditok import REMI, TokenizerConfig


def build_tokenizer() -> REMI:
    """Create a fresh REMI tokenizer configured for drum-only 4/4 patterns."""
    config = TokenizerConfig(
        use_chords=False,
        use_rests=True,             # Capture silence between hits
        use_tempos=True,            # Embed BPM as a token — useful for conditioning later
        use_time_signatures=False,  # All Groove files are 4/4; no need to encode it
        use_programs=False,         # Single drum track, no program-change tokens
        num_velocities=32,          # 128 MIDI velocities → 32 buckets
        # Resolution: 8 sub-beats per beat for the first 4 beats, 4 for beats 5-8.
        # This gives 32nd-note precision, which is fine-grained enough for drums.
        beat_res={(0, 4): 8, (4, 8): 4},
    )
    return REMI(config)


def load_tokenizer(tokenizer_dir: Path) -> REMI:
    """Load a previously saved tokenizer from disk."""
    return REMI.from_pretrained(str(tokenizer_dir))


def save_tokenizer(tokenizer: REMI, tokenizer_dir: Path) -> None:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))
