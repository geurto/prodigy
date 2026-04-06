"""
Generate a drum pattern from a trained DrumTransformer checkpoint.

Usage:
    python -m prodigy.generate \\
        --checkpoint checkpoints/epoch_050.pt \\
        --out generated.mid \\
        --n_tokens 512 \\
        --temperature 1.0 \\
        --top_k 50

The output is a standard MIDI file you can open in any DAW or play with a
software drum sampler.

Sampling parameters:
  --temperature  Controls randomness. <1 = more conservative/repetitive,
                 >1 = more surprising/chaotic. Start around 0.9–1.1.
  --top_k        At each step only the top-k most likely tokens are considered.
                 Lower = safer but less varied. 40–100 is a reasonable range.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from miditok.classes import TokSequence

from prodigy.config import ModelConfig
from prodigy.data.tokenizer import load_tokenizer
from prodigy.model.transformer import DrumTransformer


@torch.no_grad()
def sample(
    model: DrumTransformer,
    prompt: torch.Tensor,       # (1, T) seed token ids
    n_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[int]:
    """
    Autoregressively generate n_new_tokens from the prompt.

    At each step:
      1. Feed the current sequence through the model → logits over vocabulary.
      2. Take only the last position's logits (the prediction for the next token).
      3. Apply temperature scaling and top-k filtering.
      4. Sample one token from the resulting distribution.
      5. Append and repeat.
    """
    model.eval()
    seq = prompt.clone()

    for _ in range(n_new_tokens):
        # Crop to the model's context window if the sequence grows longer
        context = seq[:, -model.config.seq_len:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature         # (1, vocab_size)

        # Top-k: zero out everything below the k-th largest logit
        if top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)    # (1, 1)
        seq = torch.cat([seq, next_token], dim=1)

    return seq[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a drum MIDI file")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer_dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--out", type=Path, default=Path("generated.mid"))
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg: ModelConfig = ckpt["model_cfg"]
    model = DrumTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded model ({model.num_parameters():,} params) from {args.checkpoint}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)

    # Seed the generation with a BAR token so the model starts at a bar boundary.
    # "BAR_None" is the REMI token that marks the beginning of a measure.
    bar_id = tokenizer.vocab.get("BAR_None", 0)
    prompt = torch.tensor([[bar_id]], dtype=torch.long, device=device)

    print(f"Generating {args.n_tokens} tokens (temperature={args.temperature}, top_k={args.top_k})…")
    token_ids = sample(model, prompt, args.n_tokens, args.temperature, args.top_k)

    # Decode token ids back to a MIDI file via miditok
    tok_seq = TokSequence(ids=token_ids)
    tok_seq.ids_to_tokens(tokenizer.vocab)          # populate .tokens from .ids
    score = tokenizer.tokens_to_score([tok_seq])    # miditok → symusic Score
    score.dump_midi(str(args.out))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
