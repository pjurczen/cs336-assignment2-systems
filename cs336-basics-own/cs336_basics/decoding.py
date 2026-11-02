import argparse
import json
import os
from typing import Optional

import torch

from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.checkpoint import upgrade_state_dict
from cs336_basics.run_config import RunConfig
from cs336_basics.softmax import softmax
from cs336_basics.transformer_lm import TransformerLM


def decoding(args: argparse.Namespace):
    with open(os.path.join(args.run_dir, "config.json"), "r") as f:
        model_config: RunConfig = RunConfig.model_validate(json.load(f))
    tokenizer = BpeTokenizer.from_dir(model_config.tokenizer_dir)
    model: torch.nn.Module = TransformerLM(
        vocab_size=model_config.model_vocab_size,
        context_length=model_config.model_context_length,
        num_layers=model_config.model_num_layers,
        d_model=model_config.model_d_model,
        num_heads=model_config.model_num_heads,
        d_ff=model_config.model_d_ff,
        eps=model_config.model_eps,
        theta=model_config.model_rope_theta,
        device=torch.device(args.device),
        dtype=torch.float32,
        tie_embeddings=model_config.model_tie_embeddings,
        qk_norm=model_config.model_qk_norm,
        logit_soft_cap=model_config.model_logit_soft_cap,
    )

    checkpoint_path = args.checkpoint if args.checkpoint is not None else latest_checkpoint_path(os.path.join(args.run_dir, "checkpoints"))
    print(f"Loading model from {checkpoint_path}...")
    # The run may have trained on another device (e.g. CUDA), so load straight onto the one we decode on.
    checkpoint: dict[str, any] = torch.load(checkpoint_path, map_location=args.device)
    # Older checkpoints: torch.compile key prefix (see upgrade_state_dict).
    upgrade_state_dict(checkpoint["model"])
    model.load_state_dict(checkpoint["model"])

    context: list[int] = []
    special_token_ids: set[int] = {tokenizer.id_vocab[token] for token in tokenizer.special_tokens}

    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        if len(prompt) == 0:
            continue
        encoded_prompt: list[int] = tokenizer.encode(prompt)
        context += encoded_prompt
        generated_tokens: int = 0
        last_token_id = None

        while (last_token_id is None or last_token_id not in special_token_ids) and generated_tokens != args.max_tokens:
            context_cutoff: int = len(context) - model_config.model_context_length
            context_start: int = 0 if context_cutoff < 0 else context_cutoff
            context_window: list[int] = context[context_start:]

            model_input: torch.Tensor = torch.tensor(context_window, device=torch.device(args.device)).unsqueeze(0)
            with torch.no_grad():
                logits: torch.Tensor = model(model_input)  # (batch, sequence_length, vocab_size)

                next_token_logits = logits[0][-1]  # first (only) batch, last token

                next_token_id: int = sample_next_token(next_token_logits, args.temperature, args.top_p, args.top_k)
                last_token_id = next_token_id
                context.append(next_token_id)
                generated_tokens += 1

                if next_token_id not in special_token_ids:
                    next_token: str = tokenizer.decode([next_token_id])
                    print(next_token, end="", flush=True)
        print()


def latest_checkpoint_path(checkpoints_dir: str) -> str:
    checkpoints = [os.path.join(checkpoints_dir, name) for name in os.listdir(checkpoints_dir) if name.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {checkpoints_dir}")
    return max(checkpoints, key=os.path.getmtime)


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: Optional[float] = None, top_k: Optional[int] = None) -> int:
    if temperature == 0:
        return torch.argmax(logits).item()
    scaled_prediction: torch.Tensor = logits / temperature
    probabilities = softmax(scaled_prediction, 0)
    values, indices = torch.sort(probabilities, descending=True)

    if top_k:
        values[top_k:] = 0

    if top_p:
        cumulative_probabilities = torch.cumsum(values, dim=0)
        difference = cumulative_probabilities - values
        mask = difference >= top_p
        values[mask] = 0

    index = torch.multinomial(values, num_samples=1)
    return indices[index].item()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--run_dir", type=str, required=True)
    arg_parser.add_argument("--checkpoint", type=str, default=None)
    arg_parser.add_argument("--temperature", type=float, default=1.0)
    arg_parser.add_argument("--top_p", type=float)
    arg_parser.add_argument("--top_k", type=int)
    arg_parser.add_argument("--max_tokens", type=int, default=1000)
    arg_parser.add_argument("--device", type=str, default="mps")
    args = arg_parser.parse_args()

    decoding(args)
