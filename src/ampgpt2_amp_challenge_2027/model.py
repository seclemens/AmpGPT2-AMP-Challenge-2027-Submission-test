"""
Local AmpGPT2 model loading and autoregressive peptide generation.

The model is loaded exclusively from the bundled checkpoint and supports CPU
or CUDA generation.
"""
from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm


def clean_sequence(text: str) -> str:
    """
    clean ampgpt2 output
    """
    text = text.replace("<|endoftext|>", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    text = text.replace(" ", "")
    text = text.upper()

    return "".join(text)


class AmpGPT2Generator:
    def __init__(
        self,
        checkpoint: str = "checkpoint/AmpGPT2",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def generate(
        self,
        n: int,
        *,
        batch_size: int = 64,
        max_new_tokens: int = 100,
    ) -> list[str]:
        sequences: list[str] = []

        prompt = self.tokenizer.eos_token or "<|endoftext|>"

        progress = tqdm(
            total=n,
            desc="Generating AmpGPT2",
            unit="seq",
            dynamic_ncols=True,
        )

        while len(sequences) < n:
            current_batch = min(batch_size, n - len(sequences))

            encoded = self.tokenizer(
                [prompt] * current_batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model.generate(
                **encoded,
                do_sample=True,

                repetition_penalty=1.2,

                max_new_tokens=max_new_tokens,

                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            decoded = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )

            batch_sequences = [
                clean_sequence(x)
                for x in decoded
            ]

            sequences.extend(batch_sequences)

            progress.update(len(batch_sequences))

        return sequences[:n]