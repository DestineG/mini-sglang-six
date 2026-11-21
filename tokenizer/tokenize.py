from __future__ import annotations

from typing import List

import torch
from ..message.tokenize import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        prompts = []

        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text

            prompts.append(prompt)

        # Batch tokenization
        encoded = self.tokenizer(
            prompts,
            padding=False,
            truncation=False,
            return_tensors=None,
        )

        return [
            torch.tensor(ids, dtype=torch.int32)
            for ids in encoded["input_ids"]
        ]