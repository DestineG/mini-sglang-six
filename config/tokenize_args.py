from dataclasses import dataclass

from .common_args import CommonArgs


@dataclass(frozen=True)
class TokenizeArgs(CommonArgs):
    num_tokenizer: int = 1
    num_detokenizer: int = 0

    def __post_init__(self):
        if self.num_tokenizer < 1:
            raise ValueError("num_tokenizer must be >= 1")

        if self.num_detokenizer < 0:
            raise ValueError("num_detokenizer must be >= 0")

        if self.num_tokenizer > 1 and self.num_detokenizer == 0:
            raise ValueError("Multiple tokenizers require at least one detokenizer")

    @property
    def is_shared_mode(self) -> bool:
        return self.num_tokenizer == 1 and self.num_detokenizer == 0

    @property
    def effective_detokenizer_addr(self) -> str:
        if self.is_shared_mode:
            return self.tokenizer_addr
        return self.detokenizer_addr