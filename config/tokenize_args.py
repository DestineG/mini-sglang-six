from dataclasses import dataclass

from .common_args import CommonArgs


@dataclass(frozen=True)
class TokenizeArgs(CommonArgs):
    num_tokenizer: int = 1
    num_detokenizer: int = 0

    def __post_init__(self):
        assert self.num_tokenizer >= 1
        assert self.num_detokenizer >= 0

        if self.num_tokenizer > 1:
            assert self.num_detokenizer > 0, "Multiple tokenizers require at least one detokenizer"

    @property
    def is_shared_mode(self) -> bool:
        return self.num_tokenizer == 1 and self.num_detokenizer == 0