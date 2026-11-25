from dataclasses import dataclass, field
from typing import List
from enum import Enum
import torch

from .sampling_params import SamplingParams
from .generate_params import GenerateParams
from ..kvcache.base import BaseCacheHandle


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    CHUNKEDPREFILL = "chunkedprefill"
    DECODE = "decode"
    FINISHED = "finished"
    ABORTED = "aborted"

@dataclass(eq=False)
class Request:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    generate_params: GenerateParams
    table_idx: int
    prompt_len: int
    cached_len: int
    output_len: int
    schedule_len: int

    cache_handle: BaseCacheHandle | None = field(default=None, repr=False)
    status: RequestStatus = RequestStatus.WAITING
    is_forwarding: bool = False

    @property
    def seq_len(self) -> int:
        return self.input_ids.size(0)
