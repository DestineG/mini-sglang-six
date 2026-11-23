from dataclasses import dataclass, field
from enum import Enum
import torch

from .sampling_params import SamplingParams
from .generate_params import GenerateParams
from ..kvcache.base import BaseCacheHandle


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
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
    _cached_len: int
    _output_len: int
    _schedule_len: int

    cache_handle: BaseCacheHandle | None = field(default=None, repr=False)
    status: RequestStatus = RequestStatus.WAITING

    def __post_init__(self) -> None:
        assert self.input_ids.ndim == 1
        assert self.input_ids.device.type == "cpu"
        # self.validate()

    def validate(self) -> None:
        assert 0 <= self.cached_len <= self.seq_len
        assert self.output_len >= 0
        assert 0 <= self.schedule_len <= self.seq_len - self.cached_len

    @property
    def seq_len(self) -> int:
        return self.input_ids.size(0)

    @property
    def cached_len(self) -> int:
        return self._cached_len

    @cached_len.setter
    def cached_len(self, value: int) -> None:
        self._cached_len = value
        # self.validate()

    @property
    def output_len(self) -> int:
        return self._output_len

    @output_len.setter
    def output_len(self, value: int) -> None:
        self._output_len = value
        self.validate()

    @property
    def schedule_len(self) -> int:
        return self._schedule_len

    @schedule_len.setter
    def schedule_len(self, value: int) -> None:
        self._schedule_len = value
        self.validate()