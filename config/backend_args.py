from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List
import torch

from .common_args import CommonArgs
from ..distributed.info import DistributedInfo
from .model_configs import ModelConfig


if TYPE_CHECKING:
    from .model_configs import ModelConfig


@dataclass(frozen=True)
class BackendArgs(CommonArgs):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"

    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: int = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @cached_property
    def model_config(self) -> ModelConfig:
        from .model_configs import ModelConfig
        return ModelConfig.from_hf(self.model_path)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"