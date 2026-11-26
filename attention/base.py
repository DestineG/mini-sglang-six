from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List
import torch


if TYPE_CHECKING:
    import torch
    from ..protocol.request import Batch


@dataclass
class BaseCaptureData:
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    page_table: torch.Tensor
    kv_slot_indices: torch.Tensor

    @classmethod
    def create(
        cls,
        max_bs: int,
        max_schedule_len: int,
        max_pages_per_seq: int,
        device: torch.device,
        **kwargs
    ):
        return cls(
            positions=torch.zeros(
                (max_bs * max_schedule_len,),
                dtype=torch.int32,
                device=device
            ),
            cu_seqlens_q=torch.arange(
                max_bs + 1,
                dtype=torch.int32,
                device=device
            ),
            cu_seqlens_k=torch.arange(
                max_bs + 1,
                dtype=torch.int32,
                device=device
            ),
            page_table=torch.zeros(
                (max_bs, max_pages_per_seq),
                dtype=torch.int32,
                device=device
            ),
            kv_slot_indices=torch.zeros(
                (max_bs * max_schedule_len,),
                dtype=torch.int32,
                device=device
            ),
            **kwargs,
        )

@dataclass
class BaseAttnMetadata(ABC):
    ...

class BaseAttnBackend(ABC):
    @abstractmethod
    def prepare_metadata(self, batch: Batch):
        raise NotImplemented

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        batch: Batch
    ) -> torch.Tensor:
        raise NotImplemented

    @abstractmethod
    def init_capture_graph(
        self,
        max_seq_len: int,
        bs_list: List[int]
    ):
        raise NotImplemented

    @abstractmethod
    def prepare_capture(self, batch: Batch):
        raise NotImplemented

    @abstractmethod
    def prepare_reply(self, batch: Batch):
        raise NotImplemented