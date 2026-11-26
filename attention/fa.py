from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple
import torch

from ..protocol.context import get_global_ctx
from .base import (
    BaseAttnMetadata,
    BaseAttnBackend,
    BaseCaptureData
)


if TYPE_CHECKING:
    import torch
    from ..protocol.request import Batch
    from ..config.model_configs import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    pass

@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    kv_slot_indices: torch.Tensor

    page_table: torch.Tensor

class FABackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        self.config = config
        ctx = get_global_ctx()
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.page_table = ctx.page_table
        self.capture: FACaptureData | None = None
        self.scale = config.head_dim**(-0.5)
        # flash-attn 的分页 KV Cache 要求每页为 256 的倍数
        if self.page_size % 256 != 0:
            raise ValueError("FlashAttention requires page_size to be a multiple of 256")

    def prepare_metadata(self, batch: Batch):
        reqs = batch.requests

        seqlens_q = [req.schedule_len for req in reqs]
        seqlens_k = [req.cached_len + req.schedule_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        max_seqlen_k = max(seqlens_k)
        
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        device = self.kvcache.device

        cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_q = cu_seqlens_q.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        new_page_table = torch.stack(
            [self.page_table[req.table_idx, : max_seqlen_k : self.page_size]
             for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")

        kv_slot_indices = torch.cat([
            self.page_table[req.table_idx, seq_len - schedule_len:seq_len]
            for req, schedule_len, seq_len
            in zip(reqs, seqlens_q, seqlens_k)
        ])

        batch.attn_metadata = FAMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=new_page_table,
            kv_slot_indices=kv_slot_indices
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        batch: Batch
    ):
        metadata: FAMetadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(
            k=k,
            v=v,
            out_loc=metadata.kv_slot_indices,
            layer_idx=layer_idx
        )
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_idx),
            v_cache=self.kvcache.v_cache(layer_idx),
            page_table=metadata.page_table,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            softmax_scale=self.scale,
        )

    def init_capture_graph(
        self,
        max_seq_len: int,
        bs_list: List[int]
    ):
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        max_pages_per_seq = max_seq_len // self.page_size
        # max_schedule_len = 1: 仅支持 decode cuda graph
        capture = FACaptureData.create(
            max_bs=max_bs,
            max_schedule_len=1,
            max_pages_per_seq=max_pages_per_seq,
            device=self.kvcache.device
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_capture(self, batch: Batch):
        assert (bs := batch.size) in self.capture_bs and self.capture
        reqs = batch.requests

        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            max_seqlen_q=1,  # decode only
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            page_table=capture.page_table[:bs, :],
            kv_slot_indices=capture.kv_slot_indices[:bs]
        )
        batch.attn_metadata = metadata

    def prepare_reply(self, batch: Batch):
        metadata = batch.attn_metadata
        bs = batch.size

        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs

        capture = self.capture
        table_len = metadata.page_table.size(1)
        num_tokens = metadata.kv_slot_indices.numel()

        assert table_len <= capture.page_table.size(1)
        assert num_tokens <= capture.kv_slot_indices.numel()

        # 更新本轮 Q、K 的请求边界
        capture.cu_seqlens_q[:bs + 1].copy_(metadata.cu_seqlens_q)
        capture.cu_seqlens_k[:bs + 1].copy_(metadata.cu_seqlens_k)

        # 更新本轮 KV 写入的物理 Slot
        capture.kv_slot_indices[:num_tokens].copy_(
            metadata.kv_slot_indices
        )

        # 清理旧映射，再写入本轮 Page Table
        capture.page_table[:bs].zero_()
        capture.page_table[:bs, :table_len].copy_(
            metadata.page_table
        )


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    causal: bool = True,
) -> torch.Tensor:
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as e:
        raise ImportError(
            "flash_attn is not found. Please install a compatible flash-attn build."
        ) from e

    return flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=page_table,
        softmax_scale=softmax_scale,
        window_size=window_size,
        softcap=softcap,
        causal=causal,
    )
