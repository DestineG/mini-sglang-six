from dataclasses import dataclass

import torch

from ..attention.base import BaseAttnBackend
from ..kvcache.base import BaseKVCachePool
# from ..moe.base import BaseMoeBackend
from .request import Batch


@dataclass
class Context:
    num_pages: int
    page_size: int
    page_table: torch.Tensor
    kv_cache: BaseKVCachePool
    attn_backend: BaseAttnBackend
    # moe_backend: BaseMoeBackend | None = None
    _batch: Batch

_GLOBAL_CTX: Context | None = None

def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx

def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
