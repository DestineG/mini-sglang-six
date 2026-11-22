import torch

from ..distributed.info import get_tp_info
from ..utils.misc import div_even
from .base import BaseKVCachePool


class PagedKVCache(BaseKVCachePool):
    def __init__(
        self,
        num_layers: int
    ):
        ...