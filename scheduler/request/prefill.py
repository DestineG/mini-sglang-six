
from dataclasses import dataclass, field
from typing import List, Tuple, Any

import torch

from .base import BaseRequestManager
from ...protocol.request import Request, RequestStatus
from ..table import TableManager
from ..cache import CacheManager
from ...kvcache.base import BaseCacheHandle
from ...utils.misc import div_ceil


@dataclass
class PrefillManager(BaseRequestManager):
    table_manager: TableManager = field(default=None)
    cache_manager: CacheManager = field(default=None)

    def schedule(
        self,
        page_budget: int,
        max_schedule_len: int
    ) -> Tuple[List[Request], List[Request]]:
        ...

    def post_process(
        self,
        requests: List[Request],
        sampled_tokens: List[torch.Tensor],
        eos_token_id: int
    ) -> Tuple[List[Request], List[Request]]:
        ...