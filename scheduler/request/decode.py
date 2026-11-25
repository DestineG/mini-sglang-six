from dataclasses import dataclass, field
from typing import List, Any
import torch

from .base import BaseRequestManager
from ...protocol.request import Request, RequestStatus
from ..cache import CacheManager
from ...utils.misc import div_ceil


@dataclass
class DecodeManager(BaseRequestManager):
    cache_manager: CacheManager = field(default=None)
    max_batch_size: int = 32

    def schedule(self, page_budget: int) -> List[Request]:
        ...

    def post_process(
        self,
        requests: List[Request],
        sampled_tokens: List[torch.Tensor],
        eos_token_id: int
    ) -> List[Request]:
        ...