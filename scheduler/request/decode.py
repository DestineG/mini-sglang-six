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

    def schedule(
        self,
        page_budget: int,
        max_batch_size: int = 32
    ) -> List[Request]:
        reqs: List[Request] = []

        for req in self.requests:
            # 达到调度上限
            if len(reqs) >= max_batch_size:
                break

            # 跳过正在调度的请求
            if req.is_forwarding:
                continue

            # 状态校验
            if req.status != RequestStatus.DECODE:
                raise ValueError

            # 调度长度校验
            schedule_len = req.seq_len - req.cached_len
            if schedule_len != 1:
                raise ValueError

            # 请求页数计算
            needed_pages = (
                div_ceil(req.cached_len + schedule_len, self.cache_manager.page_size)
                - div_ceil(req.cached_len, self.cache_manager.page_size)
            )

            # 页数不足则跳过
            if needed_pages > page_budget:
                continue

            req.schedule_len = schedule_len
            reqs.append(req)
            page_budget -= needed_pages

        self.cache_manager.allocate_paged(requests=reqs)
        for req in reqs:
            req.is_forwarding = True

        return reqs

    def post_process(
        self,
        requests: List[Request],
        sampled_tokens: List[torch.Tensor],
        eos_token_id: int
    ) -> List[Request]:
        finished_reqs: List[Request] = []
        assert len(requests) == len(sampled_tokens)
        for req, token_id in zip(requests, sampled_tokens):
            assert req.is_forwarding

            new_cached_len = req.cached_len + req.schedule_len
            remain_schedule_len = req.seq_len - new_cached_len

            if remain_schedule_len == 0:
                req.is_forwarding = False
                assert req.status == RequestStatus.DECODE
                req.cached_len = new_cached_len
                req.schedule_len = 1
                req.output_len += 1
                token_id = token_id.reshape(1).to(req.input_ids)
                req.input_ids = torch.cat((req.input_ids, token_id))

                finished = (
                    req.output_len >= req.generate_params.max_tokens
                    or (
                        not req.generate_params.ignore_eos
                        and token_id.item() == eos_token_id
                    )
                )
                if finished:
                    req.status = RequestStatus.FINISHED
                    finished_reqs.append(req)

            else:
                raise ValueError

        # 移除已完成请求
        self.remove(finished_reqs)

        return finished_reqs
