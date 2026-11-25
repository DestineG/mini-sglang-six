
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
        reqs: List[Request] = []
        decode_reqs: List[Request] = []

        # 收集前缀命中页表映射
        prefix_indices: List[torch.Tensor] = []
        prefix_info: List[Tuple[int, int, int]] = []

        for req in self.requests:
            if req.is_forwarding:
                continue

            if page_budget == 0:
                break
            
            # 处理 Prefill
            if req.status == RequestStatus.PREFILL:
                match_result, match_evictable_slots = self.cache_manager.match_req(req)
                cached_handle = match_result.cuda_handle
                req_max_schedule_len = req.prompt_len - cached_handle.cached_len

                assert match_evictable_slots % self.cache_manager.page_size == 0
                assert cached_handle.cached_len % self.cache_manager.page_size == 0

                # 送入 Decode
                if req_max_schedule_len == 1:
                    # 预算不足就跳过，预算刚好足够时继续执行
                    if div_ceil(match_evictable_slots, self.cache_manager.page_size) > page_budget:
                        continue

                    req.cached_len = cached_handle.cached_len
                    req.schedule_len = 1
                    req.cache_handle = cached_handle
                    req.status = RequestStatus.DECODE
                    self.cache_manager.lock(req.cache_handle)

                    if cached_handle.cached_len > 0:
                        prefix_indices.append(cached_handle.get_prefix_indices())
                        prefix_info.append((
                            req.table_idx,
                            0,
                            cached_handle.cached_len // self.cache_manager.page_size
                        ))

                    decode_reqs.append(req)
                    page_budget -= div_ceil(match_evictable_slots, self.cache_manager.page_size)
                    self.remove([req])
                    continue

                schedule_len = min(max_schedule_len, req_max_schedule_len)

                # 仅仅 evictable 部分就已经超预算那就暂不纳入
                # 极限情况：超长 evictable 前缀命中会出不去
                if div_ceil(match_evictable_slots, self.cache_manager.page_size) >= page_budget:
                    continue

                # 不考虑预算的情况下需要的 page 数量
                needed_pages = div_ceil(
                    match_evictable_slots + schedule_len,
                    self.cache_manager.page_size
                )

                # 超支则截断
                if needed_pages > page_budget:
                    schedule_len = page_budget * self.cache_manager.page_size - match_evictable_slots
                    needed_pages = page_budget

                req.cached_len = cached_handle.cached_len
                req.schedule_len = schedule_len
                self.cache_manager.lock(cached_handle)
                req.cache_handle = cached_handle

                if cached_handle.cached_len > 0:
                    prefix_indices.append(cached_handle.get_prefix_indices())
                    prefix_info.append((
                        req.table_idx,
                        0,
                        cached_handle.cached_len // self.cache_manager.page_size
                    ))

                reqs.append(req)
                page_budget -= needed_pages

            elif req.status == RequestStatus.CHUNKEDPREFILL:
                req_max_schedule_len = req.prompt_len - req.cached_len
                schedule_len = min(max_schedule_len, req_max_schedule_len)

                needed_pages = (
                    div_ceil(req.cached_len + schedule_len, self.cache_manager.page_size)
                    - div_ceil(req.cached_len, self.cache_manager.page_size)
                )

                # 超支则截断
                if needed_pages > page_budget:
                    # 已缓存长度恰好页对齐时，已有页全部用满，剩余 Slot 为 0。
                    # 再取模避免把这种情况误算成还有一整页可用。
                    remaining_slots = (
                        self.cache_manager.page_size
                        - req.cached_len % self.cache_manager.page_size
                    ) % self.cache_manager.page_size

                    schedule_len = page_budget * self.cache_manager.page_size + remaining_slots
                    needed_pages = page_budget

                req.schedule_len = schedule_len
                reqs.append(req)
                page_budget -= needed_pages

            else:
                raise ValueError

        # 批量写入命中前缀的映射，不重复分配已有物理页
        if prefix_indices:
            self.cache_manager.write_page_table(
                allocated=torch.cat(prefix_indices),
                allocation_info=prefix_info
            )

        # 统一分配本轮 Prefill 新页，转入 Decode 的请求不在这里分配
        self.cache_manager.allocate_paged(requests=reqs)

        for req in reqs:
            req.is_forwarding = True

        return reqs, decode_reqs


    def post_process(
        self,
        requests: List[Request],
        sampled_tokens: List[torch.Tensor],
        eos_token_id: int
    ) -> Tuple[List[Request], List[Request]]:
        ...
