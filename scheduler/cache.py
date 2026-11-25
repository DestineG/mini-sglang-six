from typing import List, Tuple
import torch

from contextlib import contextmanager

from ..kvcache.radix_cache import RadixPrefixCache
from ..protocol.request import Request
from ..kvcache.base import BaseCacheHandle, MatchResult
from ..utils.misc import div_ceil

class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor
    ):
        self.device = page_table.device
        self.free_slots = (
            torch.arange(num_pages, dtype=torch.int32, device=self.device)
            * page_size
        )
        self.prefix_cache = RadixPrefixCache(device=self.device)
        self.num_pages = num_pages
        self.page_size = page_size
        self.page_table = page_table

    def match_req(self, request: Request) -> MatchResult:
        input_len = len(request.input_ids)
        assert input_len > 0, "Input length must be greater than 0."
        # 最后一个 token 未填充，不必参加匹配
        return self.prefix_cache.match_prefix(request.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return (
            len(self.free_slots) * self.page_size +
            self.prefix_cache.size_info.evictable_size
        )

    @property
    def available_pages(self) -> int:
        return self.available_size // self.page_size

    def lock(self, handle: BaseCacheHandle):
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle):
        self.prefix_cache.lock_handle(handle, unlock=True)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict(
                (needed_pages - free_pages) * self.page_size
            )
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            self.page_size,
            device=self.device,
            dtype=torch.int32
        )
        return (pages.unsqueeze(1) + positions.unsqueeze(0)).flatten()

    def write_page_table(
        self,
        allocated: torch.Tensor,
        allocation_info: List[Tuple[int, int, int]],
    ):
        needed_tokens = len(allocated)
        table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
        positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)

        offset = 0
        for table_idx, first_page, last_page in allocation_info:
            first_pos, last_pos = first_page * self.page_size, last_page * self.page_size
            length = last_pos - first_pos

            # 填充 page_table 行坐标
            table_idx_host[offset : offset + length].fill_(table_idx)

            # 填充 page_table 列坐标
            torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
            offset += length

        # 校验
        assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."

        # 非阻塞搬运
        table_idxs = table_idx_host.to(self.page_table.device, non_blocking=True)
        positions = positions_host.to(self.page_table.device, non_blocking=True)

        # page_table 填充
        self.page_table[table_idxs, positions] = allocated

    def allocate_paged(self, requests: List[Request]):
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []

        for req in requests:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(
                req.cached_len + req.schedule_len,
                self.page_size
            )

            needed_page = last_page - first_page

            if needed_page > 0:
                needed_pages += needed_page
                allocation_info.append(
                    (req.table_idx, first_page, last_page)
                )

        if needed_pages > 0:
            allocated = self._page_to_token(
                self._allocate(needed_pages=needed_pages)
            )
            self.write_page_table(
                allocated=allocated,
                allocation_info=allocation_info,
            )

    def _free(self, indices: torch.Tensor):
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def cache_request(
        self,
        request: Request,
        finished: bool
    ):
        """
        将 request 中的 kvcache 更新到 prefix tree
        """
        # 获取可插入长度
        insert_ids = request.input_ids[: request.cached_len]
        page_indices = self.page_table[request.table_idx, : request.cached_len]

        # 释放旧的引用
        old_handle = request.cache_handle
        self.unlock(old_handle)

        # 更新 prefix tree
        cached_len, new_handle = self.prefix_cache.insert_prefix(
            insert_ids,
            page_indices
        )

        # 释放该次命中的 page slot
        self._free(page_indices[old_handle.cached_len : cached_len])

        # 请求已完成则释放尾部非对齐 slot，否则对缓存重新上锁
        if finished:
            self._free(page_indices[new_handle.cached_len:])
        else:
            request.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()

        cache_pages = (
            self.prefix_cache.size_info.total_size // self.page_size
        )

        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError("CacheManager integrity check failed")

        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free(self):
        def lazy_free(indices: torch.Tensor):
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)
