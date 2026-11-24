from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    @abstractmethod
    def k_cache(self, layer_idx: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, layer_idx: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_idx: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_prefix_indices(self) -> torch.Tensor: ...

class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # TODO: support HiCache

class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix

class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(
        self,
        handle: BaseCacheHandle,
        unlock: bool = False
    ): ...

    @abstractmethod
    def match_prefix(
        self,
        input_ids: torch.Tensor
    ) -> MatchResult: ...

    @abstractmethod
    def insert_prefix(
        self,
        input_ids: torch.Tensor,
        indices: torch.Tensor
    ) -> InsertResult: ...

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor: ...

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo: ...

    @abstractmethod
    def check_integrity(self) -> None: ...
