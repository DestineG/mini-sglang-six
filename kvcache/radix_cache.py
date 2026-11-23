from __future__ import annotations

import time
from typing import Dict, Any, Callable
import torch


class RadixTreeNode:
    counter: int = 0

    def __init__(
        self, 
        key_fn: Callable[[torch.Tensor], Any],
        tic: int | None = None
    ):
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1

        self.key_fn = key_fn
        self.timestamp = tic if tic is not None else time.monotonic_ns()
        self.ref_counter: int = 0
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None

        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    @property
    def length(self) -> int:
        return self._length

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from ..kernel.radix import fast_compare_key

        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        ...

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp