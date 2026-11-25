from __future__ import annotations

from dataclasses import dataclass
import time
import heapq
from typing import Dict, Any, Callable, Tuple, List
import torch

from .base import (
    BaseCacheHandle,
    BasePrefixCache,
    InsertResult,
    MatchResult,
    SizeInfo
)

from ..protocol.context import get_global_ctx
from ..utils.misc import align_down

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
        self.ref_count: int = 0
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
        return len(self.children) == 0

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
        assert 0 < pos < self.length

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(self.parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp

@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_prefix_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []

        while not node.is_root():
            value_list.append(node.value)
            node = node.parent

        if not value_list:
            return node.value[:0]

        value_list.reverse()
        return torch.cat(value_list)

def _get_key_fn(page_size: int) -> Callable[[torch.Tensor], Any]:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())

class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.set_key_value(torch.empty(0, dtype=torch.int32, device="cpu"), self.empty_tensor)
        self.root_node.ref_count = 1

    def lock_handle(
        self,
        handle: RadixCacheHandle,
        unlock: bool = False,
        end_handle: RadixCacheHandle | None = None
    ):
        # 操作包含起点，不包含终点；默认到根节点为止
        end_node = self.root_node if end_handle is None else end_handle.node
        check_node = handle.node
        while check_node is not end_node:
            if check_node.is_root():
                raise ValueError("End handle is not an ancestor of the start handle")
            check_node = check_node.parent

        node = handle.node
        if unlock:
            while node is not end_node:
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while node is not end_node:
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int, int]:
        prefix_len = 0
        matched_evictable_slots = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            # 不存在被命中的孩子节点
            if child_node is None:
                return node, prefix_len, matched_evictable_slots

            # 更新被命中节点
            node = child_node

            # 获取命中长度(对齐 page_size)
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # 统计本次匹配到的可淘汰 Slot，不修改引用计数
            if node.ref_count == 0:
                matched_evictable_slots += match_len

            # 命中未完全覆盖 node，则切分 node，搜索就此结束
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len, matched_evictable_slots

            # 命中完全覆盖 node，则继续向后搜索
            node.timestamp = tic

        return node, prefix_len, matched_evictable_slots


    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[MatchResult, int]:
        node, prefix_len, matched_evictable_slots = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node)), matched_evictable_slots

    def insert_prefix(
        self,
        input_ids: torch.Tensor,
        indices: torch.Tensor
    ) -> InsertResult:
        # 插入长度对齐 page_size
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        
        # 查找已缓存的前缀
        node, prefix_len, _ = self._tree_walk(input_ids)

        # 处理不存在前缀缓存的部分
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:].clone(), indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node

        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()

            # 收集无引用的的叶子节点
            if node.is_leaf() and node.ref_count == 0:
                leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor

        # 检查是否有足够slot
        assert size <= self.evictable_size, f"Cannot evict {size}, only {self.evictable_size} is evictable"

        # 仅收集叶子中的可驱逐节点
        leave_nodes = self._collect_leave_nodes_for_evict()
        # 按照访问时间建立 heap
        heapq.heapify(leave_nodes)

        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0
        while evicted_size < size:
            assert leave_nodes, f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            # 驱逐 node 节点
            node = heapq.heappop(leave_nodes)
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]

            # 若父节点为子节点并且引用为零也加入可驱逐列表
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    def check_integrity(self) -> None:
        pass
