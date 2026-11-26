from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any

from ...protocol.request import Request


@dataclass
class BaseRequestManager(ABC):
    requests: List[Request] = field(default_factory=list, repr=False)
    uid_map: Dict[Any, Request] = field(default_factory=dict, repr=False)

    @property
    def empty(self):
        return len(self.requests) == 0

    def add(self, requests: List[Request], insert_idx: int = -1):
        # 检查待添加的 UID 是否重复
        new_uids = [req.uid for req in requests]

        if len(new_uids) != len(set(new_uids)):
            raise ValueError("Duplicate uid in input requests")

        duplicate_uids = set(new_uids) & self.uid_map.keys()
        if duplicate_uids:
            raise ValueError(f"Duplicate request uid: {duplicate_uids}")

        # 默认追加到末尾
        if insert_idx == -1:
            insert_idx = len(self.requests)

        if not 0 <= insert_idx <= len(self.requests):
            raise IndexError(f"Invalid insert index: {insert_idx}")

        # 更新 List
        self.requests[insert_idx:insert_idx] = requests

        # 更新 UID 索引
        for req in requests:
            self.uid_map[req.uid] = req

    def remove(self, requests: List[Request]):
        remove_uids = {req.uid for req in requests}

        # 检查待删除的 UID 是否全部存在
        missing_uids = remove_uids - self.uid_map.keys()
        if missing_uids:
            raise KeyError(f"Requests not found: {missing_uids}")

        # 更新 List
        self.requests = [
            req for req in self.requests
            if req.uid not in remove_uids
        ]

        # 更新 UID 索引
        for uid in remove_uids:
            del self.uid_map[uid]
