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

    def add(self, requests: List[Request]):
        for req in requests:
            # 检查待添加的 UID 是否已存在
            if req.uid in self.uid_map:
                raise ValueError(f"Duplicate request uid: {req.uid}")

            # 更新 List
            self.requests.append(req)
            
            # 更新 UID 索引
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
