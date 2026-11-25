from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, ClassVar, List

from .utils import serialize_type, deserialize_type, register_message


@dataclass
class BaseMessage:
    message_type: ClassVar[str]

    @staticmethod
    def encoder(obj) -> Dict:
        return serialize_type(obj)

    @staticmethod
    def decoder(data: Dict) -> BaseMessage:
        return deserialize_type(data)

@register_message
@dataclass
class BaseBatchMessage:
    message_type: ClassVar[str] = "base-batch"
    data: List[BaseMessage]