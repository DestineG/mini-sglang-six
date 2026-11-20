from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, ClassVar

from .utils import serialize_type, deserialize_type


@dataclass
class BaseMessage:
    message_type: ClassVar[str]

    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(data: Dict) -> BaseMessage:
        return deserialize_type(data)