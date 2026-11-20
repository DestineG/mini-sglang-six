from dataclasses import dataclass
from typing import ClassVar, List

from .base import BaseMessage
from .utils import register_message


@register_message
@dataclass
class FrontendMsg(BaseMessage):
    message_type: ClassVar[str] = "frontend"
    uid: str
    incremental_output: str
    finished: bool

@register_message
@dataclass
class FrontendBatchMsg(BaseMessage):
    message_type: ClassVar[str] = "frontend-batch"
    data: List[FrontendMsg]
