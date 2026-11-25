from dataclasses import dataclass
from typing import ClassVar, List, Dict

from .base import BaseMessage, BaseBatchMessage
from .utils import register_message
from ..protocol.sampling_params import SamplingParams
from ..protocol.generate_params import GenerateParams


@register_message
@dataclass
class TokenizeMsg(BaseMessage):
    message_type: ClassVar[str] = "tokenize"
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams
    generate_params: GenerateParams

@register_message
@dataclass
class TokenizeBatchMsg(BaseBatchMessage):
    message_type: ClassVar[str] = "tokenize-batch"
    data: List[TokenizeMsg]

@register_message
@dataclass
class DeTokenizeMsg(BaseMessage):
    message_type: ClassVar[str] = "detokenize"
    uid: int
    next_token: int
    finished: bool

@register_message
@dataclass
class DeTokenizeBatchMsg(BaseBatchMessage):
    message_type: ClassVar[str] = "detokenize-batch"
    data: List[DeTokenizeMsg]

@register_message
@dataclass
class TokenizeAbortMsg(BaseMessage):
    message_type: ClassVar[str] = "tokenize-abort"
    uid: int

@register_message
@dataclass
class TokenizeAbortBatchMsg(BaseBatchMessage):
    message_type: ClassVar[str] = "tokenize-abort-batch"
    data: List[TokenizeAbortMsg]
