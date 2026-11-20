from dataclasses import dataclass
from typing import ClassVar, List
import torch

from .base import BaseMessage
from .utils import register_message
from ..protocol.sampling_params import SamplingParams
from ..protocol.generate_params import GenerateParams


@register_message
@dataclass
class BackendMsg(BaseMessage):
    message_type: ClassVar[str] = "backend"
    uid: str
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    generate_params: GenerateParams

@register_message
@dataclass
class BackendBatchMsg(BaseMessage):
    message_type: ClassVar[str] = "backend-batch"
    data: List[BackendMsg]

@register_message
@dataclass
class BackendAbortMsg(BaseMessage):
    message_type: ClassVar[str] = "backend-abort"
    uid: str

@register_message
@dataclass
class BackendAbortBatchMsg(BaseMessage):
    message_type: ClassVar[str] = "backend-abort-batch"
    data: List[BackendAbortMsg]

@register_message
@dataclass
class BackendExitMsg(BaseMessage):
    message_type: ClassVar[str] = "backend-exit"
