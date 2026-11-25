from dataclasses import dataclass
from typing import ClassVar, List
import torch

from .base import BaseMessage, BaseBatchMessage
from .utils import register_message
from ..protocol.sampling_params import SamplingParams
from ..protocol.generate_params import GenerateParams


@register_message
@dataclass
class BackendMsg(BaseMessage):
    message_type: ClassVar[str] = "backend"
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    generate_params: GenerateParams

@register_message
@dataclass
class BackendBatchMsg(BaseBatchMessage):
    message_type: ClassVar[str] = "backend-batch"
    data: List[BackendMsg]

@register_message
@dataclass
class BackendAbortMsg(BaseMessage):
    message_type: ClassVar[str] = "backend-abort"
    uid: int

@register_message
@dataclass
class BackendAbortBatchMsg(BaseBatchMessage):
    message_type: ClassVar[str] = "backend-abort-batch"
    data: List[BackendAbortMsg]

@register_message
@dataclass
class BackendExitMsg(BaseMessage):
    message_type: ClassVar[str] = "backend-exit"
