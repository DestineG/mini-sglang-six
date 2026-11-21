from typing import ClassVar

from pydantic import BaseModel

from ..message.utils import register_message


@register_message
class SamplingParams(BaseModel):
    message_type: ClassVar[str] = "sampling-params"

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
