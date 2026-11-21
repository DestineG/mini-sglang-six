from typing import ClassVar

from pydantic import BaseModel, Field

from ..message.utils import register_message


@register_message
class GenerateParams(BaseModel):
    message_type: ClassVar[str] = "generate-params"

    max_tokens: int = 128
    n: int = 1
    stop: list[str] = Field(default_factory=list)
    ignore_eos: bool = False
