from pydantic import BaseModel, Field


class GenerateParams(BaseModel):
    max_tokens: int = 128
    n: int = 1
    stop: list[str] = Field(default_factory=list)
    ignore_eos: bool = False