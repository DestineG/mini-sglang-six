from pydantic import BaseModel


class SamplingParams(BaseModel):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
