from pydantic import BaseModel, Field
from typing import Literal, List
import time
from dataclasses import dataclass

from ..protocol.sampling_params import SamplingParams
from ..protocol.generate_params import GenerateParams


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class CompletionRequest(BaseModel):
    model: str
    prompt: str

    sampling: SamplingParams = Field(default_factory=SamplingParams)
    generation: GenerateParams = Field(default_factory=GenerateParams)

    stream: bool = False

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]

    sampling: SamplingParams = Field(default_factory=SamplingParams)
    generation: GenerateParams = Field(default_factory=GenerateParams)

    stream: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owner: str = "sgl"
    root: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)

@dataclass
class FrontendManager:
    initialized: bool = False


if __name__ == "__main__":
    # 1. 测试 Message
    message = Message.model_validate({
        "role": "user",
        "content": "什么是 KV Cache？"
    })
    print("Message:", message)

    # 2. 测试 CompletionRequest
    completion = CompletionRequest.model_validate({
        "model": "Qwen3-0.6B",
        "prompt": "什么是 KV Cache？",
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9
        },
        "generation": {
            "max_tokens": 256,
            "n": 3
        },
        "stream": True
    })

    print("\nCompletionRequest:")
    print(completion)
    print("temperature:", completion.sampling.temperature)
    print("max_tokens:", completion.generation.max_tokens)

    # 3. 测试 ChatCompletionRequest
    chat = ChatCompletionRequest.model_validate({
        "model": "Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "你是一个 AI 助手"},
            {"role": "user", "content": "什么是 KV Cache？"}
        ]
    })

    print("\nChatCompletionRequest:")
    print(chat)
    print("first role:", chat.messages[0].role)

    # 4. 测试默认值是否独立
    req1 = CompletionRequest(model="Qwen3-0.6B", prompt="Hello")
    req2 = CompletionRequest(model="Qwen3-0.6B", prompt="World")

    req1.generation.stop.append("END")
    req1.sampling.temperature = 0.5

    print("\nDefault Factory Test:")
    print("req1.stop:", req1.generation.stop)
    print("req2.stop:", req2.generation.stop)
    print("req1.temperature:", req1.sampling.temperature)
    print("req2.temperature:", req2.sampling.temperature)

    assert req1.generation.stop == ["END"]
    assert req2.generation.stop == []
    assert req1.generation.stop is not req2.generation.stop
    assert req1.sampling is not req2.sampling
    assert req1.generation is not req2.generation

    print("\nAll tests passed!")
