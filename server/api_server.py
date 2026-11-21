from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
import time
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
from starlette.background import BackgroundTask
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

from ..transport.zmq_queue import ZmqAsyncPushQueue, ZmqAsyncPullQueue
from ..message.tokenize import TokenizeMsg, TokenizeAbortMsg
from ..message.frontend import FrontendMsg, FrontendBatchMsg
from ..config.frontend_args import FrontendArgs
from ..config.tokenize_args import TokenizeArgs
from ..config.backend_args import BackendArgs
from ..protocol.sampling_params import SamplingParams
from ..protocol.generate_params import GenerateParams
from ..utils.env import ENV
from ..utils.logger import init_logger



logger = init_logger(__name__, "FrontendAPI")

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class GenerateRequest(BaseModel):
    model: str
    prompt: str

    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
    generate_params: GenerateParams = Field(default_factory=GenerateParams)

    stream: bool = False

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]

    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
    generate_params: GenerateParams = Field(default_factory=GenerateParams)

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
    send_tokenizer: ZmqAsyncPushQueue[TokenizeMsg]
    recv_detokenizer: ZmqAsyncPullQueue[FrontendMsg | FrontendBatchMsg]
    backend_configs: BackendArgs

    initialized: bool = False
    uid_counter: int = 0
    ack_map: Dict[int, List[FrontendMsg]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.uid_counter += 1
        return uid

    async def listen(self):
        while True:
            message = await self.recv_detokenizer.get()
            assert isinstance(message, FrontendMsg | FrontendBatchMsg)
            msgs: List[FrontendMsg] = []
            if isinstance(message, FrontendBatchMsg):
                for msg in message.data:
                    msgs.append(msg)
            else:
                msgs.append(message)

            for msg in msgs:
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _launch_listener(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: TokenizeMsg):
        self._launch_listener()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        try:
            while True:
                await event.wait()
                event.clear()

                pending = self.ack_map[uid]
                self.ack_map[uid] = []

                finished = bool(pending and pending[-1].finished)

                for ack in pending:
                    yield ack

                if finished:
                    break

        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n\n".encode()

        yield "data: [DONE]\n\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"chatcmpl-{uid}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        # send final finish_reason
        end_chunk = {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def collect_with_cancellation(self, uid: int, request: Request) -> str:
        async def collect():
            full_content = ""
            async for ack in self.wait_for_ack(uid):
                full_content += ack.incremental_output
            return full_content

        collect_task = asyncio.create_task(collect())

        try:
            while not collect_task.done():
                if await request.is_disconnected():
                    raise asyncio.CancelledError

                await asyncio.sleep(0.1)

            return await collect_task

        except asyncio.CancelledError:
            collect_task.cancel()
            await asyncio.gather(
                collect_task,
                return_exceptions=True,
            )
            await self.abort_user(uid)
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(TokenizeAbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_detokenizer.stop()

_GLOBAL_STATE = None

def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()

app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=req.sampling_params,
            generate_params=req.generate_params
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_generate(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = await state.collect_with_cancellation(uid, request)

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}

@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.backend_configs.model_path, root=state.backend_configs.model_path)])

@app.post("/v1/chat/completions")
async def v1_completions(req: ChatCompletionRequest, request: Request):
    state = get_global_state()
    assert req.messages, f""
    prompt = [msg.model_dump() for msg in req.messages]
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=req.sampling_params,
            generate_params=req.generate_params
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = await state.collect_with_cancellation(uid, request)

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

async def shell_completion(req: ChatCompletionRequest):
    state = get_global_state()
    assert req.messages, f""
    prompt = [msg.model_dump() for msg in req.messages]
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=req.sampling_params,
            generate_params=req.generate_params
        )
    )

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream"
    )

async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            
            if cmd == "":
                continue

            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")

            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))

            top_k=ENV.SHELL_TOP_K.value
            top_p=ENV.SHELL_TOP_P.value
            temperature=ENV.SHELL_TEMPERATURE.value
            max_tokens=ENV.SHELL_MAX_TOKENS.value
            req = ChatCompletionRequest(
                model="",
                messages=history_messages + [
                    Message(role="user", content=cmd)
                ],
                sampling_params=SamplingParams(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ),
                generate_params=GenerateParams(max_tokens=max_tokens),
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n\n"), msg
                msg = msg[:-2]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))

    except EOFError:
        # user pressed Ctrl-D
        pass

    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()

def run_api_server(
        frontend_config: FrontendArgs,
        tokenize_config: TokenizeArgs,
        backend_config: BackendArgs,
        start_backend: Callable[[], None],
        run_shell: bool
) -> None:
    global _GLOBAL_STATE

    if run_shell:
        assert not backend_config.use_dummy_weight, "Shell mode does not support dummy weights."

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=backend_config,
        send_tokenizer=ZmqAsyncPushQueue(
            tokenize_config.tokenizer_addr,
            create=True,
            encoder=TokenizeMsg.encoder,
                ),
        recv_detokenizer=ZmqAsyncPullQueue(
            frontend_config.frontend_addr,
            create=True,
            decoder=FrontendMsg.decoder,
        ),
    )

    start_backend()

    host = frontend_config.host
    port = frontend_config.port
    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())

if __name__ == "__main__":
    # 1. 测试 Message
    message = Message.model_validate({
        "role": "user",
        "content": "什么是 KV Cache？"
    })
    print("Message:", message)

    # 2. 测试 GenerateRequest
    completion = GenerateRequest.model_validate({
        "model": "Qwen3-0.6B",
        "prompt": "什么是 KV Cache？",
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9
        },
        "generate_params": {
            "max_tokens": 256,
            "n": 3
        },
        "stream": True
    })

    print("\nCompletionRequest:")
    print(completion)
    print("temperature:", completion.sampling_params.temperature)
    print("max_tokens:", completion.generate_params.max_tokens)

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
    req1 = GenerateRequest(model="Qwen3-0.6B", prompt="Hello")
    req2 = GenerateRequest(model="Qwen3-0.6B", prompt="World")

    req1.generate_params.stop.append("END")
    req1.sampling_params.temperature = 0.5

    print("\nDefault Factory Test:")
    print("req1.stop:", req1.generate_params.stop)
    print("req2.stop:", req2.generate_params.stop)
    print("req1.temperature:", req1.sampling_params.temperature)
    print("req2.temperature:", req2.sampling_params.temperature)

    assert req1.generate_params.stop == ["END"]
    assert req2.generate_params.stop == []
    assert req1.generate_params.stop is not req2.generate_params.stop
    assert req1.sampling_params is not req2.sampling_params
    assert req1.generate_params is not req2.generate_params

    print("\nAll tests passed!")
