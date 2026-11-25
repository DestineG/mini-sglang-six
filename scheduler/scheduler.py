from typing import List, Set, Any

from ..protocol.request import Request
from ..message.base import BaseMessage, BaseBatchMessage
from ..message.backend import (
    BackendMsg,
    BackendBatchMsg,
    BackendAbortMsg,
    BackendAbortBatchMsg,
    BackendExitMsg
)
from ..utils.logger import init_logger
from .request import (
    WaitingManager,
    PrefillManager,
    DecodeManager
)
from .table import TableManager
from .cache import CacheManager
from ..config.backend_args import BackendArgs
from ..protocol.context import get_global_ctx
from ..engine.engine import Engine


logger = init_logger(__name__)


class Scheduler:
    def __init__(self, config: BackendArgs):
        self.config = config
        self.engine = Engine(config)
        self.ctx = get_global_ctx()
        self.page_table_manager = TableManager(
            max_running_req=self.config.max_running_req,
            max_seq_len=self.config.max_seq_len,
            needed_dummy_request=self.config.needed_dummy_request
        )
        self.cache_manager = CacheManager(
            num_pages=self.ctx.num_pages,
            page_size=self.ctx.page_size,
            page_table=self.page_table_manager.page_table
        )

        self.waiting_manager = WaitingManager()
        self.prefill_manager = PrefillManager()
        self.decode_manager = DecodeManager()


    def _exit(self):
        raise KeyboardInterrupt

    def handle_backend_msg_batch(self, msgs: List[BackendMsg]):
        reqs = [
            Request(
                uid=msg.uid,
                input_ids=msg.input_ids,
                sampling_params=msg.sampling_params,
                generate_params=msg.generate_params,
                table_idx=-1,
                prompt_len=len(msg.input_ids),
                cached_len=0,
                output_len=0,
                schedule_len=len(msg.input_ids),
                cache_handle=None
            ) for msg in msgs
        ]
        self.waiting_manager.add(requests=reqs)

    def handle_backend_abort_batch(self, msgs: List[BackendAbortMsg]):
        ...

    def handle_msgs_from_tokenize(self, msgs: BaseBatchMessage):

        msgs = msgs.data

        if not msgs:
            return

        msg_list: List[BackendMsg] = []
        abort_list: List[BackendAbortMsg] = []
        abort_set: Set[Any] = set()
        for item in msgs:
            if isinstance(item, BackendMsg):
                msg_list.append(item)
            elif isinstance(item, BackendAbortMsg):
                abort_list.append(item)
                abort_set.add(item.uid)
            elif isinstance(item, BackendExitMsg):
                self._exit()
            else:
                raise TypeError(
                    f"Unsupported message type in batch: {type(item).__name__}"
                )

        msg_list = [item for item in msg_list if item.uid not in abort_set]

        if msg_list:
            self.handle_backend_msg_batch(msg_list)

        if abort_list:
            self.handle_backend_abort_batch(abort_list)

    def get_max_schedule_len(self, req: Request) -> int:
        decode_count = len(self.decode_manager.requests)
        prefill_count = len(self.prefill_manager.requests)
        chunked_count = len(self.prefill_manager.chunked_requests)

        # 根据当前系统压力决定 Chunk Size
        if decode_count >= 64:
            return 128
        elif decode_count >= 32:
            return 256
        elif decode_count >= 16:
            return 512
        else:
            return 1024