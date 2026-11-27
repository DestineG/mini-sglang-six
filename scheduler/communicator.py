from __future__ import annotations

from typing import Final, List
import torch

from ..message.base import BaseMessage, BaseBatchMessage
from ..message.tokenize import DeTokenizeBatchMsg

from ..utils.logger import init_logger
from ..transport.zmq_queue import (
    ZmqPushQueue,
    ZmqPullQueue,
    ZmqPubQueue,
    ZmqSubQueue
)
from ..config.backend_args import BackendArgs


class SchedulerCommunicator:
    def __init__(
        self,
        config: BackendArgs,
        tp_cpu_group: torch.distributed.ProcessGroup
    ):
        self.config = config
        self.tp_cpu_group = tp_cpu_group

        # 收发函数初始化
        self.recv_msgs_from_tokenizer = None
        self.send_msgs_to_detokenizer = None
        self.send_msgs_to_scheduler = None
        self.recv_msgs_from_scheduler = None

        # 通信节点初始化
        self._init_tokenizer_connection()
        self._init_scheduler_connection()

    def _init_tokenizer_connection(self):
        """初始化与 Tokenizer、Detokenizer 的通信连接"""

        if not self.config.tp_info.is_primary():
            return

        self.tokenizer_queue = ZmqPullQueue(
            addr=self.config.tokenizer_addr,
            create=True,
            decoder=BaseMessage.decoder
        )

        self.detokenizer_queue = ZmqPushQueue(
            addr=self.config.detokenizer_addr,
            create=True,
            encoder=BaseMessage.encoder
        )

        # 收发函数
        self.recv_msgs_from_tokenizer = self._recv_msgs_from_tokenizer
        self.send_msgs_to_detokenizer = self._send_msgs_to_detokenizer

    def _init_scheduler_connection(self):
        """初始化 TP Rank 之间的通信连接"""

        if self.config.tp_info.size <= 1:
            return

        if self.config.tp_info.is_primary():
            # Primary Rank：负责广播
            self.scheduler_queue = ZmqPubQueue(
                addr=self.config.backend_broadcast_addr,
                create=True,
                encoder=BaseMessage.encoder
            )

        else:
            # Other Rank：负责接收广播
            self.scheduler_queue = ZmqSubQueue(
                addr=self.config.backend_broadcast_addr,
                create=False,
                decoder=BaseMessage.decoder
            )

            # 接收函数
            self.recv_msgs_from_scheduler = self._recv_msgs_from_scheduler

    def sync_all_ranks(self):
        self.tp_cpu_group.barrier().wait()

    def _append_decoded_msg(
        self,
        pending_msgs: BaseBatchMessage,
        msg: BaseMessage | BaseBatchMessage
    ) -> None:
        if isinstance(msg, BaseMessage):
            pending_msgs.append(msg)
        elif isinstance(msg, BaseBatchMessage):
            pending_msgs.extend(msg)
        else:
            raise TypeError(f"Unexpected message type: {type(msg)}")

    def _recv_msgs_from_tokenizer(self, blocking: bool) -> BaseBatchMessage:
        pending_msgs = BaseBatchMessage(data=[])

        # 阻塞接收
        if blocking:
            raw = self.tokenizer_queue.get_raw()
            if self.config.tp_info.size > 1:
                self.scheduler_queue.put_raw(raw)
            msg = self.tokenizer_queue.decoder(raw)
            self._append_decoded_msg(
                pending_msgs=pending_msgs,
                msg=msg
            )

        # 非阻塞接收
        pending_raw_msgs: List[bytes] = []
        while not self.tokenizer_queue.empty():
            pending_raw_msgs.append(self.tokenizer_queue.get_raw())

        # 广播消息数量
        if self.config.tp_info.size > 1:
            num_pending_raw_msgs = torch.tensor(len(pending_raw_msgs), dtype=torch.int32)
            self.tp_cpu_group.broadcast(num_pending_raw_msgs, root=0).wait()

        # 广播消息
        for raw in pending_raw_msgs:
            if self.config.tp_info.size > 1:
                self.scheduler_queue.put_raw(raw)
            msg = self.tokenizer_queue.decoder(raw)
            self._append_decoded_msg(
                pending_msgs=pending_msgs,
                msg=msg
            )

        return pending_msgs

    def _recv_msgs_from_scheduler(self, blocking: bool = False) -> BaseBatchMessage:
        pending_msgs = BaseBatchMessage(data=[])

        # 阻塞接收
        if blocking:
            msg = self.scheduler_queue.get()
            self._append_decoded_msg(
                pending_msgs=pending_msgs,
                msg=msg
            )

        # 接收消息数量
        num_pending_msgs = torch.tensor(-1, dtype=torch.int32)
        self.tp_cpu_group.broadcast(num_pending_msgs, root=0).wait()
        num_pending_msgs = num_pending_msgs.item()

        # 接收消息
        for _ in range(num_pending_msgs):
            msg = self.scheduler_queue.get()
            self._append_decoded_msg(
                pending_msgs=pending_msgs,
                msg=msg
            )

        return pending_msgs

    def _send_msgs_to_detokenizer(self, msgs: DeTokenizeBatchMsg):
        self.detokenizer_queue.put(msgs)