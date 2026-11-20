from dataclasses import dataclass, field
import os
from typing import ClassVar


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class CommonArgs:
    silent_output: bool = False
    offline_mode: bool = False

    # Networking config
    _unique_prefix: ClassVar[str] = "ipc:///tmp/minisgl"
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def _build_addr(self, index: int) -> str:
        return f"{self._unique_prefix}_{index}{self._unique_suffix}"

    @property
    def frontend_addr(self) -> str:
        return self._build_addr(0)

    @property
    def tokenizer_addr(self) -> str:
        return self._build_addr(1)

    @property
    def detokenizer_addr(self) -> str:
        return self._build_addr(2)

    @property
    def backend_addr(self) -> str:
        return self._build_addr(3)

    @property
    def backend_broadcast_addr(self) -> str:
        return self._build_addr(4)
