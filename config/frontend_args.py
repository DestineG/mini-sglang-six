from dataclasses import dataclass

from .common_args import CommonArgs


@dataclass(frozen=True)
class FrontendArgs(CommonArgs):
    host: str = "127.0.0.1"
    port: int = 1919