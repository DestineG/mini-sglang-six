from dataclasses import dataclass
from typing import List

from .base import BaseRequestManager
from ...protocol.request import Request


@dataclass
class WaitingManager(BaseRequestManager):
    ...