from typing import Tuple

from .frontend_args import FrontendArgs
from .tokenize_args import TokenizeArgs
from .backend_args import BackendArgs

def build() -> Tuple[Tuple[FrontendArgs, TokenizeArgs, BackendArgs], bool]:
    pass