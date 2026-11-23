from dataclasses import dataclass

from .request import Request


@dataclass(eq=False)
class Batch:
    reqs: list[Request]

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def num_tokens(self) -> int:
        return sum(req.schedule_len for req in self.reqs)