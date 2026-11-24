import torch


class TableManager:
    def __init__(self, page_table: torch.Tensor):
        self.page_table = page_table
        self._free_slots = list(range(page_table.size(0)))

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)