import torch


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32

class TableManager:
    def __init__(
        self,
        max_running_req: int,
        max_seq_len: int,
        needed_dummy_request: bool,
        device: torch.device
    ):
        # + 1 for dummy request
        if needed_dummy_request:
            max_running_req += 1

        # 128 bytes 向上取整
        aligned_max_seq_len = _align_up_32(max_seq_len)

        self.page_table = torch.zeros(
            (max_running_req, aligned_max_seq_len),
            dtype=torch.int32,
            device=device
        )
        self._free_slots = list(range(max_running_req))

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)