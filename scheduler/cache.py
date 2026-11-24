from typing import List, Tuple
import torch

def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int
):
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)

    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        
        # 填充 page_table 行坐标
        table_idx_host[offset : offset + length].fill_(table_idx)

        # 填充 page_table 列坐标
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length

    # 校验
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."

    # 非阻塞搬运
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    positions = positions_host.to(page_table.device, non_blocking=True)

    # page_table 填充
    page_table[table_idxs, positions] = allocated