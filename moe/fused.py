import functools
from typing import Dict, Tuple
import torch

from .base import BaseMoeBackend
from ..utils.misc import div_ceil


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import topk_softmax

    assert hidden_states.shape[0] == gating_output.shape[0]

    M, _ = hidden_states.shape
    # 每个 token 的专家路由权重
    topk_weights = torch.empty(
        M, topk,
        dtype=torch.int32,
        device=hidden_states.device
    )

    # 每个 token 的专家索引
    topk_ids = torch.empty(
        M, topk,
        dtype=torch.int32,
        device=hidden_states.device
    )

    topk_softmax(
        topk_weights,
        topk_ids,
        gating_output.float(),
        renormalize
    )

    return topk_weights, topk_ids

def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False
) -> torch.Tensor:
    from ..kernel.moe_routing import build_expert_inputs
    from ..kernel.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton
    from ..layers.activation import silu_and_mul, gelu_and_mul

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"

    num_experts, intermediate_size, hidden_size = w1.shape
    num_tokens, _ = hidden_states.shape
    topk = topk_ids.shape[1]

    # 展平专家索引
    expert_ids_flat = topk_ids.reshape(-1)

    # 按 Expert ID 排序
    sorted_route_indices = torch.argsort(expert_ids_flat)

    # 预分配专家输入
    expert_inputs = torch.empty(
        (num_tokens * topk, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # 预分配 Expert 偏移
    expert_token_offsets = torch.empty(
        (num_experts + 1,),
        dtype=torch.int32,
        device=hidden_states.device,
    )

    # 构建 moe 输入
    build_expert_inputs(
        hidden_states,
        topk,
        expert_ids_flat,
        sorted_route_indices,
        expert_inputs,
        expert_token_offsets,
    )



    # 3. 第一层 Expert GEMM
    # TODO: 根据 sorted_expert_ids 分组执行 Group GEMM
    # intermediate = ...

    # 4. 激活函数
    # TODO: silu_and_mul / gelu_and_mul

    # 5. 第二层 Expert GEMM
    # TODO: 根据 sorted_expert_ids 执行第二次 Group GEMM

    # 6. 乘以 Router 权重
    # TODO: 使用 sorted_topk_slots 找到对应的 topk_weights

    # 7. 将 Expert 输出按照原始 Token 累加
    # TODO: index_add_ 或自定义 reduce kernel

    # return output