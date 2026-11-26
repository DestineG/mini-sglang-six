
## Environment

Python 3.13; NVIDIA CUDA 13.x toolkit and driver required for the GPU extra.

```sh
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
export FLASH_ATTN_CUDA_ARCHS=120 NVCC_THREADS=2
uv sync --extra cu130
```

`uv sync` installs common dependencies only. Use `--extra cu130` for RTX5090;
`--extra cpu` installs CPU PyTorch without FlashAttention. These extras conflict.
FlashAttention imports torch during its build; extra-build-dependencies uses the
same torch version as the selected runtime. The first GPU installation downloads
CUDA libraries and may compile FlashAttention locally. Keep MAX_JOBS bounded.
After installing a GPU environment, keep specifying `--extra cu130` when syncing
or running through uv, e.g. `uv run --extra cu130 python main.py`.

Installation log for the current repair: /tmp/mini-sglang-cu130-install.log.

Pinned torch2.9.1+cu130 and flash-attn2.8.3.post2: newer torch2.14 headers require C++20, while this FlashAttention builds with C++17. FLASH_ATTN_CUDA_ARCHS=120 targets this RTX5090 only; choose the appropriate architecture on another GPU.

Verified on RTX5090: uv sync --extra cu130 exited0; flash_attn_func BF16 causal attention matched PyTorch SDPA (max absolute difference0). Logs: /tmp/mini-sglang-cu130-install-v2.log and /tmp/mini-sglang-fa-check.log.
