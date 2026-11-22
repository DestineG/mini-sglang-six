from typing import Tuple, List
import argparse

from .frontend_args import FrontendArgs
from .tokenize_args import TokenizeArgs
from .backend_args import BackendArgs

def build(
        frontend_config: FrontendArgs,
        tokenize_config: TokenizeArgs,
        backend_config: BackendArgs
) -> Tuple[Tuple[FrontendArgs, TokenizeArgs, BackendArgs], bool]:
    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    # common
    parser.add_argument(
        "--silent-output",
        action="store_true",
        dest="silent_output",
        default=frontend_config.silent_output,
        help="Suppress non-essential server output.",
    )

    parser.add_argument(
        "--offline-mode",
        action="store_true",
        dest="offline_mode",
        default=frontend_config.offline_mode,
        help="Run the server in offline mode.",
    )

    # frontend
    parser.add_argument(
        "--host",
        type=str,
        dest="host",
        default=frontend_config.host,
        help="Host address for the frontend HTTP server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="port",
        default=frontend_config.port,
        help="Port number for the frontend HTTP server.",
    )

    # tokenize
    parser.add_argument(
        "--num-tokenizer",
        type=int,
        dest="num_tokenizer",
        default=tokenize_config.num_tokenizer,
        help="Number of tokenizer worker processes (>= 1).",
    )

    parser.add_argument(
        "--num-detokenizer",
        type=int,
        dest="num_detokenizer",
        default=tokenize_config.num_detokenizer,
        help="Number of detokenizer worker processes (0 or 1).",
    )

    # backend
    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=backend_config.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="Source for downloading model weights.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        dest="tensor_parallel_size",
        default=backend_config.tp_info.world_size,
        help="Number of GPUs used for tensor parallelism.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=backend_config.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        dest="model_path",
        required=True,
        help="Path to the model weights (local directory or Hugging Face repo ID).",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=backend_config.dtype,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. "
            "'auto' uses the dtype specified in the model config.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=backend_config.max_running_req,
        help="Maximum number of concurrent running requests.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=backend_config.attention_backend,
        help="Attention backend to use. If two backends are specified, "
            "the first is used for prefill and the second for decode.",
    )

    parser.add_argument(
        "--moe-backend",
        type=str,
        default=backend_config.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="MoE computation backend to use.",
    )

    parser.add_argument(
        "--cuda-graph-bs",
        type=int,
        nargs="+",
        default=backend_config.cuda_graph_bs,
        help="List of batch sizes to capture with CUDA Graph.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=backend_config.cuda_graph_max_bs,
        help="Maximum batch size for automatic CUDA Graph capture.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=backend_config.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=backend_config.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    parser.add_argument(
        "--distributed-timeout",
        type=float,
        default=backend_config.distributed_timeout,
        help="Timeout in seconds for distributed communication initialization and operations.",
    )

    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        default=backend_config.use_dummy_weight,
        help="Use dummy weights for testing.",
    )

    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        default=backend_config.use_pynccl,
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=backend_config.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=backend_config.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )