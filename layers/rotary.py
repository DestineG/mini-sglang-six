import functools
import torch


class RotaryEmbedding:
    def __init__(
        self,
        head_dim: int,
        max_position: int,
        base: float,
        device: torch.device,
    ):
        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0, head_dim, 2,
                    dtype=torch.float32,
                    device=device
                ) / head_dim
            )
        )

        positions = torch.arange(
            max_position,
            dtype=torch.float32,
            device=device
        )

        freqs = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos_sin_cache",
            torch.cat([freqs.cos(), freqs.sin()], dim=-1),
            persistent=False
        )

        from flashinfer import apply_rope_with_cos_sin_cache_inplace
        self.apply_rope = apply_rope_with_cos_sin_cache_inplace

        self.head_dim = head_dim

    def forward(self, positions, query, key):
        self.apply_rope(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_dim,
            cos_sin_cache=self.cos_sin_cache,
        )
        return query, key


@functools.lru_cache(maxsize=None)
def get_rope(
    head_dim: int,
    max_position: int,
    base: float,
    device: str,
):
    return RotaryEmbedding(
        head_dim=head_dim,
        max_position=max_position,
        base=base,
        device=torch.device(device),
    )