from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, ConvBnAct, Mlp, PatchEmbed

default_cfgs = {
    "svtr_tiny": {
        "embed_dim": [64, 128, 256, 192],
        "depths": [3, 6, 3],
        "num_heads": [2, 4, 8],
        "mixers": ["local"] * 6 + ["global"] * 6,
    },
}


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hw: tuple[int, int] = (8, 25),
        local_k: tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.hw = hw
        self.local_k = local_k

        self.local_mixer = nn.Conv2d(
            dim, dim, kernel_size=local_k, padding=local_k[0] // 2, groups=num_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.hw
        x = rearrange(x, "b (h w) d -> b d h w", h=h, w=w)
        x = self.local_mixer(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mixer: Literal["local", "global"] = "global",
        hw: Optional[tuple[int, int]] = None,
        local_k: tuple[int, int] = (7, 11),
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.mixer = mixer
        self.hw = hw
        self.local_k = local_k
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if mixer == "local":
            assert hw is not None, "hw must be specified for local mixer"
            h, w = hw
            hk, wk = local_k
            mask = torch.ones((h * w, h + hk - 1, w + wk - 1), dtype=torch.float32)
            for i in range(h):
                for j in range(w):
                    mask[w * i + j, i : i + hk, j : j + wk] = 0
            mask_paddle = mask[:, hk // 2 : h + hk // 2, wk // 2 : w + wk // 2].flatten(
                1
            )
            mask_inf = torch.full((h * w, h * w), float("-inf"), dtype=torch.float32)
            mask = torch.where(mask_paddle == 1, mask_inf, mask_paddle)
            self.mask = mask[None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hw is not None:
            N = self.hw[0] * self.hw[1]
            C = self.dim
        else:
            N, C = x.shape[-2:]

        # TODO: add flash attention
        qkv = rearrange(
            self.qkv(x),
            "b s (three h d) -> three b h s d",
            three=3,
            h=self.num_heads,
            d=C // self.num_heads,
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn: torch.Tensor = q @ k.transpose(-2, -1)

        if self.mixer == "local":
            attn = attn + self.mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = rearrange(attn @ v, "b h s d -> b s (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding that uses 2 or 3 conv layers to embed the image
    Reference: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/modeling/backbones/rec_svtrnet.py#L269
    """

    def __init__(
        self, in_channels: int = 3, embed_dim: int = 768, num_layers: int = 2
    ) -> None:
        super().__init__()

        assert num_layers in [2, 3], "num_layers must be 2 or 3"

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        proj_config = {
            2: (in_channels, embed_dim // 2, embed_dim),
            3: (in_channels, embed_dim // 4, embed_dim // 2, embed_dim),
        }[num_layers]

        self.proj = nn.Sequential(
            *[
                ConvBnAct(
                    x,
                    y,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act_layer=nn.GELU,
                    bias=True,
                )
                for x, y in zip(proj_config, proj_config[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, c, h, w]
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


if __name__ == "__main__":
    model = Attention(128, hw=(8, 8), local_k=(3, 3), mixer="local")
