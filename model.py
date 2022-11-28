"""
 Copyright 2022 Lengyue

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# Reference
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/modeling/backbones/rec_svtrnet.py

from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import Mlp, DropPath

default_cfgs = {
    "svtr_tiny": {
        "embed_dim": [64, 128, 256],
        "out_channels": 192,
        "depth": [3, 6, 3],
        "num_heads": [2, 4, 8],
        "mixer": ["local"] * 6 + ["global"] * 6,
    },
    "svtr_small": {
        "embed_dim": [96, 192, 256],
        "out_channels": 192,
        "depth": [3, 6, 6],
        "num_heads": [3, 6, 8],
        "mixer": ["local"] * 8 + ["global"] * 7,
    },
    "svtr_base": {
        "embed_dim": [128, 256, 384],
        "out_channels": 256,
        "depth": [3, 6, 9],
        "num_heads": [4, 8, 12],
        "mixer": ["local"] * 8 + ["global"] * 10,
    },
    "svtr_large": {
        "embed_dim": [192, 256, 512],
        "out_channels": 384,
        "depth": [3, 9, 9],
        "num_heads": [6, 8, 16],
        "mixer": ["local"] * 10 + ["global"] * 11,
    },
}


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias=False,
        groups=1,
        act_layer: Optional[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
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

        assert mixer in ["local", "global"], "mixer must be local or global"

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
            self.mask = nn.Parameter(mask[None, None], requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hw is not None:
            C = self.dim
        else:
            C = x.shape[-2:]

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


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: Literal["local", "global"] = "global",
        hw: tuple[int, int] = None,
        local_k: tuple[int, int] = (7, 11),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mixer = Attention(
            dim=dim,
            num_heads=num_heads,
            mixer=mixer,
            hw=hw,
            local_k=local_k,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding that uses 2 or 3 conv layers to embed the image
    Reference: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/modeling/backbones/rec_svtrnet.py#L269
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (32, 100),
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        assert num_layers in [2, 3], "num_layers must be 2 or 3"

        num_patches = (img_size[1] // (2**num_layers)) * (
            img_size[0] // (2**num_layers)
        )

        self.img_size = img_size
        self.num_patches = num_patches
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


class SubSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (2, 1),
        act_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        x = self.act(x)

        return x


class SVTRNet(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (32, 100),
        in_channels: int = 3,
        embed_dim: list[int] = [64, 128, 256],
        depth: list[int] = [3, 6, 3],
        num_heads: list[int] = [2, 4, 8],
        mixer: list[Literal["local", "global"]] = ["local"] * 6 + ["global"] * 6,
        local_mixer: list[tuple[int, int]] = [(7, 11)] * 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        last_drop: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_channels: int = 192,
        out_char_num: int = 25,
        sub_num: int = 2,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.pre_norm = pre_norm

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            num_layers=sub_num,
        )
        num_patches = self.patch_embed.num_patches

        self.hw = (img_size[0] // (2**sub_num), img_size[1] // (2**sub_num))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, sum(depth))

        # Build layers
        self.blocks1 = nn.Sequential(
            *[
                Block(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mixer=mixer[i],
                    hw=self.hw,
                    local_k=local_mixer[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pre_norm=pre_norm,
                )
                for i in range(depth[0])
            ]
        )

        self.sub_sample1 = SubSample(
            embed_dim[0],
            embed_dim[1],
        )

        self.blocks2 = nn.Sequential(
            *[
                Block(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mixer=mixer[depth[0] + i],
                    hw=(self.hw[0] // 2, self.hw[1]),
                    local_k=local_mixer[1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] + i],
                    pre_norm=pre_norm,
                )
                for i in range(depth[1])
            ]
        )

        self.sub_sample2 = SubSample(
            embed_dim[1],
            embed_dim[2],
        )

        self.blocks3 = nn.Sequential(
            *[
                Block(
                    dim=embed_dim[2],
                    num_heads=num_heads[2],
                    mixer=mixer[depth[0] + depth[1] + i],
                    hw=(self.hw[0] // 4, self.hw[1]),
                    local_k=local_mixer[2],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] + depth[1] + i],
                    pre_norm=pre_norm,
                )
                for i in range(depth[2])
            ]
        )

        # Classifier head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, out_char_num))
        self.last_conv = nn.Conv2d(
            embed_dim[2], out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.hard_swish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

        # Init weights
        nn.init.trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks1(x)
        x = self.sub_sample1(
            rearrange(x, "b (h w) c -> b c h w", c=self.embed_dim[0], h=self.hw[0], w=self.hw[1])
        )
        x = self.blocks2(x)
        x = self.sub_sample2(
            rearrange(x, "b (h w) c -> b c h w", c=self.embed_dim[1], h=self.hw[0] // 2, w=self.hw[1])
        )
        x = self.blocks3(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        x = rearrange(x, "b (h w) c -> b c h w", h=self.hw[0] // 4, w=self.hw[1])

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hard_swish(x)
        x = self.dropout(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class CTCHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.log_softmax(dim=2)

        return x


def build_network(cfg_name: str, out_channels: int, **kwargs) -> nn.Module:
    cfg = default_cfgs[cfg_name]

    backbone = SVTRNet(**cfg, **kwargs)
    model = nn.Sequential(
        backbone,
        Rearrange("b c h w -> b (h w) c"),
        CTCHead(backbone.out_channels, out_channels),
    )

    return model


if __name__ == "__main__":
    model = build_network("svtr_tiny", 100)

    test_input = torch.randn(1, 3, 32, 100)
    test_target = torch.randint(0, 25, (1, 10))
    x = model(test_input)
    x = x.permute(1, 0, 2)
    print(x.shape)

    loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    loss(
        x, test_target, torch.tensor([x.shape[0]]), torch.tensor([test_target.shape[1]])
    )
