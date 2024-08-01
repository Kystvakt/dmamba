from typing import Optional, Union, Sequence, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as fn


class FlexiPatchEmbed3d(nn.Module):
    def __init__(
            self,
            input_size: tuple[int, int, int],
            patch_size: tuple[int, int, int],
            stride: tuple[int, int, int],
            channels: int,
            d_embed: int,
            patch_size_seq: Sequence[tuple[int, int, int]],
            patch_size_probs: Optional[Sequence[float]] = None,
            norm_layer: Optional[Union[nn.LayerNorm, nn.BatchNorm3d, nn.Module]] = None,
            bias: bool = True,
    ):
        super().__init__()
        assert len(input_size) == len(patch_size) == len(stride), (
            f"Length of the input size ({input_size}), patch size ({patch_size}) and stride ({stride}) should be equal"
        )
        self.input_size = input_size
        self.patch_size = patch_size
        self.stride = stride
        self.ratio = tuple(float(s / p) for s, p in zip(self.stride, self.patch_size))

        self.proj = nn.Conv3d(
            in_channels=channels,
            out_channels=d_embed,
            kernel_size=patch_size,
            stride=stride,
            bias=bias
        )
        self.norm = norm_layer(d_embed) if norm_layer else nn.Identity()

        self.patch_size_seq = patch_size_seq
        if self.patch_size_seq is not None:
            if patch_size_probs is None:
                n = len(self.patch_size_seq)
                self.patch_size_probs = [1. / n] * n
            else:
                self.patch_size_probs = [p / sum(patch_size_probs) for p in patch_size_probs]
        else:
            self.patch_size_probs = list()
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        pinvs = dict()
        for ps in self.patch_size_seq:
            pinvs[ps] = self._calculate_pinv(self.patch_size, ps)
        return pinvs

    def _calculate_pinv(self, old_shape: tuple, new_shape: tuple) -> torch.Tensor:
        mat = list()
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resized_matrix = torch.stack(mat)
        return torch.linalg.pinv(resized_matrix)

    @staticmethod
    def _resize(x: torch.Tensor, shape: tuple) -> torch.Tensor:
        x_resized = fn.interpolate(
            x[None, None, ...],
            size=shape,
            mode='trilinear',
            antialias=False
        )
        return x_resized[0, 0, ...]

    def resize_patch_embed(self, patch_embed: torch.Tensor, new_patch_size: tuple) -> torch.Tensor:
        if self.patch_size == new_patch_size:
            return patch_embed

        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)

        pinv = self.pinvs[new_patch_size].to(patch_embed.device)

        # inner function
        def resample_patch_embed(patch_embed: torch.Tensor):
            t, h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            resampled_kernel = rearrange(resampled_kernel, '(t h w) -> t h w', t=t, h=h, w=w)
            return resampled_kernel

        v_resampled_patch_embed = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resampled_patch_embed(patch_embed)

    def forward(
            self,
            x: torch.Tensor,
            patch_size: Optional[tuple] = None,
            return_patch_size: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, tuple]]:
        if patch_size is None and not self.training:
            patch_size = self.patch_size
        elif patch_size is None:
            assert self.patch_size_seq, "No patch size specified during forward and no patch_size_seq given"
            patch_size = self.patch_size_seq[np.random.choice(len(self.patch_size_seq), p=self.patch_size_probs)]

        # Resize
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        # stride = (patch_size[0], int(patch_size[1] // 2), int(patch_size[2] // 2))
        # print(stride)
        stride = tuple(int(p * r) for p, r in zip(patch_size, self.ratio))
        # print(stride)
        x = fn.conv3d(x, weight, bias=self.proj.bias, stride=stride)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        if return_patch_size:
            return x, patch_size
        else:
            return x


class UnPatchEmbed3d(nn.Module):
    def __init__(
            self,
            patch_size: tuple[int, int, int],
            stride: tuple[int, int, int],
            channels: int,
            d_embed: int,
            bias: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose3d(
            in_channels=d_embed,
            out_channels=channels,
            kernel_size=patch_size,
            stride=stride,
            bias=bias
        )

    def _calculate_pinv(self, old_shape: Tuple, new_shape: Tuple) -> torch.Tensor:
        mat = list()
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    @staticmethod
    def _resize(x: torch.Tensor, shape: Tuple) -> torch.Tensor:
        x_resized = fn.interpolate(
            x[None, None, ...],
            size=shape,
            mode='trilinear',
            antialias=False
        )
        return x_resized[0, 0, ...]

    def resize_patch_embed(self, patch_embed: torch.Tensor, new_patch_size: Tuple):
        if self.patch_size == new_patch_size:
            return patch_embed
        pinv = self._calculate_pinv(self.patch_size, new_patch_size).to(patch_embed.device)

        def resample_patch_embed(patch_embed):
            t, h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            resampled_kernel = rearrange(resampled_kernel, '(t h w) -> t h w', t=t, h=h, w=w)
            return resampled_kernel

        v_resample_patch_embed = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)
        return v_resample_patch_embed(patch_embed)

    def forward(
            self,
            x: torch.Tensor,
            patch_size: Tuple,
            stride: Tuple
    ) -> torch.Tensor:
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        x = fn.conv_transpose3d(x, weight, bias=self.proj.bias, stride=stride)

        return x
