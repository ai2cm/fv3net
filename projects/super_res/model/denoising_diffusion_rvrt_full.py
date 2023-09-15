import os
import math
from pathlib import Path
from random import random, randint
from functools import partial, reduce, lru_cache
from collections import namedtuple
from operator import mul

import numpy as np
import cv2
from scipy.stats import wasserstein_distance

import xarray as xr

import torch
from torch import nn
import torch.nn.functional as F
import wandb

import piq
import pickle

from torchvision.transforms.functional import crop

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable as smap

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from einops import rearrange
import einops
from einops.layers.torch import Rearrange

from PIL import Image

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from distutils.version import LooseVersion
from .op.deform_attn import deform_attn, DeformAttnPack

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, reduction = None):
        diff = x - y
        loss = torch.sqrt((diff * diff) + self.eps)
        return loss

def save_image(tensor, path):
    im = Image.fromarray((tensor[:,:,:3] * 255).astype(np.uint8))
    im.save(path)
    return None

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# model helpers

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=26, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [ref]
        supp = [supp]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2 ** (5 - level)  # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h // scale, w // scale), mode='bilinear',
                                         align_corners=False)
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


class GuidedDeformAttnPack(DeformAttnPack):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(GuidedDeformAttnPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv3d(self.in_channels * (1 + self.clip_size) + self.clip_size * 2, 64, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, self.clip_size * self.deformable_groups * self.attn_size * 2, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
        )
        self.init_offset()

        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = int(self.in_channels * 2)
        self.proj_q = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_k = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_v = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                  nn.Linear(self.proj_channels, self.in_channels),
                                  Rearrange('n d h w c -> n d c h w'))
        self.mlp = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                 Mlp(self.in_channels, self.in_channels * 2, self.in_channels),
                                 Rearrange('n d h w c -> n d c h w'))

    def init_offset(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, q, k, v, v_prop_warped, flows, return_updateflow):
        offset1, offset2 = torch.chunk(self.max_residue_magnitude * torch.tanh(
            self.conv_offset(torch.cat([q] + v_prop_warped + flows, 2).transpose(1, 2)).transpose(1, 2)), 2, dim=2)
        offset1 = offset1 + flows[0].flip(2).repeat(1, 1, offset1.size(2) // 2, 1, 1)
        offset2 = offset2 + flows[1].flip(2).repeat(1, 1, offset2.size(2) // 2, 1, 1)
        offset = torch.cat([offset1, offset2], dim=2).flatten(0, 1)

        b, t, c, h, w = offset1.shape
        q = self.proj_q(q).view(b * t, 1, self.proj_channels, h, w)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(q, kv, offset, self.kernel_h, self.kernel_w, self.stride, self.padding, self.dilation,
                        self.attention_heads, self.deformable_groups, self.clip_size).view(b, t, self.proj_channels, h,
                                                                                           w)
        v = self.proj(v)
        v = v + self.mlp(v)

        if return_updateflow:
            return v, offset1.view(b, t, c // 2, 2, h, w).mean(2).flip(2), offset2.view(b, t, c // 2, 2, h, w).mean(
                2).flip(2)
        else:
            return v

def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class WindowAttention(nn.Module):
    """ Window based multi-head self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C))

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, -1, dtype=q.dtype)  # Don't use attn.dtype after addition!
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size):
        ''' Get pair-wise relative position index for each token inside the window. '''

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index

class STL(nn.Module):
    """ Swin Transformer Layer (STL).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(2, 8, 8),
                 shift_size=(0, 0, 0),
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        return x

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        x = x + self.forward_part2(x)

        return x


class STG(nn.Module):
    """ Swin Transformer Group (STG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=[2, 8, 8],
                 shift_size=None,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            STL(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x

class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB).

    Args:
        kwargs: Args for RSTB.
    """

    def __init__(self, groups = 8, **kwargs):
        super(RSTB, self).__init__()
        self.input_resolution = kwargs['input_resolution']

        self.residual_group = STG(**kwargs)
        self.linear = nn.Linear(kwargs['dim'], kwargs['dim'])
        self.proj = nn.Conv3d(kwargs['dim'],
                           kwargs['dim'],
                           kernel_size=(1,3,3),
                           padding=(0,1,1),
                           groups=groups)
        self.norm = nn.GroupNorm(groups, kwargs['dim'])
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        x = self.act(x)
        
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)

class RSTBWithInputConv(nn.Module):
    """RSTB with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        kernel_size (int): Size of kernel of the first conv.
        stride (int): Stride of the first conv.
        group (int): Group of the first conv.
        num_blocks (int): Number of residual blocks. Default: 2.
         **kwarg: Args for RSTB.
    """

    def __init__(self, in_channels=3, kernel_size=(1, 3, 3), stride=1, groups=1, num_blocks=2, **kwargs):
        super(RSTBWithInputConv, self).__init__()

        self.in_channels = in_channels
        self.init_conv = nn.Conv3d(in_channels,
                           kwargs['dim'],
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                           groups=groups)
        
        self.init_norm = nn.LayerNorm(kwargs['dim'])

        # RSTB blocks
        #kwargs['use_checkpoint_attn'] = kwargs.pop('use_checkpoint_attn')[0]
        #kwargs['use_checkpoint_ffn'] = kwargs.pop('use_checkpoint_ffn')[0]
        
        #main.append(make_layer(RSTB, num_blocks, **kwargs))
        self.main1 = []
        for _ in range(num_blocks):
            self.main1.append(RSTB(**kwargs).cuda())

        main2 = []
        main2 += [Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n d c h w')]

        self.main2 = nn.Sequential(*main2)
        
    def forward(self, x):
        """
        Forward function for RSTBWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, t, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, t, out_channels, h, w)
        """

        
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.init_conv(x)
        
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.init_norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        for i in range(len(self.main1)):
            x = self.main1[i](x)
        x = self.main2(x)

        return x

class Upsample(nn.Module):
    '''Upsample module for video SR.
    
    Args:
        scale (int): Scale factor. Supported scales: 4.
        num_feat (int): Channel number of intermediate features.
    '''

    def __init__(self, scale, num_feat, **kwargs):
        super(Upsample, self).__init__()

        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        self.feat1 = nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.feat2 = nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.feat3 = nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
        self.upsample1 = nn.PixelShuffle(2)
        self.upsample2 = nn.PixelShuffle(2)
        self.upsample3 = nn.PixelShuffle(2)
        
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.1)
        
        self.final = nn.Conv3d(num_feat, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.feat1(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample1(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.lrelu1(x)
        x = self.feat2(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample2(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.lrelu2(x)
        x = self.feat3(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample3(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.lrelu3(x)
        
        x = self.final(x)
        x = rearrange(x, 'n c d h w -> n d c h w')

        return x

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        feat_ext,
        feat_up,
        backbone,
        deform_align,
        recon,
        spynet,
        *,
        image_size,
        timesteps = 1200,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        auto_normalize = True
    ):
        super(GaussianDiffusion, self).__init__()
        self.clip_size = 2
        self.feat_ext = feat_ext
        self.feat_up = feat_up
        
        self.backbone = backbone
        
        self.deform_align = deform_align
        
        self.recon = recon

        self.spynet = spynet
        
        self.channels = self.feat_ext.in_channels
        
        self.image_size = image_size

        self.loss_type = loss_type

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'charbonnier':
            return CharbonnierLoss()
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
        
    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward
    
    def propagate(self, feats, flows, module_name, updated_flows=None):
        """Propagate the latent clip features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, clip_size, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
            updated_flows dict(list[tensor]): Each component is a list of updated
                optical flows with shape (n, clip_size, 2, h, w).

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """
        
        n, t, _, h, w = flows.size()
        if 'backward' in module_name:
            flow_idx = range(0, t + 1)[::-1]
            clip_idx = range(0, (t + 1) // self.clip_size)[::-1]
        else:
            flow_idx = range(-1, t)
            clip_idx = range(0, (t + 1) // self.clip_size)

        if '_1' in module_name:
            updated_flows[f'{module_name}_n1'] = []
            updated_flows[f'{module_name}_n2'] = []

        feat_prop = torch.zeros_like(feats['shallow'][0])#.cuda()

        last_key = list(feats)[-2]

        for i in range(0, len(clip_idx)):
            idx_c = clip_idx[i]
            if i > 0:
                if '_1' in module_name:
                    flow_n01 = flows[:, flow_idx[self.clip_size * i - 1], :, :, :]
                    flow_n12 = flows[:, flow_idx[self.clip_size * i], :, :, :]
                    flow_n23 = flows[:, flow_idx[self.clip_size * i + 1], :, :, :]
                    flow_n02 = flow_n12 + flow_warp(flow_n01, flow_n12.permute(0, 2, 3, 1))
                    flow_n13 = flow_n23 + flow_warp(flow_n12, flow_n23.permute(0, 2, 3, 1))
                    flow_n03 = flow_n23 + flow_warp(flow_n02, flow_n23.permute(0, 2, 3, 1))
                    flow_n1 = torch.stack([flow_n02, flow_n13], 1)
                    flow_n2 = torch.stack([flow_n12, flow_n03], 1)
                else:
                    module_name_old = module_name.replace('_2', '_1')
                    flow_n1 = updated_flows[f'{module_name_old}_n1'][i - 1]
                    flow_n2 = updated_flows[f'{module_name_old}_n2'][i - 1]

            
                if 'backward' in module_name:
                    feat_q = feats[last_key][idx_c].flip(1)
                    feat_k = feats[last_key][clip_idx[i - 1]].flip(1)
                else:
                    feat_q = feats[last_key][idx_c]
                    feat_k = feats[last_key][clip_idx[i - 1]]

                feat_prop_warped1 = flow_warp(feat_prop.flatten(0, 1),
                                           flow_n1.permute(0, 1, 3, 4, 2).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
                feat_prop_warped2 = flow_warp(feat_prop.flip(1).flatten(0, 1),
                                           flow_n2.permute(0, 1, 3, 4, 2).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
                
                if '_1' in module_name:
                    feat_prop, flow_n1, flow_n2 = self.deform_align[module_name](feat_q, feat_k, feat_prop,
                                                                                [feat_prop_warped1, feat_prop_warped2],
                                                                                [flow_n1, flow_n2],
                                                                                True)
                    updated_flows[f'{module_name}_n1'].append(flow_n1)
                    updated_flows[f'{module_name}_n2'].append(flow_n2)
                else:
                    feat_prop = self.deform_align[module_name](feat_q, feat_k, feat_prop,
                                                            [feat_prop_warped1, feat_prop_warped2],
                                                            [flow_n1, flow_n2],
                                                            False)

            if 'backward' in module_name:
                feat = [feats[k][idx_c].flip(1) for k in feats if k not in [module_name]] + [feat_prop]
            else:
                feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
            
            #print(len(feat), feat[0].shape, feat[1].shape)
            fp = self.backbone[module_name](torch.cat(feat, dim=2))
            #fp = self.backbone[module_name](torch.cat(feat, dim=1))
            feat_prop = feat_prop + fp
            
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
            feats[module_name] = [f.flip(1) for f in feats[module_name]]

        return feats

    def forward(self, lres, hres, *args, **kwargs):
        
        b, f, c, h, w, device, img_size = *hres.shape, hres.device, self.image_size
        
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        topo = hres[:, :, 1:2, :, :]
        hres = hres[:, :, 0:1, :, :]
        topo_low = rearrange(F.interpolate(rearrange(topo, 'b t c h w -> (b t) c h w'), size=(h//8, w//8), mode='bilinear'), '(b t) c h w -> b t c h w', b = b)
        lres = torch.cat([lres, topo_low], dim = 2)

        lres = self.normalize(lres)
        hres = self.normalize(hres)

        flows_forward, flows_backward = self.compute_flow(lres)

        feats = {}
        ff = self.feat_ext(lres)
        
        feats['shallow'] = list(torch.chunk(ff, f // self.clip_size, dim = 1))

        updated_flows = {}
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward if flows_forward is not None else flows_backward.flip(1)

                module_name = f'{direction}_{iter_}'
                feats[module_name] = []
                
                feats = self.propagate(feats, flows, module_name, updated_flows)
        
        feats['shallow'] = torch.cat(feats['shallow'], 1)
        feats['backward_1'] = torch.cat(feats['backward_1'], 1)
        feats['forward_1'] = torch.cat(feats['forward_1'], 1)
        feats['backward_2'] = torch.cat(feats['backward_2'], 1)
        feats['forward_2'] = torch.cat(feats['forward_2'], 1)
        upsampled = torch.cat([feats[k] for k in feats], dim=2)
        upsampled = self.recon(upsampled)
        upsampled = self.feat_up(upsampled)
        upsampled = upsampled + F.interpolate(lres[:,:,0:1,:,:], size = (1, h, w), mode = 'trilinear', align_corners = False)

        loss = self.loss_fn(upsampled, hres, reduction = 'none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
        
        return loss.mean(), upsampled

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_dl,
        val_dl,
        config,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        #augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 1,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1,
        #num_samples = 25,
        eval_folder = './evaluate',
        results_folder = './results',
        #tensorboard_dir = './tensorboard',
        val_num_of_batch = 2,
        amp = False,
        fp16 = False,
        #fp16 = True,
        split_batches = True,
        #split_batches = False,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            log_with = 'wandb',
        )
        self.accelerator.init_trackers("climate", 
            init_kwargs={
                "wandb": {
                    "name":  None,
                }
            },
        )
        self.config = config
        self.accelerator.native_amp = amp
        self.multi = config.data_config["multi"]
        self.rollout = config.rollout
        self.rollout_batch = config.rollout_batch
        #self.flow = config.data_config["flow"]
        self.minipatch = config.data_config["minipatch"]
        self.logscale = config.data_config["logscale"]

        self.model = diffusion_model

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.val_num_of_batch = val_num_of_batch
        
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        
        self.sched = CosineAnnealingLR(self.opt, train_num_steps, 5e-7)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.eval_folder = Path(eval_folder)
        
        self.eval_folder.mkdir(exist_ok=True, parents=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.sched, train_dl, val_dl = self.accelerator.prepare(self.model, self.opt, self.sched, train_dl, val_dl)
        self.train_dl = cycle(train_dl)
        self.val_dl = val_dl

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            #'version': __version__
        }

        torch.save(data, str(self.results_folder / f'qmodel-{milestone%3}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'qmodel-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        #self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        print('loaded')

        #if 'version' in data:
        #    print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):

        accelerator = self.accelerator
        device = accelerator.device

        cmap = mpl.colormaps['RdBu_r']
        fcmap = mpl.colormaps['gray_r']

        # c384_lgmin = np.load('data/only_precip/c384_lgmin.npy')
        # c384_lgmax = np.load('data/only_precip/c384_lgmax.npy')
        # c384_gmin = np.load('data/only_precip/c384_gmin.npy')

        # c48_lgmin = np.load('data/only_precip/c48_lgmin.npy')
        # c48_lgmax = np.load('data/only_precip/c48_lgmax.npy')
        # c48_gmin = np.load('data/only_precip/c48_gmin.npy')

        # c384_min = np.load('data/only_precip/c384_min.npy')
        # c384_max = np.load('data/only_precip/c384_max.npy')

        # c48_min = np.load('data/only_precip/c48_min.npy')
        # c48_max = np.load('data/only_precip/c48_max.npy')

        with open("data/ensemble_c48_trainstats/chl.pkl", 'rb') as f:
            c48_chl = pickle.load(f)
        
        with open("data/ensemble_c48_trainstats/log_chl.pkl", 'rb') as f:
            c48_log_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/chl.pkl", 'rb') as f:
            c384_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/log_chl.pkl", 'rb') as f:
            c384_log_chl = pickle.load(f)

        c384_lgmin = c384_log_chl["PRATEsfc"]['min']
        c384_lgmax = c384_log_chl["PRATEsfc"]['max']
        c48_lgmin = c48_log_chl["PRATEsfc_coarse"]['min']
        c48_lgmax = c48_log_chl["PRATEsfc_coarse"]['max']
        
        c384_min = c384_chl["PRATEsfc"]['min']
        c384_max = c384_chl["PRATEsfc"]['max']
        c48_min = c48_chl["PRATEsfc_coarse"]['min']
        c48_max = c48_chl["PRATEsfc_coarse"]['max']

        c384_gmin = c384_min
        c48_gmin = c48_min

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    
                    data = next(self.train_dl)
                    lres = data['LR'].to(device)
                    hres = data['HR'].to(device)

                    if self.minipatch:
                        
                        x_st = randint(0, 36)
                        y_st = randint(0, 36)
                        lres = crop(lres, x_st, y_st, 12, 12)
                        hres = crop(hres, 8 * x_st, 8 * y_st, 96, 96)

                    with self.accelerator.autocast():
                        
                        loss, _ = self.model(lres, hres)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.log({"loss": total_loss}, step = self.step)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():

                            vlosses = []
                            vids = []
                            hr = []
                            lr = []
                            bases, ress, flowss = [], [], []
                            num_frames = 5
                            img_size = 384
                            
                            for i, batch in enumerate(self.val_dl):
                                
                                lres = batch['LR'].to(device)
                                hres = batch['HR'].to(device)
                                
                                if i >= self.val_num_of_batch:
                                    break

                                loss, videos = self.model(lres, hres)
                                

                                vids.append(videos)
                                vlosses.append(loss)
                                hr.append(hres)
                                lr.append(lres)
                            
                            videos = torch.cat(vids, dim = 0)
                            vloss = torch.stack(vlosses, dim = 0).mean()
                            #self.sched.step(vloss)
                            hres = torch.cat(hr, dim = 0)
                            lres = torch.cat(lr, dim = 0)
                            del vids, vlosses, hr, lr

                            

                            lres = lres[:, :, 0:1, :, :]
                            hres = hres[:, :, 0:1, :, :]

                            if not self.logscale:
                                target = hres[:,:,:,:,:].detach().cpu().numpy() * (c384_max - c384_min) + c384_min
                                output = videos.detach().cpu().numpy() * (c384_max - c384_min) + c384_min
                                coarse = lres[:,:,:,:,:].detach().cpu().numpy() * (c48_max - c48_min) + c48_min
                            
                            else:
                                target = hres[:,:,:,:,:].detach().cpu().numpy() * (c384_lgmax - c384_lgmin) + c384_lgmin
                                output = videos.detach().cpu().numpy() * (c384_lgmax - c384_lgmin) + c384_lgmin
                                coarse = lres[:,:,:,:,:].detach().cpu().numpy() * (c48_lgmax - c48_lgmin) + c48_lgmin
                            
                            if self.logscale:
                                target = np.exp(target) + c384_gmin - 1e-14
                                output = np.exp(output) + c384_gmin - 1e-14
                                coarse = np.exp(coarse) + c48_gmin - 1e-14

                            ssim_index = piq.ssim(torch.from_numpy(target).view(-1, 1, 384, 384), torch.from_numpy(output).view(-1, 1, 384, 384).clamp(0., 1.), data_range=1., reduction='none')
                            gmsd_index = piq.gmsd(torch.from_numpy(target).view(-1, 1, 384, 384), torch.from_numpy(output).view(-1, 1, 384, 384).clamp(0., 1.), data_range=1., reduction='none')

                            nn_upscale = np.repeat(np.repeat(coarse, 8, axis = 3), 8, axis = 4)
                            diff_output = (output - nn_upscale).flatten()
                            diff_target = (target - nn_upscale).flatten()
                            vmin = min(diff_output.min(), diff_target.min())
                            vmax = max(diff_output.max(), diff_target.max())
                            bins = np.linspace(vmin, vmax, 100 + 1)

                            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                            ax.hist(
                                diff_output, bins=bins, alpha=0.5, label="Output", histtype="step", density=True
                            )
                            ax.hist(
                                diff_target, bins=bins, alpha=0.5, label="Target", histtype="step", density=True
                            )
                            ax.set_xlim(vmin, vmax)
                            ax.legend()
                            ax.set_ylabel("Density")
                            ax.set_yscale("log")

                            output1 = output.flatten()
                            target1 = target.flatten()
                            rmse = np.sqrt(np.mean((output1 - target1)**2))
                            pscore = np.abs(np.percentile(output1, 99.999) - np.percentile(target1, 99.999))
                            vmin1 = min(output1.min(), target1.min())
                            vmax1 = max(output1.max(), target1.max())
                            bins1 = np.linspace(vmin1, vmax1, 100 + 1)
                            #histo = np.histogram(output1, bins=bins1, density=True)[0].ravel().astype('float32')
                            #histt = np.histogram(target1, bins=bins1, density=True)[0].ravel().astype('float32')
                            count_o, bin_o = np.histogram(output1, bins=bins1, density=True)
                            count_t, bin_t = np.histogram(target1, bins=bins1, density=True)
                            histo = count_o.ravel().astype('float32')
                            histt = count_t.ravel().astype('float32')
                            distchisqr = cv2.compareHist(histo, histt, cv2.HISTCMP_CHISQR)
                            distinter = cv2.compareHist(histo, histt, cv2.HISTCMP_INTERSECT)
                            distkl = cv2.compareHist(histo, histt, cv2.HISTCMP_KL_DIV)
                            distemd = wasserstein_distance(output1, target1)                            

                            fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
                            ax1.hist(
                                #output1, bins=bins1, alpha=0.5, label="Output", histtype="step", density=True
                                bin_o[:-1], bins=bin_o, weights = count_o, alpha=0.5, label="Output", histtype="step"#, density=True
                            )
                            ax1.hist(
                                #target1, bins=bins1, alpha=0.5, label="Target", histtype="step", density=True
                                bin_t[:-1], bins=bin_t, weights = count_t, alpha=0.5, label="Target", histtype="step"#, density=True
                            )
                            ax1.set_xlim(vmin1, vmax1)
                            ax1.legend()
                            ax1.set_ylabel("Density")
                            ax1.set_yscale("log")
                        
                            if self.logscale:
                                
                                accelerator.log({"true_high": wandb.Video((hres[0:1,:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"true_low": wandb.Video((lres[0:1,:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"samples": wandb.Video((videos.clamp(0.0, 1.0)[0:1,:,0:1,:,:].repeat(1,1,3,1,1).detach().cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                            
                            else:

                                accelerator.log({"true_high": wandb.Video((hres[0:1,:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"true_low": wandb.Video((lres[0:1,:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"samples": wandb.Video((videos[0:1,:,:,:,:].clamp(0.0, 1.0).repeat(1,1,3,1,1).detach().cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                target = np.log(target - c384_gmin + 1e-14)
                                output = np.log(output - c384_gmin + 1e-14)
                                coarse = np.log(coarse - c48_gmin + 1e-14)
                                target = (target - c384_lgmin) / (c384_lgmax - c384_lgmin)
                                output = (output - c384_lgmin) / (c384_lgmax - c384_lgmin)
                                coarse = (coarse - c48_lgmin) / (c48_lgmax - c48_lgmin)
                                accelerator.log({"true_loghigh": wandb.Video((np.repeat(target[0:1,:,:,:,:], 3, axis=-3)*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"true_loglow": wandb.Video((np.repeat(coarse[0:1,:,:,:,:], 3, axis=-3)*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"logsamples": wandb.Video((np.repeat(output[0:1,:,:,:,:], 3, axis=-3)*255).astype(np.uint8))}, step=self.step)

                            accelerator.log({"difference_histogram": wandb.Image(fig, mode = 'RGB')}, step=self.step)
                            accelerator.log({"histogram": wandb.Image(fig1, mode = 'RGB')}, step=self.step)
                            accelerator.log({"ssim": ssim_index.mean()}, step=self.step)
                            accelerator.log({"gmsd": gmsd_index.mean()}, step=self.step)
                            accelerator.log({"rmse": rmse}, step=self.step)
                            accelerator.log({"pscore": pscore}, step=self.step)
                            accelerator.log({"distchisqr": distchisqr}, step=self.step)
                            accelerator.log({"distinter": distinter}, step=self.step)
                            accelerator.log({"distkl": distkl}, step=self.step)
                            accelerator.log({"distemd": distemd}, step=self.step)
                            accelerator.log({"vloss": vloss}, step=self.step)
                            accelerator.log({"lr": self.opt.param_groups[0]['lr']}, step=self.step)
                                
                            milestone = self.step // self.save_and_sample_every
                            
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def sample(self):

        accelerator = self.accelerator
        device = accelerator.device
        
        self.ema.ema_model.eval()

        PATH = "/extra/ucibdl0/shared/data/fv3gfs"
        XX = xr.open_zarr(f"{PATH}/c48_precip_plus_more_ave/0011/sfc_8xdaily_ave_coarse.zarr")
        XX_ = xr.open_zarr(f"{PATH}/c48_atmos_ave/0011/atmos_8xdaily_ave_coarse.zarr")
        yy = xr.open_zarr(f"{PATH}/c384_precip_ave/0011/sfc_8xdaily_ave.zarr")
        topot = xr.open_zarr(f"{PATH}/c384_topo/0011/atmos_static.zarr")

        with open("data/ensemble_c48_trainstats/chl.pkl", 'rb') as f:

            c48_chl = pickle.load(f)
        
        with open("data/ensemble_c48_trainstats/atm_chl.pkl", 'rb') as f:
            
            c48_atm_chl = pickle.load(f)

        with open("data/ensemble_c48_trainstats/log_chl.pkl", 'rb') as f:
            
            c48_log_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/chl.pkl", 'rb') as f:

            c384_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/log_chl.pkl", 'rb') as f:

            c384_log_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/topo.pkl", 'rb') as f:

            c384_topo = pickle.load(f)

        if self.multi:

            c48_channels = ["PRATEsfc_coarse", "UGRD10m_coarse", "VGRD10m_coarse", "TMPsfc_coarse", "CPRATsfc_coarse", "DSWRFtoa_coarse"]
            c48_channels_atmos = ["ps_coarse", "u700_coarse", "v700_coarse", "vertically_integrated_liq_wat_coarse", "vertically_integrated_sphum_coarse"]
            c384_channels = ["PRATEsfc"]

        else:

            c48_channels = ["PRATEsfc_coarse"]
            c384_channels = ["PRATEsfc"]

        with torch.no_grad():

            for tile in range(6):

                if self.rollout == 'full':

                    seq_len = self.rollout_batch
                    st = 0
                    en = seq_len + 2
                    count = 0

                    while en < 3176:

                        print(tile, st)

                        X = XX.isel(time = slice(st, en), tile = tile)
                        X_ = XX_.isel(time = slice(st, en), tile = tile)
                        y = yy.isel(time = slice(st, en), tile = tile)


                        X = np.stack([X[channel].values for channel in c48_channels], axis = 1)
                        X_ = np.stack([X_[channel].values for channel in c48_channels_atmos], axis = 1)
                        y = np.stack([y[channel].values for channel in c384_channels], axis = 1)
                        topo = topot.isel(tile = tile)
                        topo = topo['zsurf'].values
                        topo = np.repeat(topo.reshape((1,1,384,384)), seq_len + 2, axis = 0)
                        
                        X[:,0:1,:,:] = np.log(X[:,0:1,:,:] - c48_chl["PRATEsfc_coarse"]['min'] + 1e-14)
                        y = np.log(y - c384_chl["PRATEsfc"]['min'] + 1e-14)
                        X[:,0:1,:,:] = (X[:,0:1,:,:] - c48_log_chl["PRATEsfc_coarse"]['min']) / (c48_log_chl["PRATEsfc_coarse"]['max'] - c48_log_chl["PRATEsfc_coarse"]['min'])
                        y = (y - c384_log_chl["PRATEsfc"]['min']) / (c384_log_chl["PRATEsfc"]['max'] - c384_log_chl["PRATEsfc"]['min'])

                        for i in range(1, X.shape[1]):

                            X[:,i,:,:] = (X[:,i,:,:] - c48_chl[c48_channels[i]]['min']) / (c48_chl[c48_channels[i]]['max'] - c48_chl[c48_channels[i]]['min'])

                        for i in range(X_.shape[1]):

                            X_[:,i,:,:] = (X_[:,i,:,:] - c48_atm_chl[c48_channels_atmos[i]]['min']) / (c48_atm_chl[c48_channels_atmos[i]]['max'] - c48_atm_chl[c48_channels_atmos[i]]['min'])

                        topo = (topo - c384_topo["zsurf"]['min']) / (c384_topo["zsurf"]['max'] - c384_topo["zsurf"]['min'])

                        X = np.concatenate((X, X_), axis = 1)
                        y = np.concatenate((y, topo), axis = 1)

                        lres = torch.from_numpy(X).unsqueeze(0).to(device)
                        hres = torch.from_numpy(y).unsqueeze(0).to(device)

                        loss, videos = self.model(lres, hres)

                        torch.save(videos, os.path.join(self.eval_folder) + "/gen_{}_{}.pt".format(tile, count))
                        torch.save(hres[:,:,0:1,:,:], os.path.join(self.eval_folder) + "/truth_hr_{}_{}.pt".format(tile, count))
                        torch.save(lres[:,:,0:1,:,:], os.path.join(self.eval_folder) + "/truth_lr_{}_{}.pt".format(tile, count))
                        count += 1

                        st += seq_len
                        en += seq_len

                if self.rollout == 'partial':

                    seq_len = self.rollout_batch
                    #indices = get_random_idx_with_difference(0, 3176 - (seq_len + 2), 75 // seq_len, seq_len + 2) # 75 samples per tile
                    indices = list(range(0, 3176 - (seq_len + 2), 250)) # deterministic, 325 samples per tile for seq_len of 25

                    for count, st in enumerate(indices):

                        print(tile, count)

                        X = XX.isel(time = slice(st, st+(seq_len+2)), tile = tile)
                        X_ = XX_.isel(time = slice(st, st+(seq_len+2)), tile = tile)
                        y = yy.isel(time = slice(st, st+(seq_len+2)), tile = tile)

                        X = np.stack([X[channel].values for channel in c48_channels], axis = 1)
                        X_ = np.stack([X_[channel].values for channel in c48_channels_atmos], axis = 1)
                        y = np.stack([y[channel].values for channel in c384_channels], axis = 1)
                        topo = topot.isel(tile = tile)
                        topo = topo['zsurf'].values
                        topo = np.repeat(topo.reshape((1,1,384,384)), seq_len + 2, axis = 0)

                        X[:,0:1,:,:] = np.log(X[:,0:1,:,:] - c48_chl["PRATEsfc_coarse"]['min'] + 1e-14)
                        y = np.log(y - c384_chl["PRATEsfc"]['min'] + 1e-14)
                        X[:,0:1,:,:] = (X[:,0:1,:,:] - c48_log_chl["PRATEsfc_coarse"]['min']) / (c48_log_chl["PRATEsfc_coarse"]['max'] - c48_log_chl["PRATEsfc_coarse"]['min'])
                        y = (y - c384_log_chl["PRATEsfc"]['min']) / (c384_log_chl["PRATEsfc"]['max'] - c384_log_chl["PRATEsfc"]['min'])

                        for i in range(1, X.shape[1]):

                            X[:,i,:,:] = (X[:,i,:,:] - c48_chl[c48_channels[i]]['min']) / (c48_chl[c48_channels[i]]['max'] - c48_chl[c48_channels[i]]['min'])

                        for i in range(X_.shape[1]):

                            X_[:,i,:,:] = (X_[:,i,:,:] - c48_atm_chl[c48_channels_atmos[i]]['min']) / (c48_atm_chl[c48_channels_atmos[i]]['max'] - c48_atm_chl[c48_channels_atmos[i]]['min'])

                        topo = (topo - c384_topo["zsurf"]['min']) / (c384_topo["zsurf"]['max'] - c384_topo["zsurf"]['min'])

                        X = np.concatenate((X, X_), axis = 1)
                        y = np.concatenate((y, topo), axis = 1)

                        lres = torch.from_numpy(X).unsqueeze(0).to(device)
                        hres = torch.from_numpy(y).unsqueeze(0).to(device)

                        loss, videos = self.model(lres, hres)

                        torch.save(videos, os.path.join(self.eval_folder) + "/gen_{}_{}.pt".format(tile, count))
                        torch.save(hres[:,:,0:1,:,:], os.path.join(self.eval_folder) + "/truth_hr_{}_{}.pt".format(tile, count))
                        torch.save(lres[:,:,0:1,:,:], os.path.join(self.eval_folder) + "/truth_lr_{}_{}.pt".format(tile, count))