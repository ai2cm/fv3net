import os
import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from joblib import Parallel, delayed

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
import wandb

import piq

from kornia import filters
from torch.optim import Adam

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image

import matplotlib as mpl
from matplotlib.cm import ScalarMappable as smap

from tqdm.auto import tqdm
from ema_pytorch import EMA

import flow_vis

from accelerate import Accelerator

from .network_swinir import SwinIR as net

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def calculate_crps(truth, pred, num_samples, num_videos_per_batch, num_frames, img_channels, img_size):
    truth_cdf = np.zeros((256, 1, num_videos_per_batch, num_frames, img_channels, img_size, img_size), dtype = 'uint8')
    for i in range(256):
        truth_cdf[i, :, :, :, :, :, :] = (truth <= i).astype('uint8')
    pred_cdf = np.zeros((256, num_samples, 1, num_videos_per_batch, num_frames, img_channels, img_size, img_size), dtype = 'uint8')
    for j in range(256):
        pred_cdf[j, :, :, :, :, :, :, :] = (pred <= j).astype('uint8')
    red_pred_cdf = pred_cdf.mean(1)
    temp = np.square(red_pred_cdf - truth_cdf)
    temp_dz = temp.sum(0)
    temp_dz_dd = temp_dz.mean(axis = (3, 4, 5))
    temp_dz_dd_dt = temp_dz_dd.mean(2)
    return temp_dz_dd_dt.mean()

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

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# ssf modules

def gaussian_pyramids(input, base_sigma = 1, m = 5):
    
    output = [input]
    N, C, H, W = input.shape
    
    kernel = filters.get_gaussian_kernel2d((5, 5), (base_sigma, base_sigma))#.unsqueeze(0)
    
    for i in range(m):
        
        input = filters.filter2d(input, kernel)
        
        if i == 0:
            
            output.append(input)
        
        else:
            
            tmp = input
            
            for j in range(i):
                
                tmp = F.interpolate(tmp, scale_factor = 2., mode = 'bilinear', align_corners = True)
                
            output.append(tmp)
        
        input = F.interpolate(input, scale_factor = 0.5)
    
    return torch.stack(output, 2)

def scale_space_warp(input, flow):
    
    N, C, H, W = input.shape
    
    assert flow.shape == (N, 3, H, W)
    
    flow = flow.unsqueeze(0)
    #multi_scale = gaussian_pyramids(input, self.base_scale, self.gaussian_dim)
    multi_scale = gaussian_pyramids(input, 1.0, 5)
    
    h = torch.arange(H, device=input.device, dtype=input.dtype)
    w = torch.arange(W, device=input.device, dtype=input.dtype)
    d = torch.zeros(1, device=input.device, dtype=input.dtype)
    
    grid = torch.stack(torch.meshgrid(d, h, w)[::-1], -1).unsqueeze(0)
    grid = grid.expand(N, -1, -1, -1, -1)
    flow = flow.permute(1, 0, 3, 4, 2)  # N, 1, H, W, 3

    # reparameterization
    # var_channel = (flow[..., -1].exp())**2
    # var_space = [0.] + [(2.**i * self.base_scale)**2 for i in range(self.gaussian_dim)]
    # d_offset = var_to_position(var_channel, var_space).unsqueeze(-1)
    d_offset = flow[..., -1].clamp(min=-1.0, max=1.0).unsqueeze(-1)

    flow = torch.cat((flow[..., :2], d_offset), -1)
    flow_grid = flow + grid
    flow_grid[..., 0] = 2.0 * flow_grid[..., 0] / max(W - 1.0, 1.0) - 1.0
    flow_grid[..., 1] = 2.0 * flow_grid[..., 1] / max(H - 1.0, 1.0) - 1.0
    
    warped = F.grid_sample(multi_scale, flow_grid, padding_mode = "border", align_corners = True).squeeze(2)
    
    return warped

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(2*dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(2*dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, context, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        count = 0
        
        for block1, block2, attn, downsample in self.downs:
            x = torch.cat((x, context[count]), dim = 1)
            count += 1
            x = block1(x, t)
            h.append(x)

            x = torch.cat((x, context[count]), dim = 1)
            count += 1
            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
class Flow(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x):

        x = self.init_conv(x)
        r = x.clone()

        h = []
        context = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)
            context.append(x)
            x = block2(x)
            x = attn(x)
            h.append(x)
            context.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)
        return self.final_conv(x), context

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        flow,
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
        super().__init__()
        #assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        #assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        
        self.umodel = net(upscale=8, in_chans=1, img_size=48, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6], embed_dim=200,
        num_heads=[8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='3conv')
        
        self.flow = flow
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)
        
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    #def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
    def model_predictions(self, x, t, l_cond, context, x_self_cond = None, clip_x_start = False):
        
        #model_output = self.model(x, t, x_self_cond)
        #print(x.shape, l_cond.shape)
        model_output = self.model(torch.cat((x, l_cond), 1), t, context, x_self_cond)
        
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    #def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
    def p_mean_variance(self, x, t, context, x_self_cond = None, clip_denoised = True):
        
        #preds = self.model_predictions(x, t, x_self_cond)
        preds = self.model_predictions(x, t, context, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    #def p_sample(self, x, t: int, x_self_cond = None):
    def p_sample(self, x, t: int, context, x_self_cond = None):
        
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        #model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, context = context, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    #def p_sample_loop(self, shape, return_all_timesteps = False):
    def p_sample_loop(self, shape, context, return_all_timesteps = False):
        
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None
        
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            #img, x_start = self.p_sample(img, t, self_cond)
            img, x_start = self.p_sample(img, t, context, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        #ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    #def ddim_sample(self, shape, return_all_timesteps = False):
    def ddim_sample(self, shape, l_cond, context, return_all_timesteps = False):
        print('here!!!')
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            #pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, l_cond, context, self_cond, clip_x_start = True)

            imgs.append(img)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        
        imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        #ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, lres, return_all_timesteps = False):
        
        b, f, c, h, w = lres.shape
        
        ures = self.umodel(rearrange(lres, 'b t c h w -> (b t) c h w'))
        ures = rearrange(ures, '(b t) c h w -> b t c h w', b = b)
        
        lres = self.normalize(lres)
        ures = self.normalize(ures)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        l = ures.clone()
        r = torch.roll(l, -1, 1)
        ures_flow = rearrange(ures[:, 1:(f-1), :, :, :], 'b t c h w -> (b t) c h w')
        l_cond = self.upsample(rearrange(lres[:, 2:, :, :, :], 'b t c h w -> (b t) c h w'))
        
        m = lres.clone()
        m1 = rearrange(m, 'b t c h w -> (b t) c h w')
        m1 = self.upsample(m1)
        m1 = rearrange(m1, '(b t) c h w -> b t c h w', t = f)
        m1 = torch.roll(m1, -2, 1)
        
        stack = torch.cat((l, r, m1), 2)
        stack = stack[:, :(f-2), :, :, :]
        stack = rearrange(stack, 'b t c h w -> (b t) c h w')
        
        flow, context = self.flow(stack)
        
        warped = scale_space_warp(ures_flow, flow)
        
        res = sample_fn((b*(f-2),c,8*h,8*w), l_cond, context, return_all_timesteps = return_all_timesteps)
        sres = warped + res
        sres = rearrange(sres, '(b t) c h w -> b t c h w', b = b)

        warped = rearrange(warped, '(b t) c h w -> b t c h w', b = b)
        res = rearrange(res, '(b t) c h w -> b t c h w', b = b)
        flow = rearrange(flow, '(b t) c h w -> b t c h w', b = b)
        
        return self.unnormalize(sres), self.unnormalize(warped), self.unnormalize(res), self.unnormalize(flow)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    #def p_losses(self, x_start, t, noise = None):
    def p_losses(self, stack, hres, lres, ures, t, noise = None):
        
        b, f, c, h, w = hres.shape
        
        stack = rearrange(stack, 'b t c h w -> (b t) c h w')
        ures_flow = rearrange(ures[:, 1:(f-1), :, :, :], 'b t c h w -> (b t) c h w')
        
        flow, context = self.flow(stack)
        
        warped = scale_space_warp(ures_flow, flow)
        
        x_start = rearrange(hres[:, 2:, :, :, :], 'b t c h w -> (b t) c h w')
        x_start = x_start - warped
        
        l_cond = self.upsample(rearrange(lres[:, 2:, :, :, :], 'b t c h w -> (b t) c h w'))
        
        b, c, h, w = x_start.shape
        
        del f
        
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        
        model_out = self.model(torch.cat((x, l_cond), 1), t, context, x_self_cond)
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        loss1 = self.loss_fn(ures, hres, reduction = 'none')
        loss1 = reduce(loss1, 'b ... -> b (...)', 'mean')

        loss2 = self.loss_fn(x_start, warped, reduction = 'none')
        loss2 = reduce(loss2, 'b ... -> b (...)', 'mean')

        return loss.mean() + loss1.mean() + loss2.mean()

    #def forward(self, data, *args, **kwargs):
    def forward(self, lres, hres, *args, **kwargs):
        
        #b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        b, f, c, h, w, device, img_size = *hres.shape, hres.device, self.image_size
        
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        
        #t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        t = torch.randint(0, self.num_timesteps, (b*(f-2),), device=device).long()

        ures = self.umodel(rearrange(lres, 'b t c h w -> (b t) c h w'))
        ures = rearrange(ures, '(b t) c h w -> b t c h w', b = b)
        
        #img = self.normalize(img)
        lres = self.normalize(lres)
        hres = self.normalize(hres)
        ures = self.normalize(ures)
        
        l = ures.clone()
        r = torch.roll(l, -1, 1)

        m = lres.clone()
        m1 = rearrange(m, 'b t c h w -> (b t) c h w')
        m1 = self.upsample(m1)
        #m1 = rearrange(m1, '(b t) c h w -> b t c h w', t = 7)
        m1 = rearrange(m1, '(b t) c h w -> b t c h w', b = b)
        m1 = torch.roll(m1, -2, 1)

        stack = torch.cat((l, r, m1), 2)
        stack = stack[:, :(f-2), :, :, :]
        
        #return self.p_losses(img, t, *args, **kwargs)
        return self.p_losses(stack, hres, lres, ures, t, *args, **kwargs)

# trainer class

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
        save_and_sample_every = 10,
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
        self.accelerator.init_trackers("vsr-orig-autoreg-hres", 
            init_kwargs={
                "wandb": {
                    "notes": "Use VSR to improve precipitation forecasting.",
                    # Change "name" to set the name of the run.
                    "name":  None,
                }
            },
        )
        self.config = config
        self.accelerator.native_amp = amp

        self.model = diffusion_model

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.val_num_of_batch = val_num_of_batch
        
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(exist_ok=True, parents=True)
        
        self.eval_folder = eval_folder

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, train_dl, val_dl = self.accelerator.prepare(self.model, self.opt, train_dl, val_dl)
        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)        

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

        #if 'version' in data:
        #    print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):

        accelerator = self.accelerator
        device = accelerator.device

        cmap = mpl.colormaps['RdBu_r']
        

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    
                    #data = next(self.dl).to(device)
                    data = next(self.train_dl)
                    lres = data['LR'].to(device)
                    hres = data['HR'].to(device)

                    with self.accelerator.autocast():
                        
                        #loss = self.model(data)
                        loss = self.model(lres, hres)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                #self.writer.add_scalar("loss", total_loss, self.step)
                accelerator.log({"loss": total_loss}, step = self.step)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            
                            for i, batch in enumerate(self.val_dl):
                                
                                lres = batch['LR'].to(device)
                                hres = batch['HR'].to(device)
                                
                                if i >= self.val_num_of_batch:
                                    break

                                num_samples = 5
                                num_videos_per_batch = 2
                                num_frames = 5
                                img_size = 384
                                img_channels = 1

                                truth = np.zeros((1, num_videos_per_batch, num_frames, img_channels, img_size, img_size), dtype = 'uint8')
                                pred = np.zeros((num_samples, 1, num_videos_per_batch, num_frames, img_channels, img_size, img_size), dtype = 'uint8')
                                truth[0,:,:,:,:,:] = (hres[:,2:,:,:,:].repeat(1,1,1,1,1).cpu().numpy()*255).astype(np.uint8)
                                for k in range(num_samples):
                                    videos, base, res, flows = self.ema.ema_model.sample(lres)
                                    pred[k,0,:,:,:,:] = (videos.clamp(0.0, 1.0)[:,:,0:1,:,:].repeat(1,1,1,1,1).detach().cpu().numpy()*255).astype(np.uint8)
                                
                                crps_index = calculate_crps(truth, pred, num_samples, num_videos_per_batch, num_frames, img_channels, img_size)
                                psnr_index = piq.psnr(hres[:,2:,0:1,:,:], videos.clamp(0.0, 1.0)[:,:,0:1,:,:], data_range=1., reduction='none')

                                videos_time_mean = videos.mean(dim = 1)
                                hres_time_mean = hres[:,2:,:,:,:].mean(dim = 1)
                                bias = videos_time_mean - hres_time_mean

                                norm = mpl.colors.Normalize(vmin = bias.min(), vmax = bias.max())
                                sm = smap(norm, cmap)
                                bias_color = sm.to_rgba(bias[:,0,:,:].cpu().numpy())

                                nn_upscale = F.interpolate(lres[:,2:,:,:,:], scale_factor = 8, mode = 'nearest-exact')
                                diff_output = torch.flatten(videos - nn_upscale).detach().cpu().numpy()
                                diff_target = torch.flatten(hres[:,2:,:,:,:] - nn_upscale).detach().cpu().numpy()
                                vmin = min(diff_output.min(), diff_target.min())
                                vmax = max(diff_output.max(), diff_target.max())
                                bins = np.linspace(vmin, vmax, 100 + 1)
                                hist = np.histogram(diff_output, bins = bins)[0]
                                
                                accelerator.log({"true_high": wandb.Video((hres[:,2:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"true_low": wandb.Video((lres[:,2:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"pred": wandb.Video((base.clamp(0.0, 1.0)[:,:,0:1,:,:].repeat(1,1,3,1,1).detach().cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"samples": wandb.Video((videos.clamp(0.0, 1.0)[:,:,0:1,:,:].repeat(1,1,3,1,1).detach().cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"res": wandb.Video((res.clamp(0.0, 1.0)[:,:,0:1,:,:].repeat(1,1,3,1,1).detach().cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"flows": wandb.Video((flows.clamp(0.0, 1.0).detach().cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"pattern_bias": wandb.Image((bias_color*255).astype(np.uint8), mode = 'RGBA')}, step=self.step)
                                accelerator.log({"difference_histogram": wandb.Histogram(np_histogram = hist)}, step=self.step)
                                accelerator.log({"psnr": psnr_index.mean()}, step=self.step)
                                accelerator.log({"crps": crps_index}, step=self.step)
                                
                            milestone = self.step // self.save_and_sample_every
                            
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def sample(self):
        
        accelerator = self.accelerator
        device = accelerator.device
        
        self.ema.ema_model.eval()

        cmap = mpl.colormaps['viridis']
        sm = smap(None, cmap)

        with torch.no_grad():

            for k, batch in enumerate(self.val_dl):

                lres = batch['LR'].to(device)
                hres = batch['HR'].to(device)
                
                if k >= self.val_num_of_batch:
                    break
                
                limit = lres.shape[1]
                if limit < 8:
                    
                    #videos, base, nsteps, flows = self.ema.ema_model.sample(lres, hres, True)
                    videos, base, nsteps, flows = self.ema.ema_model.sample(lres, hres)

                    torch.save(videos, os.path.join(self.eval_folder) + "/gen.pt")
                    torch.save(hres[:,2:,:,:,:], os.path.join(self.eval_folder) + "/truth_hr.pt")
                    torch.save(lres[:,2:,:,:,:], os.path.join(self.eval_folder) + "/truth_lr.pt")

                    for i, b in enumerate(videos.clamp(0, 1)):
                        if not os.path.isdir(os.path.join(self.eval_folder, "generated")):
                            os.makedirs(os.path.join(self.eval_folder, "generated"))
                        Parallel(n_jobs=4)(
                            delayed(save_image)(sm.to_rgba(f[0,:,:]), os.path.join(self.eval_folder, "generated") + f"/{k}-{i}-{j}.png")
                            for j, f in enumerate(b.cpu())
                        )

                    #videos = torch.log(videos.clamp(0.0, 1.0) + 1)
                    #hres = torch.log(hres + 1)
                
                    #for i, b in enumerate(videos.clamp(0, 1)):
                    # for i, b in enumerate(videos):
                    #     if not os.path.isdir(os.path.join(self.eval_folder, "generated")):
                    #         os.makedirs(os.path.join(self.eval_folder, "generated"))
                    #     Parallel(n_jobs=4)(
                    #         delayed(save_image)(f, os.path.join(self.eval_folder, "generated") + f"/{k}-{i}-{j}.png")
                    #         for j, f in enumerate(b.cpu())
                    #     )
                          
#                     for i, b in enumerate(nsteps.clamp(0, 1)):
#                     #for i, b in enumerate(sampled):   
#                         if not os.path.isdir(os.path.join(self.eval_folder, "residual")):
#                             os.makedirs(os.path.join(self.eval_folder, "residual"))
#                         Parallel(n_jobs=4)(
#                             delayed(save_image)(f, os.path.join(self.eval_folder, "residual") + f"/{k}-{i}-{j}.png")
#                             for j, f in enumerate(b.cpu())
#                         )
                          
#                     for i, b in enumerate(base.clamp(0, 1)):
#                     #for i, b in enumerate(sampled):   
#                         if not os.path.isdir(os.path.join(self.eval_folder, "warped")):
#                             os.makedirs(os.path.join(self.eval_folder, "warped"))
#                         Parallel(n_jobs=4)(
#                             delayed(save_image)(f, os.path.join(self.eval_folder, "warped") + f"/{k}-{i}-{j}.png")
#                             for j, f in enumerate(b.cpu())
#                         )
                        
#                     for i, b in enumerate(flows.clamp(0, 1)):
#                     #for i, b in enumerate(sampled):   
#                         if not os.path.isdir(os.path.join(self.eval_folder, "flows")):
#                             os.makedirs(os.path.join(self.eval_folder, "flows"))
#                         Parallel(n_jobs=4)(
#                             delayed(save_image)(f, os.path.join(self.eval_folder, "flows") + f"/{k}-{i}-{j}.png")
#                             for j, f in enumerate(b.cpu())
#                         )
                
                    for i, b in enumerate(hres[:,2:,:,:,:].clamp(0, 1)):
                        if not os.path.isdir(os.path.join(self.eval_folder, "truth")):
                            os.makedirs(os.path.join(self.eval_folder, "truth"))
                        Parallel(n_jobs=4)(
                            delayed(save_image)(sm.to_rgba(f[0,:,:]), os.path.join(self.eval_folder, "truth") + f"/{k}-{i}-{j}.png")
                            for j, f in enumerate(b.cpu())
                        )
                
#                 else:
                    
#                     videos, base, nsteps, flows = self.ema.ema_model.sample(lres[:,:7,:,:], hres[:,:7,:,:], True)
                    
#                     st = 5
#                     ed = st + 7
                    
#                     while ed < limit:
                        
#                         vi, ba, ns, fl = self.ema.ema_model.sample(lres[:,st:ed,:,:], hres[:,st:ed,:,:], True)
#                         st += 5
#                         ed += 5
#                         videos = torch.cat((videos, vi), 1)
#                         #base = torch.cat((base, ba), 1)
#                         #nsteps = torch.cat((nsteps, ns), 1)
#                         #flows = torch.cat((flows, fl), 1)
                    
#                     for i, b in enumerate(videos.clamp(0, 1)):
#                     #for i, b in enumerate(sampled):   
#                         if not os.path.isdir(os.path.join(self.eval_folder, "generated")):
#                             os.makedirs(os.path.join(self.eval_folder, "generated"))
#                         Parallel(n_jobs=4)(
#                             delayed(save_image)(f, os.path.join(self.eval_folder, "generated") + f"/{k}-{i}-{j}.png")
#                             for j, f in enumerate(b.cpu())
#                         )
                
#                     for i, b in enumerate(hres[:,2:,:,:,:].clamp(0, 1)):
#                     #for i, b in enumerate(sampled):   
#                         if not os.path.isdir(os.path.join(self.eval_folder, "truth")):
#                             os.makedirs(os.path.join(self.eval_folder, "truth"))
#                         Parallel(n_jobs=4)(
#                             delayed(save_image)(f, os.path.join(self.eval_folder, "truth") + f"/{k}-{i}-{j}.png")
#                             for j, f in enumerate(b.cpu())
#                         )
                        
# #                     for i, b in enumerate(nsteps.clamp(0, 1)):
# #                     #for i, b in enumerate(sampled):   
# #                         if not os.path.isdir(os.path.join(self.eval_folder, "residual")):
# #                             os.makedirs(os.path.join(self.eval_folder, "residual"))
# #                         Parallel(n_jobs=4)(
# #                             delayed(save_image)(f, os.path.join(self.eval_folder, "residual") + f"/{k}-{i}-{j}.png")
# #                             for j, f in enumerate(b.cpu())
# #                         )
                          
# #                     for i, b in enumerate(base.clamp(0, 1)):
# #                     #for i, b in enumerate(sampled):   
# #                         if not os.path.isdir(os.path.join(self.eval_folder, "warped")):
# #                             os.makedirs(os.path.join(self.eval_folder, "warped"))
# #                         Parallel(n_jobs=4)(
# #                             delayed(save_image)(f, os.path.join(self.eval_folder, "warped") + f"/{k}-{i}-{j}.png")
# #                             for j, f in enumerate(b.cpu())
# #                         )
                        
# #                     for i, b in enumerate(flows.clamp(0, 1)):
# #                     #for i, b in enumerate(sampled):   
# #                         if not os.path.isdir(os.path.join(self.eval_folder, "flows")):
# #                             os.makedirs(os.path.join(self.eval_folder, "flows"))
# #                         Parallel(n_jobs=4)(
# #                             delayed(save_image)(f, os.path.join(self.eval_folder, "flows") + f"/{k}-{i}-{j}.png")
# #                             for j, f in enumerate(b.cpu())
# #                         )
                        
#                     for i, b in enumerate(flows.clamp(0, 1)):
#                     #for i, b in enumerate(sampled):   
#                         if not os.path.isdir(os.path.join(self.eval_folder, "flows_d")):
#                             os.makedirs(os.path.join(self.eval_folder, "flows_d"))
#                         Parallel(n_jobs=4)(
#                             delayed(plt.imsave)(os.path.join(self.eval_folder, "flows_d") + f"/{k}-{i}-{j}.png", flow_vis.flow_to_color(f.permute(1,2,0).cpu().numpy()[:,:,:2], convert_to_bgr = False))
#                             for j, f in enumerate(b.cpu())
#                         )
#                     for i, b in enumerate(flows.clamp(0, 1)):
#                     #for i, b in enumerate(sampled):   
#                         if not os.path.isdir(os.path.join(self.eval_folder, "flows_s")):
#                             os.makedirs(os.path.join(self.eval_folder, "flows_s"))
#                         Parallel(n_jobs=4)(
#                             delayed(plt.imsave)(os.path.join(self.eval_folder, "flows_s") + f"/{k}-{i}-{j}.png", f.permute(1,2,0).cpu().numpy()[:,:,2], cmap = 'gray_r')
#                             for j, f in enumerate(b.cpu())
#                         )