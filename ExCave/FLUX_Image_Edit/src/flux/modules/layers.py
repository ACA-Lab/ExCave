import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope, attention_double, apply_rope

import os
import numpy as np
torch.set_printoptions(threshold=np.inf)
import time


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, knormend_event=None) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        if knormend_event is not None:
            knormend_event.wait(torch.cuda.current_stream())
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, info, txt_index=-1, mask_tmp=None, pe_q_tmp1=None, h_tmp=48, w_tmp=85, memory_stream=None, knorm_stream=None, vnorm_stream=None, kvload_stream=None, kvstore_stream=None, compute_stream=None, maskgen_stream=None, crossattn_stream=None, kv_load_list=None, kvstore_prev=None) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2) #[8, 24, 512, 128] + [8, 24, 900, 128] -> [8, 24, 1412, 128]
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        # import pdb;pdb.set_trace()
        attn = attention_double(q, k, v, pe=pe, info=info)
 
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img_proj = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img_res = img_proj + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img_proj) + img_mod2.shift)

        # calculate the txt bloks
        txt_proj = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt_res = txt_proj + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt_proj) + txt_mod2.shift)
        
        # return img, txt
        return img_res, txt_res, info, mask_tmp, pe_q_tmp1, kv_load_list, kvstore_prev


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        self.input_cache = list()
        self.input_mask = None

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, info, txt_index=-1, mask_tmp=None, pe_q_tmp1=None, h_tmp=48, w_tmp=85, memory_stream=None, knorm_stream=None, vnorm_stream=None, kvload_stream=None, kvstore_stream=None, compute_stream=None, maskgen_stream=None, crossattn_stream=None, kv_load_list=None, kvstore_prev=None) -> Tensor:
        x_event = torch.cuda.Event()
        kv_event = torch.cuda.Event()
        kvtmp_event = torch.cuda.Event()
        kvcache_event = torch.cuda.Event()
        res_event = torch.cuda.Event()
        restmp_event = torch.cuda.Event()
        rescache_event = torch.cuda.Event()
        crossattn_event = torch.cuda.Event()
        delayload_event = torch.cuda.Event()
        delaystore_event = torch.cuda.Event()
        knormstart_event = torch.cuda.Event()
        vnormstart_event = torch.cuda.Event()
        knormend1_event = torch.cuda.Event()
        knormend2_event = torch.cuda.Event()
        vnormend_event = torch.cuda.Event()
        torch.cuda.synchronize()
        with torch.cuda.stream(compute_stream):
            k_cache = 0
            v_cache = 0
            
            with torch.cuda.stream(memory_stream):
                if not info['inverse']:
                    if (info['id'] == 0) and (not info['second_order']):#######only the first block of the first order needs to generate mask 
                        x_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'x'
                        pre_x = info['feature'][x_feature_name].cuda(non_blocking=True)
                        x_event.record(stream=memory_stream)
                        if info['t'] < 0.97:
                            info['feature'][x_feature_name] = (x[:, 512:, :]).to('cpu', non_blocking=True)
                        else:
                            info['feature'][x_feature_name] = (x[:, 512:, :]).cuda(non_blocking=True)
                    if (info['id'] == 37):
                        res_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'res'
                        res_cache = info['feature'][res_feature_name].cuda(non_blocking=True)
                        rescache_event.record(memory_stream)

            with torch.cuda.stream(kvload_stream):
                if ((info['id'] == 0) and (info['second_order']) and (not info['inverse'])):
                    kv_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'KV'
                    kv_cache = info['feature'][kv_feature_name].cuda(non_blocking=True)
                    # v_cache = info['feature'][v_feature_name].cuda(non_blocking=True)
                    batch = kv_cache.shape[0]
                    k_cache, v_cache = torch.split(kv_cache, [batch // 2, batch // 2], dim=0)
                    if info['inject'] and info['id'] > 19:
                        v_cache_tmp = v_cache.clone()
                    else:
                        v_cache_tmp = None
                    kvcache_event.record(kvload_stream)

                    kv_feature_name_next = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']+1) + '_' + info['type'] + '_' + 'KV'
                    kv_cache_next = info['feature'][kv_feature_name_next].cuda(non_blocking=True)
                    # v_cache = info['feature'][v_feature_name].cuda(non_blocking=True)
                    batch = kv_cache_next.shape[0]
                    k_cache_next, v_cache_next = torch.split(kv_cache_next, [batch // 2, batch // 2], dim=0)
                    if info['inject'] and info['id']+1 > 19:
                        v_cache_tmp_next = v_cache_next.clone()
                    else:
                        v_cache_tmp_next = None
                    kv_load_next_list = [k_cache_next, v_cache_next, v_cache_tmp_next]
                else:
                    k_cache_next = None
                    v_cache_next = None
                    v_cache_tmp_next = None
                    kv_load_next_list = [k_cache_next, v_cache_next, v_cache_tmp_next]

            with torch.cuda.stream(memory_stream):
                x_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'x'
                if (info['id'] == 0) and (info['inverse']) and (not info['second_order']):
                    if info['t'] < 0.97:
                        info['feature'][x_feature_name] = (x[:, 512:, :]).to('cpu', non_blocking=True)
                    else:
                        info['feature'][x_feature_name] = (x[:, 512:, :]).cuda(non_blocking=True)

                    input_mask = None
                # else:
            with torch.cuda.stream(maskgen_stream):
                x_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'x'
                if (info['id'] == 0) and (not info['inverse']) and (not info['second_order']):
                    # pre_x = info['feature'][x_feature_name].cuda()###############
                    x_event.wait(stream=maskgen_stream)
                        
                    input_delta = (x[:, 512:, :] - pre_x).abs().sum(-1).float()
                
                    attn_cross1 = input_delta
                    attn_cross1 = attn_cross1.reshape(h_tmp, w_tmp)
                    
                    attn_cross1 = (attn_cross1 - attn_cross1.min()) / (attn_cross1.max() - attn_cross1.min())
                    if info['t'] >= 0.90:################
                        x_bias = 0.003
                    elif info['t'] >= 0.80:
                        x_bias = 0.003
                    else:
                        x_bias = 0.020
                    input_mask = (attn_cross1 <= x_bias)

                    self.input_mask = input_mask

            mod, _ = self.modulation(vec)
            x_dimension = x.shape[-1]
            if not info['inverse']:
                if (info['id'] > 0) or (info['second_order']):
                    if mask_tmp == None:
                        raise ValueError("Didn't get correct mask_tmp, it's None")
                    mask = mask_tmp
                if ((info['id'] == 1) and (not info['second_order'])) or ((info['id'] == 0) and (info['second_order'])):
                    x_selected = torch.cat((x[:, :512, :], x[:, 512:, :][~mask].reshape(1, -1, x_dimension)), dim=1)
                    x_tmp = x_selected
                else:
                    x_tmp = x
            else:
                x_tmp = x
            
            x_mod = (1 + mod.scale) * self.pre_norm(x_tmp) + mod.shift
            if (info['id'] < 37) and (not info['inverse']) and ((not info['second_order']) or (info['second_order'] and (info['id'] > 0))):
                delayload_event.record(compute_stream)
            if (info['id'] < 37) and (not info['inverse']) and (((info['id'] > 1) and (not info['second_order'])) or ((info['id'] > 0) and (info['second_order']))):
                delaystore_event.record(compute_stream)

            with torch.cuda.stream(kvload_stream):
                if (info['id'] < 37) and (not info['inverse']) and ((not info['second_order']) or (info['second_order'] and (info['id'] > 0))):
                    delayload_event.wait(kvload_stream)
                    kv_feature_name_next = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']+1) + '_' + info['type'] + '_' + 'KV'
                    # v_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'V'
                    kv_cache_next = info['feature'][kv_feature_name_next].cuda(non_blocking=True)
                    # v_cache = info['feature'][v_feature_name].cuda(non_blocking=True)
                    batch = kv_cache_next.shape[0]
                    k_cache_next, v_cache_next = torch.split(kv_cache_next, [batch // 2, batch // 2], dim=0)
                    if info['inject'] and info['id']+1 > 19:
                        v_cache_tmp_next = v_cache_next.clone()
                    else:
                        v_cache_tmp_next = None
                    # kvcache_event.record(memory_stream)
                    kv_load_next_list = [k_cache_next, v_cache_next, v_cache_tmp_next]

            with torch.cuda.stream(kvstore_stream):
                if (not info['inverse']):
                    if ((info['id'] > 1) and (not info['second_order'])) or ((info['id'] > 0) and (info['second_order'])):
                        # with torch.cuda.stream(memory_stream):
                        if info['id'] < 37:
                            delaystore_event.wait(kvstore_stream)
                            kv_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']-1) + '_' + info['type'] + '_' + 'KV'
                            # kvtmp_event.wait(memory_stream)
                            info['feature'][kv_feature_name] = kvstore_prev.to('cpu', non_blocking=True)

            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            kv_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'KV'
            kv_tmp = None
            if info['inverse']:
                if (info['id'] > 0) or (info['second_order']):
                    kv = torch.cat((k[:, :, 512:, :], v[:, :, 512:, :]), dim=0)
                    kv_event.record(compute_stream)
            else:
                if (info['id'] > 0) or (info['second_order']):
                    if ((info['id'] == 0) and (info['second_order'])):
                        kvcache_event.wait(compute_stream)
                    else:
                        if kv_load_list is None:
                            raise ValueError("Didn't get correct kv_load_list, it's None")
                        k_cache = kv_load_list[0]
                        v_cache = kv_load_list[1]
                        v_cache_tmp = kv_load_list[2]
                    knormstart_event.record(compute_stream)
                    vnormstart_event.record(compute_stream)
            
            with torch.cuda.stream(knorm_stream):
                if ((info['id'] > 0) or (info['second_order'])) and (not info['inverse']):
                    knormstart_event.wait(knorm_stream)
                    if mask is None:
                        raise ValueError("Didn't get correct kv mask, it's None")
                    k_cache.masked_scatter_(~(mask.reshape(1, mask.shape[1], self.num_heads, -1).transpose(1, 2)), k[:, :, 512:, :])
                    k_cache = torch.cat((k[:, :, :512, :], k_cache), dim=-2)
                    k = k_cache
                    knormend1_event.record(knorm_stream)
                    knormend2_event.record(knorm_stream)

            with torch.cuda.stream(vnorm_stream):
                if ((info['id'] > 0) or (info['second_order'])) and (not info['inverse']):
                    vnormstart_event.wait(vnorm_stream)
                    if mask is None:
                        raise ValueError("Didn't get correct kv mask, it's None")
                    v_cache.masked_scatter_(~(mask.reshape(1, mask.shape[1], self.num_heads, -1).transpose(1, 2)), v[:, :, 512:, :])
                    v_cache = torch.cat((v[:, :, :512, :], v_cache), dim=-2)
                    if info['inject'] and info['id'] > 19:
                        # v = v_cache_tmp
                        v = torch.cat((v[:, :, :512, :], v_cache_tmp), dim=-2)
                    else:
                        v = v_cache
                    knormend2_event.wait(vnorm_stream)
                    kv_tmp = torch.cat((k_cache[:, :, 512:, :], v_cache[:, :, 512:, :]), dim=0)
                    if info['id'] == 37:
                        kvtmp_event.record(compute_stream)
                    vnormend_event.record(vnorm_stream)

            if ((info['id'] > 0) or (info['second_order'])) and (not info['inverse']):
                q = self.norm.query_norm(q)
                q = q.to(v)
                knormend1_event.wait(compute_stream)
                k = self.norm.key_norm(k)
                k = k.to(v)
            else:
                q, k = self.norm(q, k, v, None)
            if info['inverse']:
                q, k = apply_rope(q, k, pe, None)
                attn, attn_map = attention(q, k, v, info=info, crossattn_event=None)
                # pe_q_out = None
            else:
                if (info['id'] == 0) and (not info['second_order']):
                    q, k = apply_rope(q, k, pe, None)
                    attn, attn_map = attention(q, k, v, info=info, crossattn_event=crossattn_event)
                else:
                    q, k = apply_rope(q, k, pe, pe_q_tmp1)
                    vnormend_event.wait(compute_stream)
                    attn, attn_map = attention(q, k, v, info=info, crossattn_event=None)
                # pe_q_out = pe_q

            with torch.cuda.stream(memory_stream):
                if (info['id'] > 0) or (info['second_order']):
                    kv_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'KV'
                    if info['inverse']:
                        # kv = torch.cat((k[:, :, 512:, :], v[:, :, 512:, :]), dim=0)
                        kv_event.wait(memory_stream)
                        info['feature'][kv_feature_name] = kv.to('cpu', non_blocking=True)
            
            with torch.cuda.stream(kvstore_stream):
                if (not info['inverse']):
                    if ((info['id'] > 1) and (not info['second_order'])) or ((info['id'] > 0) and (info['second_order'])):
                        # with torch.cuda.stream(memory_stream):
                        if info['id'] == 37:
                            kv_feature_prev_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']-1) + '_' + info['type'] + '_' + 'KV'
                            # kvtmp_event.wait(memory_stream)
                            info['feature'][kv_feature_prev_name] = kvstore_prev.to('cpu', non_blocking=True)
                            kv_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'KV'
                            kvtmp_event.wait(kvstore_stream)
                            info['feature'][kv_feature_name] = kv_tmp.to('cpu', non_blocking=True)
            
            with torch.cuda.stream(maskgen_stream):
                if (not info['inverse']) and (info['id'] == 0) and (not info['second_order']):
                    # editing_gensematicmask_t1 = time.perf_counter()
                    crossattn_event.wait(maskgen_stream)
                    attn_map = attn_map.mean(dim=1)

                    attn_map_tmp = 0
                    attn_map_tmp = attn_map[:, :, txt_index].sum(dim=-1)
                    attn_map = attn_map_tmp

                    attn_min = attn_map.min()
                    attn_max = attn_map.max()
                    attn_map = (attn_map - attn_min) / (attn_max - attn_min)
                    if info['t'] >= 0.87:
                        attn_bias = 0.05
                    elif info['t'] >= 0.66:
                        attn_bias = 0.15
                    else:
                        attn_bias = 0.25
                    attn_mask = (attn_map >= attn_bias).reshape(h_tmp, w_tmp)
                    mask_final = (input_mask | (~attn_mask)).reshape(-1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, x_dimension) 
                if not info['inverse']:
                    if (info['id'] == 0) and (not info['second_order']):
                        pe_q_tmp = torch.cat((pe[:, :, :512, :, :, :], pe[:, :, 512:, :, :, :][~(mask_final[:, :, :(pe.shape[-3]*pe.shape[-2]*pe.shape[-1])].unsqueeze(0).reshape(1, *mask_final.shape[:-1], -1, 2, 2))].reshape(pe.shape[0], pe.shape[1], -1, pe.shape[3], pe.shape[4], pe.shape[5])), dim=2)
                        pe_q_out = pe_q_tmp
                    else:
                        pe_q_out = pe_q_tmp1
                else:
                    pe_q_out = None
            
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
            
            res = x_tmp + mod.gate * output
            
            res_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'res'
            if (info['id'] == 37):
                if info['inverse']:
                    res_event.record(compute_stream)
                else:
                    rescache_event.wait(compute_stream)
                    res_cache.masked_scatter_(~(mask[:, :, 0].unsqueeze(-1).repeat(1, 1, res.shape[-1])), res[:, 512:, :])
                    
                    res = torch.cat((res[:, :512, :], res_cache), dim=1)
                    restmp_event.record(compute_stream)

            with torch.cuda.stream(memory_stream):
                res_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'res'
                if (info['id'] == 37):
                    if info['inverse']:
                        res_event.wait(memory_stream)
                        if info['t'] < 0.97:
                            info['feature'][res_feature_name] = (res[:, 512:, :]).to('cpu', non_blocking=True)
                        else:
                            info['feature'][res_feature_name] = (res[:, 512:, :]).cuda(non_blocking=True)
                    else:
                        # with torch.cuda.stream(memory_stream):
                        restmp_event.wait(memory_stream)
                        if info['t'] < 0.97:
                            info['feature'][res_feature_name] = (res[:, 512:, :]).to('cpu', non_blocking=True)
                        else:
                            info['feature'][res_feature_name] = (res[:, 512:, :]).cuda(non_blocking=True)

        torch.cuda.synchronize()

        mask = None
        if (not info['inverse']) and (info['id'] == 0) and (not info['second_order']):
            mask = mask_final
        else:
            mask = mask_tmp

        return res, info, mask, pe_q_out, kv_load_next_list, kv_tmp


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
