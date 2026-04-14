from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flux.modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)
import os


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.mask_tmp = None
        self.pe_q = None
        self.d_mask_tmp = None
        self.d_pe_q = None

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        info = None,
        txt_index = -1,
        mask_tmp = None,
        pe_q = None,
        d_mask_tmp = None,
        d_pe_q = None,
        h_tmp = 48,
        w_tmp = 85,
        memory_stream = None,
        knorm_stream = None,
        vnorm_stream = None,
        kvload_stream = None,
        kvstore_stream = None,
        compute_stream = None,
        maskgen_stream = None,
        crossattn_stream = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        cnt = 0
        info['type'] = 'double'
        self.d_mask_tmp = d_mask_tmp
        # self.mask_sem = mask_sem
        self.d_pe_q = d_pe_q
        d_kv_load_next_list = None
        d_kvstore_prev = None
        for block in self.double_blocks:
            info['id'] = cnt
            img, txt, info, d_mask_tmp, d_pe_q_tmp, d_kv_load_next_list_tmp, d_kvstore_prev_tmp = block(img=img, txt=txt, vec=vec, pe=pe, info=info, txt_index=txt_index, mask_tmp=self.d_mask_tmp, pe_q_tmp1=self.d_pe_q, h_tmp=h_tmp, w_tmp=w_tmp, memory_stream=memory_stream, knorm_stream=knorm_stream, vnorm_stream=vnorm_stream, kvload_stream=kvload_stream, kvstore_stream=kvstore_stream, compute_stream=compute_stream, maskgen_stream=maskgen_stream, crossattn_stream=crossattn_stream, kv_load_list=d_kv_load_next_list, kvstore_prev=d_kvstore_prev)
            if cnt == 0:
                self.d_mask_tmp = d_mask_tmp
                # self.mask_sem = mask_sem
            if cnt == 0:
                self.d_pe_q = d_pe_q_tmp
            d_kv_load_next_list = d_kv_load_next_list_tmp
            d_kvstore_prev = d_kvstore_prev_tmp
            cnt += 1

        cnt = 0
        img = torch.cat((txt, img), 1) 
        info['type'] = 'single'
        self.mask_tmp = mask_tmp
        # self.mask_sem = mask_sem
        self.pe_q = pe_q
        kv_load_next_list = None
        kvstore_prev = None
        # print('double stream block')
        for block in self.single_blocks:
            info['id'] = cnt
            img, info, mask_tmp, pe_q_tmp, kv_load_next_list_tmp, kvstore_prev_tmp = block(img, vec=vec, pe=pe, info=info, txt_index=txt_index, mask_tmp=self.mask_tmp, pe_q_tmp1=self.pe_q, h_tmp=h_tmp, w_tmp=w_tmp, memory_stream=memory_stream, knorm_stream=knorm_stream, vnorm_stream=vnorm_stream, kvload_stream=kvload_stream, kvstore_stream=kvstore_stream, compute_stream=compute_stream, maskgen_stream=maskgen_stream, crossattn_stream=crossattn_stream, kv_load_list=kv_load_next_list, kvstore_prev=kvstore_prev)
            if cnt == 0:
                self.mask_tmp = mask_tmp
            if cnt == 0:
                self.pe_q = pe_q_tmp
            kv_load_next_list = kv_load_next_list_tmp
            kvstore_prev = kvstore_prev_tmp
            cnt += 1

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        return img, info, self.mask_tmp, self.pe_q, self.d_mask_tmp, self.d_pe_q
