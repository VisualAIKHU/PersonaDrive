from typing import List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from transformers import AutoConfig, AutoModel

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class GridSampleCrossBEVAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=8, config=None):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims,num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)


        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """

        bs, num_queries, num_points, _ = traj_points.shape
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y
        
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # bs, C, num_queries, num_points

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C
        out = self.output_proj(out)

        return self.dropout(out) + queries


class QueryPrototypePool(nn.Module):
    def __init__(self, d_model, M=8, in_dim=None):
        super().__init__()
        self.M = M
        self.in_proj = None
        if in_dim is not None and in_dim != d_model:
            self.in_proj = nn.Linear(in_dim, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.pool_seed = nn.Parameter(torch.randn(M, d_model))
        nn.init.normal_(self.pool_seed, std=0.02)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, traj_feature, time_embed=None, status_encoding=None):
        x = traj_feature
        if self.in_proj is not None:
            x = self.in_proj(x)
        x = self.ln(x)

        B, Q, d = x.shape
        seeds = self.pool_seed.unsqueeze(0).expand(B, -1, -1)
        q = self.q_proj(seeds)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax((q @ k.transpose(1, 2)) / math.sqrt(d), dim=-1)
        P = attn @ v 
        if time_embed is not None: P = P + time_embed.unsqueeze(1)
        if status_encoding is not None: P = P + status_encoding.unsqueeze(1)
        return P


class CondResampler(nn.Module):
    def __init__(self, in_dim, d_model, num_heads, num_slots=16,
                 dropout=0.1, batch_first=True):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.slot_seed = nn.Parameter(torch.randn(num_slots, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=batch_first)
        self.ln = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)
        self.cond_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        nn.init.normal_(self.slot_seed, std=0.02)

    def forward(self, P, x):
        B = x.size(0)
        x = self.proj_in(x)

        reduce_dims = tuple(range(1, P.dim() - 1))
        cond = P.mean(dim=reduce_dims, keepdim=False)

        cond = cond.unsqueeze(1)
        cond = self.cond_proj(cond)

        slots = self.slot_seed.unsqueeze(0).expand(B, -1, -1) + cond

        out, _ = self.attn(slots, x, x)
        return self.ln(slots + self.do(out))


class SourceGate(nn.Module):
    def __init__(self, d_model, src_names):
        super().__init__()
        self.src_names = src_names
        self.fc = nn.Linear(d_model, len(src_names))

    def forward(self, context_vec, tokens_by_src):
        """
        context_vec: [B, d]  (e.g., traj_feature mean or prototype mean)
        tokens_by_src: dict{name: [B,S_i,d]}
        """
        g = torch.softmax(self.fc(context_vec), dim=-1)
        out = []
        for i, name in enumerate(self.src_names):
            if name not in tokens_by_src: 
                continue
            tok = tokens_by_src[name]
            out.append(tok * g[:, i].unsqueeze(-1).unsqueeze(-1))
        return torch.cat(out, dim=1)


class PCMF_Fuser(nn.Module):
    def __init__(self, specs, d_model, num_heads, dropout=0.1):
        """
        specs: {name: (in_dim, num_slots)}
        """
        super().__init__()
        self.src_names = list(specs.keys())
        
        self.resamplers = nn.ModuleDict({
            name: CondResampler(in_dim, d_model, num_heads, num_slots, dropout)
            for name, (in_dim, num_slots) in specs.items()
        })
        self.gate = SourceGate(d_model, self.src_names)

        self.cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_o = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

        self.proto_pool = QueryPrototypePool(d_model, M=8)

    def forward(self, traj_feature, sources, time_embed=None, status_encoding=None):
        """
        traj_feature: [B,Q,d]
        sources: dict{name: tensor[B,L_i,in_dim_i]}
        return: fused_once(traj_feature)  # [B,Q,d]
        """
        P = self.proto_pool(traj_feature, time_embed, status_encoding)

        tokens_by_src = {}
        for name, x in sources.items():
            if x is None: 
                continue
            tokens_by_src[name] = self.resamplers[name](P, x)

        ctx = traj_feature.mean(dim=1)
        fused_memory = self.gate(ctx, tokens_by_src)

        q = self.ln_q(traj_feature)
        out, _ = self.cross(q, fused_memory, fused_memory)
        return self.ln_o(traj_feature + self.do(out))


class BERT(nn.Module):
    def __init__(self, backbone="sentence-transformers/all-MiniLM-L6-v2",
                 proj_dim=256, num_classes=9,
                 layer_strategy="last",          # 'last' | 'last4' | 'indices' | 'scalar'
                 layer_indices=None,             # e.g., roberta/deberta-large -> [12,16,20,24]
                 ):
        super().__init__()
        self.layer_strategy = layer_strategy
        self.layer_indices = layer_indices

        cfg = AutoConfig.from_pretrained(backbone)
        cfg.output_hidden_states = True
        self.enc = AutoModel.from_pretrained(backbone, config=cfg)
        self.enc.requires_grad_(False)

        H = self.enc.config.hidden_size  # 384
        self.proj = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(proj_dim),
        )
        self.clf = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, num_classes),
        )
         
        if self.layer_strategy == "scalar":
            self.num_blocks = self.enc.config.num_hidden_layers
            self.scalar_w = nn.Parameter(torch.zeros(self.num_blocks))  # [N]
            self.scalar_gamma = nn.Parameter(torch.ones(1))

    @staticmethod
    def mean_pool(last, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom
    
    def _mix_layers(self, out):
        # out.hidden_states: tuple, [0]=embedding, [1..N]=blocks
        if self.layer_strategy == "last":
            return out.last_hidden_state                           # [B,T,H]

        blocks = list(out.hidden_states[1:])                       # length N
        if self.layer_strategy == "last4":
            k = min(4, len(blocks))
            return torch.stack(blocks[-k:], dim=0).mean(0)         # [B,T,H]

        if self.layer_strategy == "indices":
            assert self.layer_indices is not None and len(self.layer_indices) > 0
            # layer_indices are 1..N (block numbers)
            picked = [blocks[i-1] for i in self.layer_indices]     # each [B,T,H]
            return torch.stack(picked, dim=0).mean(0)              # [B,T,H]

        if self.layer_strategy == "scalar":
            alpha = torch.softmax(self.scalar_w, dim=0)            # [N]
            mix = sum(a * h for a, h in zip(alpha, blocks))        # [B,T,H]
            return self.scalar_gamma * mix

        # fallback
        return out.last_hidden_state

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        token_feats = self._mix_layers(out)                # [B,T,H] (last or mixed)
        pooled = self.mean_pool(token_feats, attention_mask)  # [B,H]
        emb = self.proj(pooled)
        logits = self.clf(pooled)
        return emb, logits