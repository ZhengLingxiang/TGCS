import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def knn_index(pts: torch.Tensor, k: int):
    with torch.no_grad():
        dist = torch.cdist(pts, pts)            # [B,N,N]
        idx = torch.topk(dist, k=k+1, dim=-1, largest=False).indices[..., 1:]  # 去掉自身
    return idx  # [B,N,k]

def gather(x: torch.Tensor, idx: torch.Tensor):
    B, N, D = x.shape
    B2, N2, K = idx.shape
    assert B==B2 and N==N2
    b = torch.arange(B, device=x.device)[:, None, None]
    return x[b, idx, :]  # [B,N,K,D]


@torch.no_grad()
def cov_eigs(rel: torch.Tensor) -> torch.Tensor:
    rel32 = torch.nan_to_num(rel.detach(), nan=0.0, posinf=0.0, neginf=0.0).to('cpu', dtype=torch.float32)
    k = rel32.size(2)
    denom = float(max(k - 1, 1))
    C = torch.einsum('bnki,bnkj->bnij', rel32, rel32) / denom

    eigvals = torch.linalg.eigvalsh(C)
    l1, l2, l3 = eigvals[..., 2], eigvals[..., 1], eigvals[..., 0]
    eps = 1e-12
    l1s = (l1 + eps)

    linearity  = (l1 - l2) / l1s
    planarity  = (l2 - l3) / l1s
    sphericity =  l3 / l1s
    anisotropy = (l1 - l3) / l1s
    omniv      = (l1 * l2 * l3).clamp_min(eps).pow(1/3)

    psum = (l1 + l2 + l3).clamp_min(eps)
    p1, p2, p3 = l1/psum, l2/psum, l3/psum
    eigenentropy = -(p1.clamp_min(eps).log()*p1 + p2.clamp_min(eps).log()*p2 + p3.clamp_min(eps).log()*p3)

    feats = torch.stack([linearity, planarity, sphericity, anisotropy, omniv, eigenentropy], dim=-1)
    return feats.to(device=rel.device, non_blocking=True)


class TopologyAwareXCrossScaleAdapter(nn.Module):
    def __init__(self, dim, k_list=(2, 4, 8, 12, 16, 24, 32, 48),
                 topo_use=('dens', 'var', 'pca6'), weighted=False, add_normal_cons=False,
                 hidden=16, mode='density_curv'):
        super().__init__()
        self.k_list = k_list
        self.topo_use = set(topo_use)
        self.weighted = weighted
        self.add_normal_cons = add_normal_cons
        self.mode = mode

        if weighted:
            self.log_tau = nn.Parameter(torch.tensor(0.0))
        base_per_scale = (('dens' in self.topo_use) + ('var' in self.topo_use) + (6 if 'pca6' in self.topo_use else 0))
        topo_dim = base_per_scale * len(self.k_list)
        if add_normal_cons:
            topo_dim += 0
        self.topo_enc = nn.Sequential(
            nn.LayerNorm(dim*2 + topo_dim),
            nn.Linear(dim*2 + topo_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim)
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _stats_one_scale(self, pos3d, idx):
        nb = gather(pos3d, idx)
        center = pos3d.unsqueeze(2)
        rel = nb - center
        dist = torch.norm(rel, dim=-1) + 1e-6
        outs = []
        if self.weighted:
            tau = torch.exp(self.log_tau)
            w = torch.exp(-(dist ** 2) / (2 * tau ** 2))
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
            inv_mean_dist = 1.0 / ((dist * w).sum(dim=-1) + 1e-6)
            mu = (rel * w.unsqueeze(-1)).sum(dim=2, keepdim=False)
            var = ((rel - mu.unsqueeze(2)) ** 2 * w.unsqueeze(-1)).sum(dim=2).mean(dim=-1)
        else:
            inv_mean_dist = 1.0 / dist.mean(dim=-1)
            var = rel.var(dim=2).mean(dim=-1)
        if 'dens' in self.topo_use: outs.append(inv_mean_dist.unsqueeze(-1))
        if 'var' in self.topo_use: outs.append(var.unsqueeze(-1))
        if 'pca6' in self.topo_use: outs.append(cov_eigs(rel))  # [B,N,6]
        return torch.cat(outs, dim=-1) if len(outs) > 0 else None

    def forward(self, x, pos3d, idx_list=None, normal=None, cached_feat=None):
        B, N, D = x.shape
        stats = []
        for s, k in enumerate(self.k_list):
            idx = idx_list[s] if idx_list is not None else knn_index(pos3d, k)
            stats.append(self._stats_one_scale(pos3d, idx))
        topo = torch.cat(stats, dim=-1)  # [B,N,T]
        if self.add_normal_cons and (normal is not None):
            rel = gather(pos3d, idx_list[0]) - pos3d.unsqueeze(2)
            cos_sim = torch.einsum('bni,bnki->bnk', normal, rel / (torch.norm(rel, dim=-1, keepdim=True) + 1e-6))
            norm_cons = cos_sim.abs().mean(dim=-1, keepdim=True)
            norm_var = cos_sim.var(dim=-1, keepdim=True)
            topo = torch.cat([topo, norm_cons, norm_var], dim=-1)

        if cached_feat is None:
            return x
        else:
            feat = torch.cat([x, cached_feat, topo], dim=-1)  # [B,N,D+2]
            res = self.topo_enc(feat)  # [B,N,Dim+TopoDim]
        out = x + self.alpha * res

        return out * 1.0

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        topoMixScale_adapter_cfg = getattr(config, 'TopoMixXScale_adapter', {}) if hasattr(config, 'TopoMixXScale_adapter') else {}
        self.use_topoMixscale_adapter = bool(topoMixScale_adapter_cfg.get('enabled', False))
        self.topoMixscale_hidden = int(topoMixScale_adapter_cfg.get('hidden'))
        self.topo_klist = topoMixScale_adapter_cfg.k_list
        if self.use_topoMixscale_adapter:
            self.topoMixscale_adapter = TopologyAwareXCrossScaleAdapter(dim=embed_dim, k_list=self.topo_klist, hidden=self.topoMixscale_hidden)

    def forward(self, x, attn_mask, xScale_pos_in=None, topoMixscale_cached_feat=None):

        x = self.ln_1(x)
        # a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a

        if self.use_topoMixscale_adapter and xScale_pos_in is not None and topoMixscale_cached_feat is not None:
            x = x.transpose(0, 1)
            topoMixscale_cached_feat = topoMixscale_cached_feat.transpose(0, 1)
            x = self.topoMixscale_adapter(x, xScale_pos_in, cached_feat=topoMixscale_cached_feat)
            x = x.transpose(0, 1)

        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT_extractor(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_classes, trans_dim, group_size, num_group, config, pretrained=False):
        super(GPT_extractor, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.num_group = num_group

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads, config))

        self.ln_f = nn.LayerNorm(embed_dim)
        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*(self.group_size), 1)
        )

        if pretrained == False:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

            self.cls_norm = nn.LayerNorm(self.trans_dim)

    def forward(self, h, pos, attn_mask, classify=False, xScale_pos3d=None):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=h.device) * self.sos
        if not classify:
            h = torch.cat([sos, h[:-1, :, :]], axis=0)
        else:
            h = torch.cat([sos, h], axis=0)

        xScale_pos_in = None
        if xScale_pos3d is not None:
            N, B, D = h.shape
            G = self.num_group
            P = N - 1 - G
            # center: [B,G,3] 你的组中心三维坐标
            cls_xyz = torch.zeros(B, 1, 3, device=h.device, dtype=h.dtype)
            prompt_xyz = torch.zeros(B, P, 3, device=h.device, dtype=h.dtype)
            xScale_pos_in = torch.cat([cls_xyz, xScale_pos3d, prompt_xyz], dim=1)  # [B, N, 3]
            assert xScale_pos_in.size(1) == N, f"pos3d N={xScale_pos_in.size(1)} vs x N={N}"

        cache_prev1 = None
        cache_prev2 = None

        # transformer
        for layer in self.layers:
            h = layer(h + pos, attn_mask, xScale_pos_in=xScale_pos_in, topoMixscale_cached_feat=cache_prev2)

            if xScale_pos3d is not None:
                cache_token = h.detach()
                cache_prev2, cache_prev1 = cache_prev1, cache_token

        h = self.ln_f(h)

        encoded_points = h.transpose(0, 1)
        if not classify:
            return encoded_points

        h = h.transpose(0, 1)
        h = self.cls_norm(h)
        concat_f = torch.cat([h[:, 1], h[:, 2:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret, concat_f


class GPT_generator(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, trans_dim, group_size
    ):
        super(GPT_generator, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*(self.group_size), 1)
        )

    def forward(self, h, pos, attn_mask):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        # transformer
        for layer in self.layers:
            h = layer(h + pos, attn_mask)

        h = self.ln_f(h)

        rebuild_points = self.increase_dim(h.transpose(1, 2)).transpose(
            1, 2).transpose(0, 1).reshape(batch * length, -1, 3)

        return rebuild_points
