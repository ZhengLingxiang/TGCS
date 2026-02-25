import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils import misc
from knn_cuda import KNN

class SaliencyGate(nn.Module):

    def __init__(self, dim, hidden=128, temperature=1.0): # 128 64 32
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
        self.temperature = temperature

    def forward(self, tokens):  # [B, 1+G, D], cls at index 0
        score = self.net(tokens).squeeze(-1)  # [B,1+G]
        gate_vals = torch.sigmoid(score / self.temperature).unsqueeze(-1)

        ones = torch.ones_like(gate_vals[:, :1, :])
        gate = torch.cat((ones, gate_vals[:, 1:, :]), dim=1)

        return tokens * gate


class MultiScaleAggregator(nn.Module):

    def __init__(self, dim, hidden, k_small=8, k_large=24):
        super().__init__()
        self.k_small = k_small
        self.k_large = k_large
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden), # 64 # 32 24
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim) # 64 # 32 24
        )

    @staticmethod
    def knn(centers, k):
        # centers: [B,G,3]
        with torch.no_grad():
            dist = torch.cdist(centers, centers, p=2)  # [B,G,G]
            idx = torch.topk(dist, k=k + 1, dim=-1, largest=False).indices[:, :, 1:]
        return idx  # [B,G,k]

    @staticmethod
    def gather_neighbors(feat, idx):
        # feat: [B,G,D], idx: [B,G,k] -> [B,G,k,D]
        B, G, D = feat.shape
        batch = torch.arange(B, device=feat.device)[:, None, None]
        return feat[batch, idx, :]

    def forward(self, tokens_wo_cls, centers):
        # tokens_wo_cls: [B,G,D]; centers: [B,G,3]
        idx_s = self.knn(centers, self.k_small)
        idx_l = self.knn(centers, self.k_large)
        nb_s = self.gather_neighbors(tokens_wo_cls, idx_s).mean(dim=2)  # [B,G,D]
        nb_l = self.gather_neighbors(tokens_wo_cls, idx_l).mean(dim=2)  # [B,G,D]

        agg = self.mlp(torch.cat([nb_s, nb_l], dim=-1))
        return agg

@torch.no_grad()
def knn_index(pts: torch.Tensor, k: int):
    # pts: [B, N, 3]  返回 [B, N, k] 的邻接索引（不含自身）
    with torch.no_grad():
        dist = torch.cdist(pts, pts)            # [B,N,N]
        idx = torch.topk(dist, k=k+1, dim=-1, largest=False).indices[..., 1:]  # 去掉自身
    return idx  # [B,N,k]

def gather(x: torch.Tensor, idx: torch.Tensor):
    # x: [B,N,D], idx: [B,N,k] -> [B,N,k,D]
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
    C = torch.einsum('bnki,bnkj->bnij', rel32, rel32) / denom  # [B,N,3,3]

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

    feats = torch.stack([linearity, planarity, sphericity, anisotropy, omniv, eigenentropy], dim=-1)  # [B,N,6]
    return feats.to(device=rel.device, non_blocking=True)


class TopologyAwareXCrossScaleAdapter(nn.Module):
    def __init__(self, dim, k_list=(2, 4, 8, 12, 16, 24, 32, 48), topo_use=('dens', 'var', 'pca6'), weighted=False, add_normal_cons=False,
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


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 384 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 384
        return feature_global.reshape(bs, g, self.encoder_channel)  # [B, G, 384]


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, require_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if require_attn:
            return x, attn
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            BlockMAE(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class BlockMAE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.return_attn = config.show_attn

        topoMixScale_adapter_cfg = getattr(config, 'TopoMixXScale_adapter', {}) if hasattr(config, 'TopoMixXScale_adapter') else {}
        self.use_topoMixscale_adapter = bool(topoMixScale_adapter_cfg.get('enabled', False))
        self.topoMixscale_hidden = int(topoMixScale_adapter_cfg.get('hidden'))
        self.topo_klist = topoMixScale_adapter_cfg.k_list
        if self.use_topoMixscale_adapter:
            self.topoMixscale_adapter = TopologyAwareXCrossScaleAdapter(dim=dim, k_list=self.topo_klist, hidden=self.topoMixscale_hidden)



    def forward(self, x,  topoMixscale_pos_in, topoMixscale_cached_feat, layer_idx):
        if self.return_attn:
            x_before = x.detach()
            attn_out, attn = self.attn(self.norm1(x), self.return_attn)
            x = x + self.drop_path(attn_out)
        else:
            x_before = x.detach()
            x = x + self.drop_path(self.attn(self.norm1(x), self.return_attn))


        if self.use_topoMixscale_adapter:
            x = self.topoMixscale_adapter(x, topoMixscale_pos_in, cached_feat=topoMixscale_cached_feat)
            if self.return_attn:
                x_after = x.detach()
                x_delta = x_after - x_before
                heat = x_delta.norm(dim=-1)
                n_patch = topoMixscale_pos_in.shape[1]
                if heat.shape[1] == n_patch + 1:
                    heat = heat[:, 1:]
        else:
            if self.return_attn:
                x_after = x.detach()
                x_delta = x_after - x_before
                heat = x_delta.norm(dim=-1)
                n_patch = topoMixscale_pos_in.shape[1]
                if heat.shape[1] == n_patch + 1:
                    heat = heat[:, 1:]

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.return_attn:
            return x, attn, heat

        return x


class TransformerEncoderDeep(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., config=None, num_group=None):
        super().__init__()
        self.num_group = num_group
        self.blocks = nn.ModuleList([
            BlockMAE( # 这里为了验证是不是改了名字会有导入参数的影响，把名字改回去了 BlockMAE
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                config=config)
            for i in range(depth)])

        self.return_attn = config.show_attn

    def forward(self, x, pos, center):

        cache_prev1 = None
        cache_prev2 = None

        attn_list = [] if self.return_attn else None
        heat_list = [] if self.return_attn else None

        for idx, block in enumerate(self.blocks):
            x = x[:, :self.num_group + 1, :]
            pos = pos[:, :self.num_group + 1, :]

            topoMixscale_pos_in = None
            if center is not None:
                B, N, D = x.shape
                G = self.num_group
                P = N - 1 - G
                cls_xyz = torch.zeros(B, 1, 3, device=x.device, dtype=x.dtype)
                topoMixscale_pos_in = torch.cat([cls_xyz, center], dim=1)  # [B, N, 3]

            if self.return_attn:
                x, attn, heat = block(x=x + pos, topoMixscale_pos_in=topoMixscale_pos_in, topoMixscale_cached_feat=cache_prev2, layer_idx=idx)
                attn_list.append(attn)
                heat_list.append(heat)
            else:
                x = block(x=x + pos, topoMixscale_pos_in=topoMixscale_pos_in, topoMixscale_cached_feat=cache_prev2, layer_idx=idx)

            if center is not None:
                cache_token = x.detach()
                cache_prev2, cache_prev1 = cache_prev1, cache_token

        if self.return_attn:
            return x, attn_list, heat_list

        return x


@MODELS.register_module()
class PointMAE_Transformer_TGCS(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        # self.prompt_nums = config.prompt_depth

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.norm = nn.LayerNorm(self.trans_dim)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, self.cls_dim)
        )
        for layer in self.cls_head_finetune:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=5.0 ** 0.5)

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        gate_cfg = getattr(config, 'gate', {}) if hasattr(config, 'gate') else {}
        mixer_cfg = getattr(config, 'mixer', {}) if hasattr(config, 'mixer') else {}
        self.ms_on = bool(mixer_cfg.get('multi_scale_agg', False))
        self.k_small = int(mixer_cfg.get('k_small'))
        self.k_large = int(mixer_cfg.get('k_large'))
        self.mixer_hidden = int(mixer_cfg.get('hidden'))
        self._ms_agg = None
        if self.ms_on:
            self._ms_agg = MultiScaleAggregator(dim=self.trans_dim, hidden=self.mixer_hidden, k_small=self.k_small, k_large=self.k_large)
            self.mixer_alpha = nn.Parameter(torch.tensor(0.0))

        self.gate_on = bool(gate_cfg.get('saliency_gate', False))
        self.gate_hidden = int(gate_cfg.get('hidden'))
        self.gate_temp = int(gate_cfg.get('temperature'))
        self._sal_gate = None
        if self.gate_on:
            self._sal_gate = SaliencyGate(dim=self.trans_dim, hidden=self.gate_hidden, temperature=self.gate_temp)
            self.gate_alpha = nn.Parameter(torch.tensor(0.0))

        self.return_attn = config.show_attn

        self.blocks = TransformerEncoderDeep(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            num_group=self.num_group,
            config=config)

        # self.blocks = TransformerEncoder(
        #     embed_dim=self.trans_dim,
        #     depth=self.depth,
        #     drop_path_rate=dpr,
        #     num_heads=self.num_heads,
        #     num_group=self.num_group,
        #     config=config)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                if k.startswith('ACT_encoder'):
                    base_ckpt[k[len('ACT_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, batch_idx=None):
        # visualize = False
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        if self.ms_on and (self._ms_agg is not None):
            agg = self._ms_agg(group_input_tokens, center)
            x = torch.cat((x[:, :1, :], x[:, 1:, :] + self.mixer_alpha * agg), dim=1)
        if self.gate_on and (self._sal_gate is not None):
            x_gated = self._sal_gate(x)
            x = x * (1.0 - self.gate_alpha) + x_gated * self.gate_alpha

        x_raw = x

        # transformer

        if self.return_attn:
            x, attn_list, heat_list = self.blocks(x, pos, center)
        else:
            x = self.blocks(x, pos, center)

        x = self.norm(x)

        G = self.num_group
        cls_after = x[:, 0, :]
        raw_patches = x_raw[:, 1:1 + G, :]
        raw_mean = raw_patches.mean(dim=1)
        x_max = x[:, 1:].max(1)[0]
        concat_f = torch.cat([cls_after, x_max, raw_mean], dim=-1)
        ret = self.cls_head_finetune(concat_f)

        if self.return_attn:
            return ret, attn_list, heat_list, center, neighborhood

        return ret