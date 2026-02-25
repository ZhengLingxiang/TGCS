import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import os
# import ipdb
import numpy as np
from datasets import data_transforms
import cv2
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from tqdm import tqdm
import torch.profiler as profiler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from utils.misc import summary_parameters

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
        # data_transforms.PointcloudScaleAndTranslate(scale_low=0.9, scale_high=1.1, translate_range=0),
        # data_transforms.PointcloudRotate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    summary_parameters(base_model, logger=logger)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    metrics = validate(base_model, test_dataloader, 0, val_writer, args, config, logger=logger)
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            # import pdb; pdb.set_trace()
            points = train_transforms(points)

            ret = base_model(points)

            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader, smoothing=0.9)):
            points = data[0].cuda()
            label = data[1].cuda()

            points, idx = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)



import os, re
import torch


def _sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s.strip("_")[:80]

@torch.no_grad()
def save_attention_position_with_points(
    points,              # [B, N, 3]
    attn_list,            # list[L], each [B, H, 129, 129]
    center,               # [B, 128, 3]
    neighbor,             # [B, 128, 32, 3]
    labels,               # [B]
    preds,
    class_names=None,
    save_dir="./vis/attn_pos/",
    layer="last",         # "last" / "avg" / int
    head_reduce="mean",   # "mean" / "max"
    norm="minmax",        # "minmax" / "sum"
    base_color="lightgray",
    base_alpha=0.25,
    overlay_point_size=2.0,
    base_point_size=1.0,
    elev=15,
    azim=90,
    dpi=800,
):
    os.makedirs(save_dir, exist_ok=True)

    # -------- to torch --------
    if not torch.is_tensor(points): points = torch.from_numpy(points)
    if not torch.is_tensor(center): center = torch.from_numpy(center)
    if not torch.is_tensor(neighbor): neighbor = torch.from_numpy(neighbor)
    if not torch.is_tensor(labels): labels = torch.from_numpy(np.asarray(labels))
    if not torch.is_tensor(preds): preds = torch.from_numpy(np.asarray(preds))

    points = points.detach().cpu()      # [B,N,3]
    center = center.detach().cpu()
    neighbor = neighbor.detach().cpu()
    labels = labels.detach().cpu().long()
    preds = preds.detach().cpu().long()

    B = points.shape[0]

    # -------- choose layer --------
    if layer == "last":
        attn = attn_list[-1]
    elif layer == "avg":
        attn = torch.stack(attn_list, dim=0).mean(dim=0)
    elif isinstance(layer, int):
        attn = attn_list[layer]
    else:
        raise ValueError("layer must be 'last'/'avg'/int")

    attn = attn.detach().cpu()  # [B,H,129,129]

    # CLS->patch attention
    patch_attn = attn[:, :, 0, 1:]  # [B,H,128]

    if head_reduce == "mean":
        patch_score = patch_attn.mean(dim=1)  # [B,128]
    elif head_reduce == "max":
        patch_score = patch_attn.max(dim=1).values
    else:
        raise ValueError("head_reduce must be mean/max")

    # normalize per instance
    if norm == "minmax":
        mn = patch_score.min(dim=1, keepdim=True).values
        mx = patch_score.max(dim=1, keepdim=True).values
        patch_score = (patch_score - mn) / (mx - mn + 1e-8)
    elif norm == "sum":
        patch_score = patch_score / (patch_score.sum(dim=1, keepdim=True) + 1e-8)
    else:
        raise ValueError("norm must be minmax/sum")

    overlay_pts = neighbor + center.unsqueeze(2)         # [B,128,32,3]
    overlay_pts = overlay_pts.reshape(B, -1, 3)          # [B,4096,3]
    overlay_s  = patch_score.unsqueeze(-1).expand(B,128,32).reshape(B, -1)  # [B,4096]

    for b in range(B):

        if preds[b] != labels[b]: #if preds[b] != labels[b]:
            continue

        xyz_base = points[b].numpy()
        xyz_ov   = overlay_pts[b].numpy()
        s        = overlay_s[b].numpy()

        lab = int(labels[b].item())
        if class_names is not None and 0 <= lab < len(class_names):
            cname = _sanitize_filename(class_names[lab])
        else:
            cname = f"class{lab}"

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection="3d")

        # 1) base object silhouette
        ax.scatter(
            xyz_base[:,0], xyz_base[:,1], xyz_base[:,2],
            s=base_point_size, c=base_color, alpha=base_alpha, linewidths=0
        )

        # 2) overlay attention-colored points
        ax.scatter(
            xyz_ov[:,0], xyz_ov[:,1], xyz_ov[:,2],
            s=overlay_point_size, c=s, cmap="jet", alpha=1.0, linewidths=0
        )

        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        # equal-ish scale using base points (更稳定)
        max_range = (xyz_base.max(axis=0) - xyz_base.min(axis=0)).max()
        mid = (xyz_base.max(axis=0) + xyz_base.min(axis=0)) / 2.0
        ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
        ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
        ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)

        out_path = os.path.join(save_dir, f"sample_{b:03d}_{cname}.png")
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"[OK] Saved {B} images to: {save_dir}")


import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def save_pointcloud_heatmaps_only_correct(
    heat_list,
    points,          # [B, N, 3]
    center,          # [B, 128, 3] (patch centers)
    pred,            # [B] or [B, ...] 预测类别
    label,           # [B] or [B, ...] GT类别
    class_names=None,          # list[str], len = num_classes
    out_dir="./vis/attn_pos/",
    prefix="attnpos",
    agg_layers="mean",         # "mean" / "sum" / int / list[int]
    layer_idx=None,            # 若指定 int，则只画某一层
    heat_mode="cls_to_patch",  # 兼容 [B,H,T,T] 时用；你当前 [64,129] 用不到
    normalize="per_instance",  # "per_instance" 或 "global"
    view_elev=0,               # 正视图：可根据你的习惯调整
    view_azim=90,              # 正视图：可根据你的习惯调整
    point_size=3,
    dpi=800,
    max_points=None,           # None 不采样；比如 2048 可更清晰
    use_cpu_cdist=True,        # True：在CPU算 cdist（更稳）；False：在GPU算（更快）
):

    os.makedirs(out_dir, exist_ok=True)

    # --------- 1) 统一 pred/label 形状 ---------
    if torch.is_tensor(pred):
        pred_t = pred.detach()
    else:
        pred_t = torch.tensor(pred)

    if torch.is_tensor(label):
        label_t = label.detach()
    else:
        label_t = torch.tensor(label)

    pred_t = pred_t.view(-1).long()
    label_t = label_t.view(-1).long()

    # --------- 2) points / center ---------
    assert torch.is_tensor(points) and torch.is_tensor(center), "points/center 必须是 torch.Tensor"
    assert points.dim() == 3 and points.size(-1) == 3, "points 需要 [B,N,3]"
    assert center.dim() == 3 and center.size(-1) == 3, "center 需要 [B,128,3]"

    B, N, _ = points.shape
    _, T, _ = center.shape
    assert T == 128, f"center 的 token 数应为 128，但你传入的是 {T}"

    # 可选点采样（更清楚/更快）
    if max_points is not None and N > max_points:
        idx = torch.randperm(N, device=points.device)[:max_points]
        points = points[:, idx, :]
        N = max_points

    # --------- 3) 从 heat_list 提取 token_heat: [B,128] ---------
    # 先把每一层都转成 [B,128]
    per_layer = []
    for h in heat_list:
        if not torch.is_tensor(h):
            h = torch.tensor(h)

        # [T] -> [1,T]
        if h.dim() == 1:
            h = h.unsqueeze(0)

        if h.dim() == 2:
            # 可能是 [B,T] 或 [H,T] (你现在是 [64,129])
            if h.shape[0] == B:
                # [B,T]
                token_heat = h
            else:
                # [H,T] -> mean heads -> [T] -> [1,T] -> expand B
                token_heat = h.mean(dim=0, keepdim=True).expand(B, -1)

            # 去掉 CLS -> [B,128]
            if token_heat.shape[-1] == 129:
                token_heat = token_heat[:, 1:]
            per_layer.append(token_heat)
            continue

        if h.dim() == 3:
            # [B,H,T]
            if h.shape[0] != B:
                raise ValueError(f"heat dim=3 但 batch 不匹配：heat[0]={h.shape[0]} vs B={B}")
            token_heat = h.mean(dim=1)  # mean heads => [B,T]
            if token_heat.shape[-1] == 129:
                token_heat = token_heat[:, 1:]
            per_layer.append(token_heat)
            continue

        if h.dim() == 4:
            # [B,H,T,T]
            if h.shape[0] != B:
                raise ValueError(f"heat dim=4 但 batch 不匹配：heat[0]={h.shape[0]} vs B={B}")

            if heat_mode == "cls_to_patch":
                # CLS query 到各 token key 的注意力
                token_heat = h[:, :, 0, :].mean(dim=1)  # [B,T]
            else:
                # 一个更“泛化”的强度：mean over heads+queries
                token_heat = h.mean(dim=1).mean(dim=1)  # [B,T]

            if token_heat.shape[-1] == 129:
                token_heat = token_heat[:, 1:]
            per_layer.append(token_heat)
            continue

        raise ValueError(f"Unsupported heat tensor dim: {h.dim()}")

    # 选层 or 聚合层
    if layer_idx is not None:
        token_heat = per_layer[int(layer_idx)]
    else:
        if isinstance(agg_layers, int):
            token_heat = per_layer[int(agg_layers)]
        elif isinstance(agg_layers, (list, tuple)):
            stack = torch.stack([per_layer[i] for i in agg_layers], dim=0)
            token_heat = stack.mean(dim=0)
        elif agg_layers == "mean":
            token_heat = torch.stack(per_layer, dim=0).mean(dim=0)
        elif agg_layers == "sum":
            token_heat = torch.stack(per_layer, dim=0).sum(dim=0)
        else:
            raise ValueError("agg_layers must be 'mean', 'sum', int, or list[int].")

    # token_heat: [B,128]
    token_heat = token_heat.float()

    # --------- 4) token_heat 映射到 points：每个点找最近 center -> 取对应 token_heat ---------
    # 为了稳定：默认用 CPU cdist（尤其是你 GPU 上 topk 报错那种情况）
    if use_cpu_cdist:
        pts = points.detach().cpu()
        ctr = center.detach().cpu()
        th = token_heat.detach().cpu()
    else:
        pts = points.detach()
        ctr = center.detach()
        th = token_heat.detach()

        # 计算最近中心索引： [B,N]
        # dists: [B,N,128]
    dists = torch.cdist(pts, ctr)  # 欧氏距离
    nn_idx = torch.argmin(dists, dim=-1)  # [B,N]

    # point_heat: [B,N]
    point_heat = th.gather(1, nn_idx)

    # --------- 5) 归一化（可选） ---------
    if normalize == "per_instance":
        # 每个样本单独归一化到 [0,1]
        ph = point_heat.numpy()
        for b in range(B):
            vmin, vmax = ph[b].min(), ph[b].max()
            if vmax > vmin:
                ph[b] = (ph[b] - vmin) / (vmax - vmin)
            else:
                ph[b] = 0.0
        point_heat_np = ph
    elif normalize == "global":
        ph = point_heat.numpy()
        vmin, vmax = ph.min(), ph.max()
        if vmax > vmin:
            point_heat_np = (ph - vmin) / (vmax - vmin)
        else:
            point_heat_np = np.zeros_like(ph)
    else:
        point_heat_np = point_heat.numpy()

    pts_np = pts.numpy()

    # --------- 6) 保存图：只保存 pred==label ---------
    saved = []
    for b in range(B):
        p = int(pred_t[b].item())
        y = int(label_t[b].item())
        if p != y: #if p != y:
            continue  # 预测错的不画

        gt_name = str(y)
        pd_name = str(p)
        if class_names is not None and 0 <= y < len(class_names):
            gt_name = class_names[y]
        if class_names is not None and 0 <= p < len(class_names):
            pd_name = class_names[p]

        # 文件名：含GT和Pred，方便你区分/统一对比
        # 例：attnpos_b0003_GTchair_Predchair_layermean.png
        layer_tag = f"layer{layer_idx}" if layer_idx is not None else str(agg_layers)
        fname = f"{prefix}_b{b:04d}_GT{gt_name}_Pred{pd_name}_{layer_tag}.png"
        fpath = os.path.join(out_dir, fname)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            pts_np[b, :, 0], pts_np[b, :, 1], pts_np[b, :, 2],
            c=point_heat_np[b],
            s=point_size
        )

        # 正视图（你说只要正视）
        ax.view_init(elev=view_elev, azim=view_azim)

        # 去掉坐标轴
        ax.set_axis_off()

        # 尽量等比例显示
        xyz = pts_np[b]
        x_min, y_min, z_min = xyz.min(axis=0)
        x_max, y_max, z_max = xyz.max(axis=0)
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        mid_z = (z_max + z_min) * 0.5
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # 可选：加一个 colorbar（如果你不想要可以注释）
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)

        plt.tight_layout()
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        saved.append(fpath)

    return saved



def test_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer

    if args.use_gpu:    
        base_model.to(args.local_rank)

    test(base_model, test_dataloader, args, config, logger=logger)
    
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    show_attn = config.show_attn

    with torch.no_grad():
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            if show_attn:
                logits, attn_list, heat_list, center, neighborhood = base_model(points)

                pred_show = logits.argmax(-1)
                save_attention_position_with_points(
                    points=points,  # 注意这里要是 [B,N,3]，别 points=points[0]
                    attn_list=attn_list,
                    center=center,
                    neighbor=neighborhood,
                    labels=label.view(-1),  # [B]
                    preds=pred_show.view(-1),
                    class_names=None,
                    save_dir="./vis_hardest/fully_attn_pos[pred==label]/batch_idx"+str(batch_idx),
                    elev=15, azim=90,
                )


                saved_paths = save_pointcloud_heatmaps_only_correct(
                    heat_list=heat_list,
                    points=points,
                    center=center,
                    pred=pred_show,
                    label=label,
                    class_names=None,  # 你如果有 ScanObjectNN 15 类名字就传
                    out_dir="./vis_hardest/fully_heat_token[pred==label]/batch_idx"+str(batch_idx),
                    prefix="heat",
                    agg_layers="mean",  # 或 layer_idx=11 画最后一层
                    view_elev=0,
                    view_azim=90
                )
                print(saved_paths)



            else:
                logits = base_model(points)


            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)
            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        if args.vote:
            if args.distributed:
                torch.cuda.synchronize()

            print_log(f"[TEST_VOTE]", logger = logger)
            acc = 0.
            for time in range(1, 300):
                this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
                if acc < this_acc:
                    acc = this_acc
                print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
            print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

    return Acc_Metric(acc)



        # print_log(f"[TEST_VOTE]", logger = logger)
        # acc = 0.
        # for time in range(1, 300):
        #     this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
        #     if acc < this_acc:
        #         acc = this_acc
        #     print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        #print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2048
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)

                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc



def plot_embedding(data, label, title, category_nums):
    TSNE_PATH = "./vis/tsne/"
    colors = []
    if category_nums > 27:
        base = [0,0.3,0.6,0.9]
    else:
        base = [0,0.5,0.9]
    for i in range(len(base)):
        for j in range(len(base)):
            for k in range(len(base)):
                colors.append([base[i],base[j],base[k],1])

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(8, 8))
    for i in range(data.shape[0]):
        print(colors[int(label[i])])
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=colors[int(label[i])],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    if not os.path.isdir(TSNE_PATH):
        os.makedirs(TSNE_PATH)
    plt.savefig(TSNE_PATH+"tsne.png")
    return fig


import os, time, json
import numpy as np

def save_tsne_artifact(
    out_dir,
    coords2d,     # (N,2)
    labels,       # (N,)
    meta,
    name_prefix):

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    npz_path = os.path.join(out_dir, f"{name_prefix}_{ts}.npz")
    np.savez_compressed(
        npz_path,
        coords=coords2d.astype(np.float32),
        labels=labels.astype(np.int64),
        meta=json.dumps(meta or {}, ensure_ascii=False)
    )

    csv_path = os.path.join(out_dir, f"{name_prefix}_{ts}.csv")

    header = "x,y,label\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        for (x, y), lab in zip(coords2d, labels):
            f.write(f"{x},{y},{int(lab)}\n")

    meta_path = os.path.join(out_dir, f"{name_prefix}_{ts}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, ensure_ascii=False, indent=2)

    print(f"[t-SNE saved]\n- {npz_path}\n- {csv_path}\n- {meta_path}")
    return npz_path, csv_path, meta_path


def test_only_tsne(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    test_feature = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # get_local.clear()
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            _,concat_f = base_model(points)

            target = label.view(-1)

            test_label.append(target.detach())
            test_feature.append(concat_f.detach())

        test_label = torch.cat(test_label, dim=0)

        category_nums = config.model.cls_dim

        index = test_label < category_nums
        label_all = test_label[index]
        test_feature = torch.cat(test_feature, dim=0)
        test_feature = test_feature[index]

        # tsne
        test_feature = test_feature.cpu().numpy()
        label = label_all.cpu().numpy()

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(test_feature.squeeze())

        out_dir = "./vis/tsne_artifacts/"
        meta = {
            "title": "TGCS",
            "category_nums": int(category_nums),
            "ckpt": getattr(args, "ckpts", ""),
            "dataset": getattr(config.dataset, "name", ""),
            "npoints": int(npoints),
            "tsne": {
                "n_components": 2,
                "init": "pca",
                "random_state": 0
            }
        }
        save_tsne_artifact(
            out_dir=out_dir,
            coords2d=result,
            labels=label,
            meta=meta,
            name_prefix="TGCS"  # <- 改成 TGCS / baseline / PointGST 等
        )


        fig = plot_embedding(result, label, '', category_nums)



def test_tsne(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test_only_tsne(base_model, test_dataloader, args, config, logger=logger)