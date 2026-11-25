import torch
import emd_cuda


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


class earth_mover_distance(torch.nn.Module):
    """Mean approximate squared transport cost per source point."""

    def forward(self, xyz1, xyz2, transpose=False):
        """Accept coordinates shaped (B, N, 3), or (B, 3, N) with transpose=True."""
        if transpose:
            xyz1 = xyz1.transpose(1, 2)
            xyz2 = xyz2.transpose(1, 2)
        cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
        return (cost / xyz1.size(1)).mean()
