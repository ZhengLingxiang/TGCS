import math

import torch


@torch.no_grad()
def symmetric_eigenvalues3(matrix):
    a = matrix.to(dtype=torch.float64)
    magnitude = a.abs().max(-1).values.max(-1).values
    scale = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    a = a / scale[..., None, None]
    q = (a[..., 0, 0] + a[..., 1, 1] + a[..., 2, 2]) / 3.
    d0, d1, d2 = a[..., 0, 0] - q, a[..., 1, 1] - q, a[..., 2, 2] - q
    a01, a02, a12 = a[..., 0, 1], a[..., 0, 2], a[..., 1, 2]
    p = ((d0.square() + d1.square() + d2.square() +
          2. * (a01.square() + a02.square() + a12.square())) / 6.).sqrt()
    safe_p = torch.where(p > 0, p, torch.ones_like(p))
    b0, b1, b2 = d0 / safe_p, d1 / safe_p, d2 / safe_p
    b01, b02, b12 = a01 / safe_p, a02 / safe_p, a12 / safe_p
    determinant = b0 * b1 * b2 + 2. * b01 * b02 * b12 - b0 * b12.square() - b1 * b02.square() - b2 * b01.square()
    angle = torch.acos((determinant / 2.).clamp(-1., 1.)) / 3.
    largest = q + 2. * p * torch.cos(angle)
    smallest = q + 2. * p * torch.cos(angle + 2. * math.pi / 3.)
    middle = 3. * q - largest - smallest
    return torch.sort(torch.stack((smallest, middle, largest), -1) * scale[..., None], dim=-1).values


@torch.no_grad()
def analytic_pca_features(rel):
    r = torch.nan_to_num(rel.detach(), nan=0., posinf=0., neginf=0.).to(torch.float64)
    denominator = float(max(r.size(-2) - 1, 1))
    entries = {(i, j): (r[..., i] * r[..., j]).sum(-1) / denominator
               for i in range(3) for j in range(i, 3)}
    covariance = torch.stack([entries[min(i, j), max(i, j)] for i in range(3) for j in range(3)], -1)
    covariance = covariance.reshape(*r.shape[:-2], 3, 3)
    eig = symmetric_eigenvalues3(covariance).float()
    l1, l2, l3 = eig[..., 2], eig[..., 1], eig[..., 0]
    eps = 1e-12
    largest = l1 + eps
    total = (l1 + l2 + l3).clamp_min(eps)
    p1, p2, p3 = l1 / total, l2 / total, l3 / total
    entropy = -(p1.clamp_min(eps).log() * p1 + p2.clamp_min(eps).log() * p2 + p3.clamp_min(eps).log() * p3)
    return torch.stack(((l1 - l2) / largest, (l2 - l3) / largest, l3 / largest,
                        (l1 - l3) / largest, (l1 * l2 * l3).clamp_min(eps).pow(1 / 3), entropy), -1)


class GeometryCache:
    """Reuse anchor neighborhoods and geometric descriptors within one forward pass."""

    def __init__(self, positions, backend):
        self.positions = positions
        self.backend = backend
        self.distances = None
        self.indices = {}
        self.descriptors = {}

    @torch.no_grad()
    def topology(self, adapter):
        key = (tuple(adapter.k_list), frozenset(adapter.topo_use))
        if key not in self.descriptors:
            positions = self.positions
            if self.distances is None:
                self.distances = torch.cdist(positions, positions)
            values = []
            for k in adapter.k_list:
                if k not in self.indices:
                    self.indices[k] = torch.topk(self.distances, k=k + 1,
                                                 dim=-1, largest=False).indices[..., 1:]
                idx = self.indices[k]
                if self.backend == 'cpu_cached':
                    values.append(adapter._stats_one_scale(positions, idx))
                else:
                    batch = torch.arange(len(positions), device=positions.device)[:, None, None]
                    rel = positions[batch, idx] - positions.unsqueeze(2)
                    terms = []
                    if 'dens' in adapter.topo_use:
                        terms.append((1.0 / (rel.norm(dim=-1) + 1e-6).mean(-1)).unsqueeze(-1))
                    if 'var' in adapter.topo_use:
                        terms.append(rel.var(dim=2).mean(-1).unsqueeze(-1))
                    if 'pca6' in adapter.topo_use:
                        terms.append(analytic_pca_features(rel))
                    values.append(torch.cat(terms, dim=-1))
            self.descriptors[key] = torch.cat(values, dim=-1)
        return self.descriptors[key]
