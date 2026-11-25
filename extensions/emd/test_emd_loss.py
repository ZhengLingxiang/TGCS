import torch

from emd import earth_mover_distance


def test_emd_loss():
    source = torch.tensor([[[0., 0., 0.]]], device='cuda', requires_grad=True)
    target = torch.tensor([[[1., 0., 0.]]], device='cuda', requires_grad=True)
    loss = earth_mover_distance()(source, target)
    assert torch.allclose(loss, loss.new_tensor(1.))
    loss.backward()
    assert torch.allclose(source.grad, -target.grad)


if __name__ == '__main__':
    test_emd_loss()
