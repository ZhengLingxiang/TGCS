# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-10 10:38:01
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 14:21:36
# @Email:  cshzxie@gmail.com

import os
import sys
import torch
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.chamfer_dist import ChamferFunction


class ChamferDistanceTestCase(unittest.TestCase):
    def test_chamfer_dist(self):
        x = torch.tensor([[[0., 0., 0.]]], device='cuda', requires_grad=True)
        y = torch.tensor([[[1., 0., 0.]]], device='cuda', requires_grad=True)
        dist_xy, dist_yx = ChamferFunction.apply(x, y)
        self.assertTrue(torch.allclose(dist_xy, torch.ones_like(dist_xy)))
        self.assertTrue(torch.allclose(dist_yx, torch.ones_like(dist_yx)))
        (dist_xy + dist_yx).sum().backward()
        self.assertTrue(torch.allclose(x.grad, x.new_tensor([[[-4., 0., 0.]]])))
        self.assertTrue(torch.allclose(y.grad, -x.grad))


if __name__ == '__main__':
    unittest.main()
