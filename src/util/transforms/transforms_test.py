import torch

from util.transforms.fuse import cross_fuse
from util.transforms.patch import reshape_patch, reshape_patch_back


def test_patch_transformers():
    t = torch.ones(8, 10, 1, 128, 128)
    print(f"输入 shape：\t\t\t{t.shape}")
    p = reshape_patch(t, patch_size=4)
    print(f"patch 后 shape：\t{p.shape}")
    t = reshape_patch_back(p, patch_size=4)
    print(f"还原后 shape：\t\t{t.shape}")


def test_cross_fuse():
    a = torch.arange(0, 9).reshape(1, 1, 3, 3).repeat(7, 9, 1, 1)
    b = torch.arange(9, 18).reshape(1, 1, 3, 3).repeat(7, 9, 1, 1)
    c = torch.arange(18, 27).reshape(1, 1, 3, 3).repeat(7, 9, 1, 1)
    r = cross_fuse([a, b, c])
    # r.sum().backward()
    print(r[1, 0])
    print(r[1, 1])
    print(r[1, 2])
    print(r[1, 3])
    print(r[1, 4])
    print(r[1, 5])
