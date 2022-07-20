from typing import List

import torch
from torch import Tensor

__all__ = ["cross_fuse"]


def cross_fuse(tensor_list: List[Tensor]):
    batch, channel, h, w = tensor_list[0].shape
    for i, _ in enumerate(tensor_list):
        tensor_list[i] = tensor_list[i].unsqueeze(dim=-3)

    fused = torch.cat(tensor_list, dim=-3).reshape(batch, -1, h, w).contiguous()
    return fused
