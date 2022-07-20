import torch
from torch import Tensor

__all__ = [
    "regression_mse",
    "extrapolation_mse"
]


def regression_mse(outputs: Tensor, target: Tensor):
    if torch.cuda.is_available():
        outputs.to("cuda")
        target.to("cuda")

    MSE = (outputs - target) ** 2

    MSE = MSE.mean()

    return MSE


def extrapolation_mse(outputs_: Tensor, target_: Tensor):
    _, sequence, _, _, _ = target_.shape
    if torch.cuda.is_available():
        outputs_.to("cuda")
        target_.to("cuda")

    outputs = outputs_ * 70
    target = target_ * 70

    MSE = (outputs - target) ** 2

    MSE = MSE.mean(dim=0).mean(dim=-1).mean(dim=-1).mean(dim=-1).numpy().tolist()

    return MSE


if __name__ == '__main__':
    outputs = torch.load(r"/home/timwell/ice/DeepLab/src/model/CrevNet/results/prediction.pth")
    target = torch.load(r"/home/timwell/ice/DeepLab/src/model/CrevNet/results/ground_truth.pth")

    mse = extrapolation_mse(outputs, target)
    print(mse)
