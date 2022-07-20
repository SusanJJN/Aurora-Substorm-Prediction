import numpy as np

__all__ = ["batch_psnr"]

import torch


def batch_psnr(gen_frames, gt_frames):
    """
    :param gen_frames:   prediction
    :param gt_frames:    ground truth
            need to transform the pixel value to 0-255
    :return: PSNR
    """
    # [batch, width, height] or [batch, channel, width, height]
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(x[0]))
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)


def psnr_per_frame(gen_frames, gt_frames):
    _, seq, _, _, _ = gen_frames.shape
    PSNR = []
    for s in range(seq):
        pred = gen_frames[:, s] * 70
        target = gt_frames[:, s] * 70
        PSNR.append(batch_psnr(pred, target))

    return PSNR


if __name__ == '__main__':
    outputs = torch.load(r"/home/timwell/ice/DeepLab/src/model/ConvLSTM/results/prediction.pth")
    target = torch.load(r"/home/timwell/ice/DeepLab/src/model/ConvLSTM/results/ground_truth.pth")

    r = psnr_per_frame(outputs, target)
    print(r)
