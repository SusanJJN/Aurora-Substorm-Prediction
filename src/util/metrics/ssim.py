import torch
import sys
sys.path.append('/home/jjn/susan/DeepLab/src')
from util.metrics._ssim import ssim

__all__ = ["ssim_per_frame"]
result_path = '/home/jjn/susan/DeepLab/src/model/MotionRNN/results/60w-60h-1chn_5f_9697_teston98v2_fold5/'


def ssim_per_frame(pred, gt):
    assert pred.shape == gt.shape
    _, sequence, _, _, _ = pred.shape
    SSIM = []
    for s in range(sequence):
        SSIM.append(ssim(pred[:, s], gt[:, s]).item())

    return SSIM

def ssim_copy_frame(gt, copy):

    _, sequence, _, _, _ = gt.shape
    SSIM = []
    for s in range(sequence):
        SSIM.append(ssim(gt[:, s], copy).item())

    return SSIM


if __name__ == '__main__':
#     outputs = torch.load(r"/home/timwell/ice/DeepLab/src/model/ConvLSTM/results/prediction.pth")
#     target = torch.load(r"/home/timwell/ice/DeepLab/src/model/ConvLSTM/results/ground_truth.pth")
    print(result_path)
    outputs = torch.load(result_path+'prediction.pth')
    target = torch.load(result_path+'ground_truth.pth')
    last_frames = torch.load(result_path+'last_frames.pth')
    
#     print(outputs.shape)

    pred_ssim = ssim_per_frame(outputs, target)
    copy_ssim = ssim_copy_frame(target, last_frames)
    print("prediction ssim:\n", pred_ssim)
    print("copy last frame:\n", copy_ssim)
