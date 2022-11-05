from pathlib import Path
from typing import List
import sys
# you need to change this path to the absolute path of src file
sys.path.append('./src')
from torch import nn, optim
from torch.utils.data import DataLoader

from ConvLSTM import ConvLSTM
from data import AuroraDatasetMix
from util import TrainingTemplate
from util.TestingTemplate import TestingTemplate
from util.transforms.patch import reshape_patch, reshape_patch_back
import torch
import os
import shutil
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from util.metrics._ssim import ssim

# default
in_channels: int = 1
hidden_channels_list: List[int] = [8, 8]
kernel_size_list: List[int] = [3, 3]
forget_bias: float = 0.01

batch_size: int = 1
test_bs: int=1
patch: int = 1
device: str = "cuda:0"

# This is a default result_root for checkpoint.pth, you can change this result_root
result_root = 'results/'

class ConvLSTMTester(TestingTemplate):
    def check_data(self, data):
        inputs, labels, out_len = data
        inputs = inputs[:,0].permute(0,1,2,4,3)
        labels = labels[:,0].permute(0,1,2,4,3)

        inputs = reshape_patch(inputs, patch_size=patch)
        labels = reshape_patch(labels, patch_size=patch)

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        out_len = out_len.to(self.device)
        return inputs, labels[:,:out_len], out_len

    def check_data_back(self, labels, outputs):
        labels = reshape_patch_back(labels, patch_size=patch)
        outputs = reshape_patch_back(outputs, patch_size=patch)
        return labels, outputs

    def load(self):
        self.model = self.states["modell"]


def test_long_seq(out_len, boot):
    
    # here you need to change the npz_root as your path
    npz_root = './npz/test_60'


    length_list = os.listdir(npz_root)
    length_list.sort()
    for length in length_list[:20]:
        print('length:', length)
        out_length = int(length)-5
        test_set = AuroraDatasetMix.AuroraDataset('test_60/'+length, in_len=5,out_len=out_length, istest=True)

        test_loader = DataLoader(dataset=test_set, batch_size=test_bs, num_workers=4, shuffle=False, drop_last=False,
                                     pin_memory=True)

        model = ConvLSTM(in_channels=in_channels * patch ** 2, hidden_channels_list=hidden_channels_list,
                         kernel_size_list=kernel_size_list, forget_bias=forget_bias).to(device)

        to_save = Path(__file__).parent.joinpath(result_root+str(boot))
        test_save = Path(__file__).parent.joinpath(to_save,'long_test')
        if not test_save.exists():
            test_save.mkdir()
        tester = ConvLSTMTester(model=model, test_loader=test_loader, device=device, to_save=to_save)

        tester.run(out_len=out_length)

if __name__ == '__main__':

     test_long_seq(20, 0)
        
        
        
