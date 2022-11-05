from pathlib import Path
from typing import List
import sys
# you need to change this path to the absolute path of src file
sys.path.append('./src')
from torch import nn, optim
from torch.utils.data import DataLoader
from util.data_parallel import BalancedDataParallel
from util.parallel_v2 import DataParallelModel, DataParallelCriterion

from ConvLSTM import ConvLSTM
from data import AuroraDatasetMix
from util import TrainingTemplate
from util.TestingTemplate import TestingTemplate
from util.transforms.patch import reshape_patch, reshape_patch_back
from data.seq_generator import *
from data.train_corr_scripts import *
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


learning_rate: float = 0.001
batch_size: int = 1
test_bs: int=1
patch: int = 1
device: str = "cuda:0"
test_frequency: int = 1
max_epochs = 100
start_save = 0


result_root = 'results/'

class ConvLSTMTrainer(TrainingTemplate):
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

    def load(self):
        self.model = self.states['modell']
        self.optimizer = self.states['opt']
        self.start_epoch = self.states["epoch"]

    def set_states(self, epoch, loss):
        states = {
            "epoch": epoch + 1,
            "loss": loss,
            "modell": self.model,
            "model": self.model.state_dict(),
            "opt" : self.optimizer,
            "optimizer": self.optimizer.state_dict()
        }
        return states


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



def train(out_len, boot):

    train_set = AuroraDatasetMix.AuroraDataset('train_60_boots/'+str(boot)+'/tra',in_len=5, out_len=out_len)
    test_set = AuroraDatasetMix.AuroraDataset('train_60_boots/'+str(boot)+'/val',in_len=5, out_len=out_len)

    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False,
                              pin_memory=True)
    print('len(train_loader):', len(train_loader))
    
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False,
                             pin_memory=True)
    print('len(val_loader):', len(test_loader))

    model = ConvLSTM(in_channels=in_channels * patch ** 2, hidden_channels_list=hidden_channels_list,
                     kernel_size_list=kernel_size_list, forget_bias=forget_bias).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    to_save = Path(__file__).parent.joinpath(result_root+str(boot))

    if not to_save.exists():
        to_save.mkdir()

    trainer = ConvLSTMTrainer(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion,
                              optimizer=optimizer, lr_scheduler=None, max_epochs=max_epochs, device=device,
                              to_save=to_save, test_frequency=test_frequency, start_save=start_save, visualize=True, vis_name='60-bootstrap-valonoob-v3_'+str(boot))
    trainer.run()



def test_long_seq(out_len, boot):
    
    # here you need to change the npz_root as your path
    npz_root = './test_60'


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
    for boot in range(0,100):
        train(20, boot)
        test_long_seq(20, boot)

#     test_long_seq(20, 0)
        
        
        
