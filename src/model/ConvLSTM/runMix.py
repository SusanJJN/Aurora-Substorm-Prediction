from pathlib import Path
from typing import List
import sys
sys.path.append('/home/jjn/susan/DeepLab/src')
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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from util.metrics._ssim import ssim

# 定义模型参数
in_channels: int = 1
# 默认
hidden_channels_list: List[int] = [8, 8]
kernel_size_list: List[int] = [3, 3]
# 第一次调参
# hidden_channels_list: List[int] = [8, 8, 8]
# kernel_size_list: List[int] = [3, 3, 3]
# 第二次调参:16*2hid
# hidden_channels_list: List[int] = [16, 16]
# kernel_size_list: List[int] = [3, 3]
forget_bias: float = 0.01

# 定义训练参数
# max_epochs: int = 
learning_rate: float = 0.001
batch_size: int = 1
test_bs: int=1
patch: int = 1
device: str = "cuda:0"
# device = torch.device("cuda:1")
# device_ids = [1,3]
test_frequency: int = 1
# out_len: int = 10
max_epochs = 100
start_save = 0

# result_root = 'results/60w-60h-1chn_5f_16*2hid_9697_teston98v2_fold'
# result_root = 'results/60w-60h-1chn_mix_bootstrap_valonoob/'
# result_root = 'results/60w-60h-1chn_mix_hotspot_nobootstrap/'
# result_root = 'results/60w-60h-1chn_5f_9697_teston98v2_fixed'
result_root = 'results_v3/60w-60h-1chn_mix_bootstrap/'
# result_root = 'results_log/60w-60h-1chn_mix_nobootstrap'


class ConvLSTMTrainer(TrainingTemplate):
    def check_data(self, data):
        inputs, labels, out_len = data
        # 这里是去除输入中的标签，以及多余的标签，因为有时候需要其他的，就放一起了
        # inputs.shape = [b, 1, s, c, h, w]
#         print('inputs:', inputs.shape)
        inputs = inputs[:,0].permute(0,1,2,4,3)
        labels = labels[:,0].permute(0,1,2,4,3)
#         print(inputs.ndim)
#         print('inputs:', inputs.shape)
        # patch 分割，编写增加 batch_size 的 trick
        inputs = reshape_patch(inputs, patch_size=patch)
        labels = reshape_patch(labels, patch_size=patch)
#         print('inputs:', inputs.shape)
        # 转换运算设备
#         print(self.device)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        out_len = out_len.to(self.device)
#         inputs = inputs.cuda(device=device_ids[0])
#         labels = labels.cuda(device=device_ids[0])
#         print('inputs.shape:', inputs.shape)
#         print('out_len.shape:', out_len.shape)
        return inputs, labels[:,:out_len], out_len

    def load(self):
#         self.model.load_state_dict(self.states["model"])
        self.model = self.states['modell']
        self.optimizer = self.states['opt']
#         self.optimizer.load_state_dict(self.states["optimizer"])
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
        # patch 分割，编写增加 batch_size 的 trick
        inputs = reshape_patch(inputs, patch_size=patch)
        labels = reshape_patch(labels, patch_size=patch)
        # 转换运算设备
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
#         self.model.load_state_dict(self.states["model"], strict=False)


def train(out_len, boot):
    # 加载数据集
#     train_set = RadarDataset2("train")
#     test_set = RadarDataset2("test_real_sequences")
#     train_set = AuroraDataset.AuroraDataset('tra_60/fold_'+str(k), out_len=out_len)
#     test_set = AuroraDataset.AuroraDataset('val_60/fold_'+str(k), out_len=out_len)
#     train_set = AuroraDatasetMix.AuroraDataset('train_60', in_len=5, out_len=out_len)
#     test_set = AuroraDatasetMix.AuroraDataset('test_60_total', in_len=5, out_len=out_len)
    train_set = AuroraDatasetMix.AuroraDataset('train_60_boots/'+str(boot)+'/tra',in_len=5, out_len=out_len)
    test_set = AuroraDatasetMix.AuroraDataset('train_60_boots/'+str(boot)+'/val',in_len=5, out_len=out_len)
#     train_set = AuroraDataset.AuroraDataset('tra_60/fixed')
#     test_set = AuroraDataset.AuroraDataset('val_60/fixed')
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False,
                              pin_memory=True)
    print('len(train_loader):', len(train_loader))
    
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False,
                             pin_memory=True)
    print('len(val_loader):', len(test_loader))

    # 创建网络模型
    model = ConvLSTM(in_channels=in_channels * patch ** 2, hidden_channels_list=hidden_channels_list,
                     kernel_size_list=kernel_size_list, forget_bias=forget_bias).to(device)
#     torch.cuda.set_device(1)
#     model = nn.DataParallel(model, device_ids=[0,1]).to(device)
#     model = BalancedDataParallel(bsize, model, device_ids=[0,1,2]).to(device)
#     model = DataParallelModel(model, device_ids=[0,1])
#     model = model.cuda(device=device_ids[0])
    
    # 定义损失函数
    criterion = nn.MSELoss()
#     criterion  = DataParallelCriterion(criterion, device_ids=[0,1])

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义参数保存位置
    to_save = Path(__file__).parent.joinpath(result_root+str(boot))
#     to_save = Path(__file__).parent.joinpath(result_root+str(out_len))
#     to_save = Path(__file__).parent.joinpath(result_root)
#     print(to_save)
    if not to_save.exists():
        to_save.mkdir()

    # 定义训练器
    trainer = ConvLSTMTrainer(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion,
                              optimizer=optimizer, lr_scheduler=None, max_epochs=max_epochs, device=device,
                              to_save=to_save, test_frequency=test_frequency, start_save=start_save, visualize=True, vis_name='60-bootstrap-valonoob-v3_'+str(boot))
    
#     trainer = ConvLSTMTrainer(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion,
#                               optimizer=optimizer, lr_scheduler=None, max_epochs=max_epochs, device=device,
#                               to_save=to_save, test_frequency=test_frequency, start_save=start_save, visualize=True, vis_name='60-bootstrap-valonoob-v3'+)

    # 训练
    trainer.run()


def valid():
    test_set = RadarDataset2("test_real_sequences")
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=True,
                             pin_memory=True)
    # 创建网络模型
    model = ConvLSTM(in_channels=in_channels * patch ** 2, hidden_channels_list=hidden_channels_list,
                     kernel_size_list=kernel_size_list, forget_bias=forget_bias).to(device)

    # 定义参数保存位置
    to_save = Path(__file__).parent.joinpath("results")

    tester = ConvLSTMTester(model=model, test_loader=test_loader, device=device, to_save=to_save)

    tester.run(out_len=out_len)
    
def test(out_len):
    test_set = AuroraDataset.AuroraDataset('test_60', out_len=out_len)
    
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False,
                                 pin_memory=True)
    model = ConvLSTM(in_channels=in_channels * patch ** 2, hidden_channels_list=hidden_channels_list,
                     kernel_size_list=kernel_size_list, forget_bias=forget_bias).to(device)
    
    to_save = Path(__file__).parent.joinpath(result_root+str(out_len))
#     to_save = Path(__file__).parent.joinpath(result_root)

    tester = ConvLSTMTester(model=model, test_loader=test_loader, device=device, to_save=to_save)

    tester.run(out_len=out_len)
#     print(prediction.shape)

def test_long_seq(out_len, boot):
#     npz_root = '/home/jjn/susan/AuroraPrediction_v2/data/exp_npzs/test_60_long_mix_v2'
    npz_root = '/home/jjn/susan/AuroraPrediction_v2/data/exp_npzs_v3/test_60'
#     npz_root = '/home/jjn/susan/AuroraPrediction_v2/data/exp_npzs/test_60_long_mix_lowkp'

    length_list = os.listdir(npz_root)
    length_list.sort()
    for length in length_list[:20]:
        print('length:', length)
        out_length = int(length)-5
        test_set = AuroraDatasetMix.AuroraDataset('test_60/'+length, in_len=5,out_len=out_length, istest=True)

        test_loader = DataLoader(dataset=test_set, batch_size=test_bs, num_workers=4, shuffle=False, drop_last=False,
                                     pin_memory=True)
#         print('test_loader:', len(test_loader))
        model = ConvLSTM(in_channels=in_channels * patch ** 2, hidden_channels_list=hidden_channels_list,
                         kernel_size_list=kernel_size_list, forget_bias=forget_bias).to(device)

#         to_save = Path(__file__).parent.joinpath(result_root)
        to_save = Path(__file__).parent.joinpath(result_root+str(boot))
        test_save = Path(__file__).parent.joinpath(to_save,'long_test')
        if not test_save.exists():
            test_save.mkdir()
        tester = ConvLSTMTester(model=model, test_loader=test_loader, device=device, to_save=to_save)

        tester.run(out_len=out_length)

if __name__ == '__main__':
    for boot in range(37,38):
        train(20, boot)
        test_long_seq(20, boot)
#     train(20, 0)
#     test_long_seq(20, 0)
        
        
        
