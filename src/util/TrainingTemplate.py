import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
from visdom import Visdom
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.metrics._ssim import ssim
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
__all__ = ["TrainingTemplate"]


class TrainingTemplate(object):
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                 criterion: nn.Module, optimizer: Optimizer, lr_scheduler=None, max_epochs: int = 2000,
                 device: str = None, to_save: Union[str, Path] = None, test_frequency: int = 5,
                 start_save: int = 100, visualize: bool = False, vis_name: str = ''):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.max_epochs = max_epochs
        self.to_save = Path(to_save)  # 模型参数保存位置
        self.test_frequency = test_frequency
        self.start_save = start_save
        self.visualize = visualize
        self.vis_name = vis_name

        if self.to_save.joinpath("temp.pth").exists():
            self.states = torch.load(
                str(self.to_save.joinpath("temp.pth")),
                map_location=lambda storage, loc: storage if self.device == "cpu" else storage.cuda(device)
            )
        else:
            self.states = None
        
#         if self.to_save.joinpath("checkpoint.pth").exists():
#             self.states = torch.load(
#                 str(self.to_save.joinpath("checkpoint.pth")),
#                 map_location=lambda storage, loc: storage if self.device == "cpu" else storage.cuda(device)
#             )
#         else:
#             self.states = None

        self.start_epoch = 0

        # 固定随机数种子，方便复现
        torch.manual_seed(0)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(0)

    def check_data(self, data):
        # 是否需要进行处理，默认加载出来的就直接用
        # 如需定制，需重写方法
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        return inputs, labels

    def forward(self, inputs, out_len: int = 10) -> Tensor:
        # 对正向传播如有其他操作，可以重写该方法
        return self.model(inputs, out_len=out_len)

    def load(self):
        # 加载存储的字典，默认是参数存储，如需模型存储需重写 load() 和 set_states() 方法
        self.model.load_state_dict(self.states["model"])
        self.optimizer.load_state_dict(self.states["optimizer"])
        self.start_epoch = self.states["epoch"]
        if "lr_scheduler" in self.states and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(self.states["lr_scheduler"])

    def set_states(self, epoch, loss):
        # 加载存储的字典，默认是参数存储，如需模型存储需重写 load() 和 set_states() 方法
        states = {
            "epoch": epoch + 1,
            "loss": loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        if self.lr_scheduler is not None:
            states["lr_scheduler"] = self.lr_scheduler.state_dict()

        return states

    def save(self, epoch, loss):
        states = self.set_states(epoch, loss)
        torch.save(obj=states, f=self.to_save.joinpath("temp.pth"))
#         torch.save(obj=states, f=self.to_save.joinpath(str(epoch)+".pth"))

        if self.is_best(loss=loss):
            shutil.copy(self.to_save.joinpath("temp.pth"), self.to_save.joinpath("checkpoint.pth"))

    def is_best(self, loss):
        # 判断最优模型的规则，默认根据 loss 判断，越小越好
        if self.to_save.joinpath("checkpoint.pth").exists():
            best_loss = torch.load(str(self.to_save.joinpath("checkpoint.pth")))["loss"]
        else:
            best_loss = math.inf

        return loss < best_loss

    def run(self):
        if self.states is not None:
            print(self.states.keys())
            self.load()

        if self.visualize:
            viz = Visdom(env=self.vis_name)

        for epoch in range(self.start_epoch, self.max_epochs):
#             print('epoch:', epoch)
            self.model.train()
            with tqdm(
                    iterable=self.train_loader, ncols=100,
           
                bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',
            ) as t:
                start_time = datetime.now()
                loss_list = []
                ssim_list = []
                
#                 print(start_time)
                for batch, data in enumerate(self.train_loader):
#                     print(batch)
                    t.set_description_str(f"\33[36m【Epoch {epoch + 1:04d}】")
                    # 训练代码

                    self.optimizer.zero_grad()

                    inputs, labels, out_len = self.check_data(data)
                
#                     out_len = labels.shape[2]

                    outputs = self.forward(inputs, out_len)
#                     print('output:', outputs.shape)
                    loss = self.criterion(outputs, labels)
#                     print('loss:', loss.shape)
                    
#                     for i in range(loss.shape[0]):
                    loss_list.append(loss.detach().cpu().item())
                    # 反向传播
                    loss.backward()

                    # 梯度裁剪，防止梯度爆炸
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)

                    # 优化器更新
                    self.optimizer.step()

                    # 学习率更新
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    cur_time = datetime.now()
                    delta_time = cur_time - start_time
                    mean_loss = sum(loss_list) / len(loss_list)
                    
                    for i in range(outputs.shape[0]):
                        ssim_list.append(ssim(outputs[i], labels[i]))
                    mean_ssim = sum(ssim_list) / len(ssim_list)

                    t.set_postfix_str(f"train_loss={mean_loss:.6f}, train_ssim={mean_ssim:.4f}, duration：{delta_time}\33[0m")

                    t.update()

            epoch_loss = sum(loss_list) / len(loss_list)
            epoch_ssim = sum(ssim_list) / len(ssim_list)

            if self.visualize:
                # noinspection PyUnboundLocalVariable
                viz.line([epoch_loss], [epoch], win='train_loss', opts=dict(title='train_loss'),
                         update=None if epoch == 0 else 'append')
                viz.line([float(epoch_ssim)], [epoch], win='train_ssim', opts=dict(title='train_ssim'),
                         update=None if epoch == 0 else 'append')

            self.model.eval()

            if (epoch + 1) % self.test_frequency == 0:
                with tqdm(
                        iterable=self.test_frequency,
                        bar_format='{desc} {postfix}',
                ) as t:
                    # 测试
                    test_loss, test_ssim = self.__test()

                    t.set_description_str(f"\33[35m【validation】")
                    t.set_postfix_str(f"val_loss={test_loss:.6f}, val_ssim={test_ssim:.4f}\33[0m")

                t.update()

                if self.visualize:
                    viz.line([test_loss], [epoch + 1], win='validation_loss', opts=dict(title='val_loss'),
                             update=None if epoch + 1 == self.test_frequency else 'append')
                    viz.line([float(test_ssim)], [epoch + 1], win='validation_ssim', opts=dict(title='val_ssim'),
                             update=None if epoch + 1 == self.test_frequency else 'append')

            if epoch + 1 > self.start_save:
                self.save(epoch, test_loss)
            
        if self.visualize:
            viz.save([self.vis_name])

        torch.cuda.empty_cache()  # 清空 GPU 内存

    def __test(self):
        test_loss_epoch = []
        test_ssim_epoch = []
        with torch.no_grad():
            for j, data in enumerate(self.test_loader):
                inputs, labels, out_len = self.check_data(data)

                outputs = self.forward(inputs, out_len)
                test_loss = self.criterion(outputs, labels)
                test_loss_epoch.append(test_loss.item())
                
                for i in range(outputs.shape[0]):
                    test_ssim_epoch.append(ssim(outputs[i], labels[i]))

        loss = sum(test_loss_epoch) / len(test_loss_epoch)
        mssim = sum(test_ssim_epoch) / len(test_ssim_epoch)

        return loss, mssim


# Common practise for initialization.
def weights_init(layer):
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
