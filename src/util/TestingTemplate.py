from pathlib import Path
from typing import Union

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

__all__ = ["TestingTemplate"]


class TestingTemplate(object):
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: str = None, to_save: Union[str, Path] = None):
        if device is None:
            self.device = "cuda:0,1" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = model.to(device)
        self.test_loader = test_loader

        self.to_save = Path(to_save)  # 模型参数保存位置

        try:
            self.states = torch.load(
                str(self.to_save.joinpath("checkpoint.pth")),
                map_location=lambda storage, loc: storage if self.device == "cpu" else storage.cuda(device)
            )
        except:
            raise FileNotFoundError(f"{self.to_save.joinpath('checkpoint.pth')}不存在！")

    def check_data(self, data):
        # 是否需要进行处理，默认加载出来的就直接用
        # 如需定制，需重写方法
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        return inputs, labels

    def check_data_back(self, labels, outputs):
        return labels, outputs

    def forward(self, inputs, out_len: int = 10) -> Tensor:
        # 对正向传播如有其他操作，可以重写该方法
#         print('inputs:', inputs.shape)
        return self.model(inputs, out_len=out_len)

    def load(self):
        # 加载存储的字典，默认是参数存储，如需模型存储需重写 load() 和 set_states() 方法
        self.model.load_state_dict(self.states["model"])

    def run(self, out_len: int = 10):
        ground_truth, prediction, last_frames = self.test(out_len=out_len)

    def test(self, out_len: int = 10):
        self.load()
        self.model.eval()

        ground_truth = []
        prediction = []
        last_frames = []

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
#                 print(f"第{i + 1}次测试迭代")
                inputs, labels, out_len = self.check_data(data)
#                 print(inputs.shape)
                last_frame = inputs[:,-1]

                outputs = self.forward(inputs, out_len=out_len)

                labels, outputs = self.check_data_back(labels=labels, outputs=outputs)
                prediction.append(outputs)
                ground_truth.append(labels)
                last_frames.append(last_frame)

        ground_truth = torch.cat(ground_truth, dim=0).cpu()
        prediction = torch.cat(prediction, dim=0).cpu()
        last_frames = torch.cat(last_frames, dim=0).cpu()
#         print(prediction.shape, last_frames.shape)
        
        if not self.to_save.joinpath('long_test',str(out_len.cpu().item())).exists():
            self.to_save.joinpath('long_test',str(out_len.cpu().item())).mkdir()
        
        torch.save(ground_truth, self.to_save.joinpath('long_test',str(out_len.cpu().item())+"/ground_truth.pth"))
        torch.save(prediction, self.to_save.joinpath('long_test',str(out_len.cpu().item())+"/prediction.pth"))
        torch.save(last_frames, self.to_save.joinpath('long_test',str(out_len.cpu().item())+"/last_frames.pth"))

        return ground_truth, prediction, last_frames

        # torch.cuda.empty_cache()  # 清空 GPU 内存
