import os
import sys
sys.path.append('/home/jjn/susan/DeepLab/src')
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from util.congfig import load_dataset_root
# from data.seq_generator import *
# from data.train_corr_scripts import *

__all__ = ["AuroraDataset"]
DATASET_ROOT = '/home/jjn/susan/AuroraPrediction_v2/data/exp_npzs_v3/'
# DATASET_ROOT = '/home/jjn/susan/AuroraPrediction_v2/data/exp_hpst_npzs/'

def get_sequence(npz_path, npz_file, input_frames=5, img_size=60, img_chns=1):
    raw_seq = np.load(os.path.join(npz_path, npz_file))['images']  # load array
    raw_len = raw_seq.shape[0]
    raw_seq = raw_seq.reshape(1, raw_len, img_size, img_size, img_chns)
    
    basic_imgs = raw_seq[:,:input_frames]
    next_imgs = raw_seq[:,input_frames:]
    
    basic_imgs = np.transpose(basic_imgs, (0,1,4,3,2))
    next_imgs = np.transpose(next_imgs, (0,1,4,3,2))
    
    return basic_imgs, next_imgs


class AuroraDataset(Dataset):

    def __init__(self, dataset: str, in_len: int=5, out_len: int = 20, istest=False):
        super().__init__()
#         try:
#             self.dataset_dir = load_dataset_root("aurora").joinpath(dataset)
#         except:
#             raise ValueError(f"数据集 {dataset} 不存在")
        self.dataset_dir = os.path.join(DATASET_ROOT,dataset)

        self.dataset = dataset
        self.count = len(os.listdir(self.dataset_dir))
        self.out_len = out_len
        self.in_len = in_len

    def __getitem__(self, item):

        npz_file = str(item)+'.npz'

        data_basic, data_next = get_sequence(self.dataset_dir, npz_file, input_frames=self.in_len)
        inputs = torch.tensor(data_basic)
        labels = torch.tensor(data_next)
        
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        
#         print('before cat:', inputs.shape, labels.shape)
        
        out_len = torch.tensor(labels.shape[1])
        out_len = out_len.type(torch.IntTensor)
#         print('out_len type:', out_len.type())
        if labels.shape[1]<20:
            labels = torch.cat((labels, torch.zeros(1,20-labels.shape[1], labels.shape[2], labels.shape[3], labels.shape[4])), 1)
#         print('after cat:', inputs.shape, labels.shape)
        
        return inputs[:1, :self.in_len], labels[:1], out_len

    def __len__(self):
        return self.count

    
# class TestDataset(Dataset):

#     def __init__(self, dataset: str):
#         super().__init__()
#         self.dataset = torch.rand(24, 20, 1, 400, 400)

#     def __getitem__(self, item):
#         return self.dataset[item, ], self.dataset[item, 10:]

#     def __len__(self):
#         return self.dataset.shape[0]

if __name__ == '__main__':
    train_set = AuroraDataset('mix_train')
    train_loader = DataLoader(dataset=train_set, batch_size=1, num_workers=4, shuffle=True, drop_last=False,
                                  pin_memory=True)
    print('len(train_loader):', len(train_loader))