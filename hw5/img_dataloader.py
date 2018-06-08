from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import pandas as pd 
from skimage import io, transform
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from typing import Tuple


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype=np.long)[y]


class ImgDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.labels = np.load(os.path.join(dir, 'Label.npy'))
        self.len_data = len(self.labels)
        self.num_classes = 21
    def __len__(self):
        return self.len_data
    def __getitem__(self, idx):
        img_idx = idx
        img_name = os.path.join(self.dir, 'img_stk' + str(img_idx) + '.npy')
        img_stk = np.load(img_name)
        img_stk = img_stk.transpose((0, 3, 1, 2))
        
        img_pt = torch.from_numpy(img_stk)
        label = self.labels[idx]
        label = torch.LongTensor([int(label)])
        #label = to_categorical(label, self.num_classes)
        #label = torch.from_numpy(label)
        #label = label.type(torch.LongTensor)
        
        return {'Y':label, 'X':img_pt}