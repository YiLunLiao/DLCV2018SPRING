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
import skvideo.io

def sample_interval(length, points):
    interval = float(length)/float(points)
    indexes = []
    for i in range(points):
        index = int(interval*(i+1) - 1)
        indexes.append(index)
    return indexes

# only extract data without label
class Vd2ImgDataset(Dataset):
    def __init__(self, dir, max_frame=16, num_classes=11, file_list='gt_valid.csv'):
        self.dir = dir
        self.num_classes = num_classes
        self.max_frame = max_frame
        self.file_list = file_list  # csv file
        
        # create video names
        self.video_list = list() # relative path + filename
        self.file_data = pd.read_csv(self.file_list, sep=',', encoding='ISO-8859-1', 
                             usecols=['Video_name', 'Action_labels'])     
        self.video_list = self.file_data['Video_name'].values  
        self.label_data = self.file_data['Action_labels'].values                         
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_dir = self.video_list[idx].split('-')
        video_dir = video_dir[0:3]
        video_dir = '-'.join(video_dir)
        video_filename = ' '
        
        # search for the correct mp4 file
        vd_files = os.listdir(os.path.join(self.dir, video_dir))
        for i in range(len(vd_files)):
            temp = vd_files[i].split('-')
            temp = '-'.join(temp[0:5])
            if temp == self.video_list[idx]:
                video_filename = vd_files[i]
                break
        
        #print('    Open:', os.path.join(self.dir, video_dir, video_filename))
        video_raw = skvideo.io.vread(os.path.join(self.dir, video_dir, video_filename))
        frames = list()
        for i in range(video_raw.shape[0]):
            frames.append(transform.resize(video_raw[i], (299, 299, 3)))
        video = np.array(frames)
        
        #print('Video shape:', video.shape)
        video = video.transpose((0, 3, 1, 2))
        if video.shape[0] > self.max_frame and self.max_frame > 0:
            print('Trim video from:', video.shape[0], 'to', self.max_frame)
            sample_index = sample_interval(video.shape[0], self.max_frame)
            video = video[sample_index]
        
        video = torch.Tensor(video)
        
        Label = torch.LongTensor([int(self.label_data[idx])])
        
        #print(idx)
        
        return {'X': video, 'Y': Label}
    
if __name__ == '__main__':
    print(' ')