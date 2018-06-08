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
from skimage import io, transform

''' 
for seq2seq model training
'''


def sample_interval(length, points):
    interval = float(length)/float(points)
    indexes = []
    for i in range(points):
        index = int(interval*(i+1) - 1)
        indexes.append(index)
    return indexes


# get only a portion of the whole videos 
class splitLongVdDataset(Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir  = data_dir
        self.label_dir = label_dir
        self.videos    = os.listdir(data_dir)
        self.H_re = 299
        self.W_re = 299
        #assert len(self.video_list) == self.len_data
                        
    def __len__(self):
        return len(self.videos)
    
    def getVideoLen(self, idx):
        video_dir = os.path.join(self.data_dir, self.videos[idx])
        # video data in the form of multiple images
        video_files = os.listdir(video_dir) # multiple .jpg files
        return len(video_files)
    
    def getVideoFrames(self, idx, frame_idx):
        '''
        idx: video index
        frame_idx: frame index (list)
        '''
        video_dir = os.path.join(self.data_dir, self.videos[idx])
        
        # video data in the form of multiple images
        video_files = os.listdir(video_dir) # multiple .jpg files
        video_files.sort()
        img_stk = []
        img_collection = io.imread_collection([os.path.join(video_dir, video_files[i_img]) for i_img in frame_idx])
        
        for i_img in range(len(img_collection)):
            #img = io.imread(os.path.join(video_dir, video_files[i_img]))
            img = img_collection[i_img]
            img = transform.resize(img, [self.H_re, self.W_re, 3])
            img = img.astype(np.float32)
            img = img/255
            img_stk.append(img)
            #print(i_img)
        img_stk = np.stack(img_stk)
        #print('    image stack shape', img_stk.shape)
        
        # video label
        label_file = open(os.path.join(self.label_dir, self.videos[idx]) + '.txt', 'r')
        label_data = label_file.readlines()
        label_data = np.array(label_data, dtype=np.int)
        label_data = label_data[frame_idx]
        label_file.close()
        
        assert label_data.shape[0] == img_stk.shape[0]
        
        # convert to pytorch format
        img_stk = img_stk.transpose((0, 3, 1, 2))
        img_stk = torch.Tensor(img_stk)
        label_data = torch.LongTensor(label_data)
        
        return {'X': img_stk, 'Y': label_data}
    
    def __getitem__(self, idx):
        video_dir = os.path.join(self.data_dir, self.videos[idx])
        
        # video data in the form of multiple images
        video_files = os.listdir(video_dir) # multiple .jpg files
        video_files.sort()
        img_stk = []
        
        img_collection = io.imread_collection([os.path.join(video_dir, video_files[i_img]) for i_img in range(len(video_files))])
        for i_img in range(len(video_files)):
            img = img_collection[i_img]
            img = transform.resize(img, [self.H_re, self.W_re, 3])
            img = img.astype(np.float32)
            img = img/255
            img_stk.append(img_stk)
            print(i_img)
        img_stk = np.stack(img_stk)
        
        # video label
        print(os.path.join(self.label_dir, self.videos[idx]) + '.txt')
        label_file = open(os.path.join(self.label_dir, self.videos[idx]) + '.txt', 'r')
        label_data = label_file.readlines()
        label_file.close()
        label_data = np.array(label_data, dtype=np.int)
        print('Read label')
        
        
        assert label_data.shape[0] == img_stk.shape[0]
        
        # convert to pytorch format
        img_stk = img_stk.transpose((0, 3, 1, 2))
        img_stk = torch.Tensor(img_stk)
        label_data = torch.LongTensor(label_data)
        
        return {'X': img_stk, 'Y': label_data}
        
if __name__ == '__main__':
    # data directory
    data_dir =  'HW5_data/FullLengthVideos/videos/train'
    label_dir = 'HW5_data/FullLengthVideos/labels/train'
    
    dataset = splitLongVdDataset(data_dir=data_dir, label_dir=label_dir)
    print('Dataset len:', len(dataset))
    
    print('video[0] len:', dataset.getVideoLen(0))
    frame_idx = list(range(200))
    sample = dataset.getVideoFrames(0, frame_idx)
    print('image stack shape:', sample['X'].shape)
    print('label shape:      ', sample['Y'].shape)
    
    #sample = dataset[0]
    #print('image stack shape:', sample['X'].shape)
    #print('label shape:      ', sample['Y'].shape)
    
    