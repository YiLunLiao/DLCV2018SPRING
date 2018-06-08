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

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype=np.long)[y]

# convert old labels into new labels
def old2label_12(old_label):
    label_matrix = [ 0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 0, 0, 0, 12 ]
    return label_matrix[old_label]

def old2label_10(old_label):
    label_matrix = [ 0, 1, 2, 3, 4, 0, 5, 0, 6, 7, 0, 0, 0, 8, 0, 0, 9, 0, 0, 0, 10 ]

    return label_matrix[old_label]


class VdDataset(Dataset):
    def __init__(self, dir, csv_file, max_frame=-1, augment=False, num_classes=11):
        self.dir = dir
        self.num_classes = num_classes
        self.max_frame = max_frame
        
        # create label dict
        self.label_filename = csv_file #'GTEA_short/gt_train.csv'
        self.label_data = pd.read_csv(self.label_filename, sep=',', encoding='ISO-8859-1', 
                             usecols=['Video_name', 'Action_labels'])
        self.label_dict = {}
        for i in range(len(self.label_data['Video_name'].values)):
            self.label_dict[self.label_data['Video_name'].values[i]] = self.label_data['Action_labels'].values[i]
        self.len_data = len(self.label_dict)
        
        # create video names
        self.video_list = list() # relative path + filename
        self.video_dir = os.listdir(self.dir)
        for i_dir in range(len(self.video_dir)):
            files = os.listdir(os.path.join(self.dir, self.video_dir[i_dir]))
            for i_file in range(len(files)):
                video_name = files[i_file]
                self.video_list.append(os.path.join(self.dir, self.video_dir[i_dir], video_name))
                
        assert len(self.video_list) == self.len_data
                
        
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        
        video_raw = skvideo.io.vread(self.video_list[idx])
        frames = list()
        for i in range(video_raw.shape[0]):
            frames.append(transform.resize(video_raw[i], (299, 299, 3)))
        video = np.array(frames)
        #print('Video shape:', video.shape)
        video = video.transpose((0, 3, 1, 2))
        if video.shape[0] > self.max_frame and self.max_frame > 0:
            sample_index = sample_interval(video.shape[0], self.max_frame)
            #video = video[sample_index]
            print('Exceed max frame limit: trim from', video.shape[0], 'to', self.max_frame) 
            video = video[sample_index]
        video = torch.Tensor(video)
        
        video_name = self.video_list[idx]
        video_name = (video_name.split('/')[-1])
        video_name = (video_name.split('.')[0])
        temp = video_name.split('-')
        temp = temp[0:len(temp)-2]
        video_name = '-'.join(temp)
        #print('Video name:', video_name)
        Label = self.label_dict[video_name]
        #print('Label in dataset.get_item():', Label)
        if self.num_classes == 11:
            Label = torch.LongTensor([int(old2label_10(int(Label)))])
        elif self.num_classes == 13:
            Label = torch.LongTensor([int(old2label_12(int(Label)))])
        else:
            Label = torch.LongTensor([int(Label)])
        
        return {'X': video, 'Y': Label}
        
if __name__ == '__main__':
    dataset = VdDataset(dir='GTEA_short/train', csv_file='GTEA_short/gt_train.csv')
    print('Dataset len:', len(dataset))
    sample = dataset[0]
    print('Video data shape:', sample['X'].shape)
    print('Label:', sample['Y'])
    sample = sample['X']
    sample = sample.numpy()
    sample = sample.transpose((0, 2, 3, 1))
    print('Video to be saved shape:', sample.shape)
    
    writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
    for i in range(sample.shape[0]):
        writer.writeFrame(sample[i, :, :, :])
    writer.close()
