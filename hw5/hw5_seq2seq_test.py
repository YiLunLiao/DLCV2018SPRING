from inception_model_noBN import *
from vd_dataloader import  *
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
from longVd_dataloader import *
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, transform

def writePreds(filename, data):
    file = open(filename, 'w')
    print('Write', filename)
    for i in range(data.shape[0]):
        file.write(str(int(data[i])))
        file.write('\n')
    file.close()
    

def update_progress_loss(progress, loss1, loss2):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}%:  loss = {2:.3f}, acc = {3:.2f}%".format( "#"*block + "-"*(barLength-block), round(progress*100, 3), loss1, loss2*100)
    sys.stdout.write(text)
    sys.stdout.flush()


def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}%".format( "#"*block + "-"*(barLength-block), round(progress*100, 3))
    sys.stdout.write(text)
    sys.stdout.flush()


# get only a portion of the whole videos 
class splitLongVdDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir  = data_dir
        self.videos    = os.listdir(data_dir)
        self.H_re = 299
        self.W_re = 299
              
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
            img = img_collection[i_img]
            img = transform.resize(img, [self.H_re, self.W_re, 3])
            img = img.astype(np.float32)
            img = img/255
            img_stk.append(img)
        img_stk = np.stack(img_stk)
        
        # convert to pytorch format
        img_stk = img_stk.transpose((0, 3, 1, 2))
        img_stk = torch.Tensor(img_stk)
        
        return {'X': img_stk}
    
    def getVideoName(self, idx):
        return self.videos[idx]
    
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
        
        # convert to pytorch format
        img_stk = img_stk.transpose((0, 3, 1, 2))
        img_stk = torch.Tensor(img_stk)
        
        return {'X': img_stk}


class actRecog_RNN_VarLen_classifier(nn.Module):
    def __init__(self):
        # input: batch_size, num_frame, 2048
        self.drop_rate = 0.2    
        super(actRecog_RNN_VarLen_classifier, self).__init__()
        self.gru = nn.GRU(2048, 256, 2, batch_first=True, dropout=self.drop_rate, bidirectional=True)
        self.fc = nn.Linear(512, 11)
        
    def forward(self, x, hidden=None):
        out, _ = self.gru(x, hidden)
        #print('LSTM output shape:', out.shape)
        out = out[:, -1, :]
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    
    def forward_gru(self, x, hidden=None):
        _, hidden_out = self.gru(x, hidden)
        return hidden_out
    
    def forward_multiple(self, x, hidden=None):
        out, _ = self.gru(x, hidden)
        out = out.view(-1, 512)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    
    
    
def testSeq2Seq(cnn_model, rnn_model, dataset, batch_size, guid, out_dir):

    assert batch_size == 1
    
    order = list(range(len(dataset)))
    
    for i_vd in range(len(dataset)):
        print('Video', i_vd+1, '/', len(dataset))
        preds = testOneSeq(cnn_model, rnn_model, dataset, batch_size, guid, order[i_vd])
        filename = dataset.getVideoName(order[i_vd]) + '.txt'
        writePreds(os.path.join(out_dir, filename), preds)
    

def testOneSeq(cnn_model, rnn_model, dataset, batch_size, guid, i_vd):
    ''' 
    test one long sequence 
    '''
    
    frame_batch = 8 # simulataneous prediction of frame_batch frames
    interval = 64
    vd_len = dataset.getVideoLen(i_vd)
    steps = math.ceil(vd_len/interval)
    
    feature_list = list()
    
    print('Length =', vd_len)
    
    for i_step in range(steps):

        # get at most train_interval frames for seq2seq training
        start_frame = i_step*interval
        end_frame   = (i_step+1)*interval
        if end_frame >= vd_len:
            end_frame = vd_len
        frame_idx = list(range(start_frame, end_frame))
        sample = dataset.getVideoFrames(i_vd, frame_idx)
        
        x = Variable(sample['X'])
        if guid >= 0:   # move to GPU
            x = x.cuda(guid)          # (num_frame, 3, H, W)

        # split to different sub-batch
        split_x = torch.split(x, frame_batch, dim=0) # tuple
        
        for i in range(len(split_x)):
            CNN_feature = cnn_model(split_x[i])
            CNN_feature = CNN_feature.data.cpu()
            feature_list.append(CNN_feature)

        split_x = 0
        x = 0
            
        update_progress(i_step/steps)
            
    # (num_frame, 2048)
    temp_feature = torch.cat(feature_list, dim=0) 
    temp_feature = temp_feature.unsqueeze_(0)
    temp_feature = Variable(temp_feature).cuda(guid)
    
    print('    temp_feature shape:', temp_feature.shape)
    
    out = rnn_model.forward_multiple(temp_feature)
    print('    out shape:         ', out.shape)
    
    _, preds = torch.max(out.data, 1)

    feature_list = list()
    temp_feature = 0
    split_x = 0
        
    return preds
    

if __name__ == '__main__':
    # file directories
    save_dir = 'torch_model/seq2seq'
    load_dir = 'torch_model/baseline'
    save_interval = int(1)
    batch_size = 1
    epochs = 1000
    guid = 0
    
    # data directory
    data_dir = sys.argv[1]
    out_dir  = sys.argv[2]
    print('Data dir:  ', data_dir)
    print('Output dir:', out_dir)
    
    dataset = splitLongVdDataset(data_dir=data_dir)


    # load models
    cnn_model = torch.load('seq2seq_cnn.pt')
    rnn_model = torch.load('seq2seq_rnn.pt')
    cnn_model = cnn_model.cuda(guid)
    rnn_model = rnn_model.cuda(guid)

    cnn_model.eval()
    rnn_model.eval()
    
    testSeq2Seq(cnn_model, rnn_model, dataset, batch_size, guid, out_dir)
    
   

        

    
