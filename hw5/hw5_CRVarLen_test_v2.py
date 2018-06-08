from inception_model_noBN import *
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
import os
from skimage import io, transform
import skvideo.io
import numpy as np
import pandas as pd
from hw5_CRVarLen_train_v2 import *

def sample_interval(length, points):
    interval = float(length)/float(points)
    indexes = []
    for i in range(points):
        index = int(interval*(i+1) - 1)
        indexes.append(index)
    return indexes

def writePreds(filename, data):
    file = open(filename, 'w')
    print('Write', filename)
    for i in range(data.shape[0]):
        file.write(str(int(data[i])))
        file.write('\n')
    file.close()

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
    
def update_progress_acc(progress, acc):
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
    text = "\rPercent: [{0}] {1:.2f}%, acc = {2:.3f}%".format( "#"*block + "-"*(barLength-block), round(progress*100, 3), acc)
    sys.stdout.write(text)
    sys.stdout.flush()
    
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
    
def testCRNN_VarLen(cnn_model, rnn_model, dataloader, batch_size, guid):

    assert batch_size == 1
    frame_batch = 8 # simulataneous prediction of frame_batch frames
    
    preds = list()
    
    num_example = 0.0
    num_acc = 0.0

    cnn_model.eval()
    rnn_model.eval()

    for batch_i, sample in enumerate(dataloader):
        x = Variable(sample['X'])
        label = (sample['Y'])
        
        if guid >= 0:   # move to GPU
            x = x.cuda(guid)
            

        # (1, num_frame, 3, H, W) -> (num_frame, 3, H, W)
        x.squeeze_(0)
        #print('x shape:', x.shape)

        # split to different sub-batch
        split_x = torch.split(x, frame_batch, dim=0) # tuple
        feature_list = list()
        for i in range(len(split_x)):
            CNN_feature = cnn_model(split_x[i])
            CNN_feature = CNN_feature.data.cpu()
            feature_list.append(CNN_feature)
            
        # (num_frame, 2048)
        temp_feature = torch.cat(feature_list, dim=0) 
        
        # (1, num_frame, 2048)
        temp_feature = temp_feature.unsqueeze_(0)
        temp_feature = Variable(temp_feature).cuda(guid)

        #print('Temp feature shape:', temp_feature.shape)
        
        y = rnn_model.forward(temp_feature)
    
        
        _, pred = torch.max(y.data, 1)
        
        label = label.squeeze_()
        label = Variable(label).cuda(guid)
        num_acc = num_acc + torch.sum(pred == label.data)
        num_example = num_example + label.shape[0]
        
        #print('Acc:', num_acc/num_example*100)
        
        preds.extend(pred)

        update_progress_acc(batch_i/len(dataloader), num_acc/num_example*100)
        #torch.save(label.data.cpu(), 'trim_torch_data/test/label_{}.pt'.format(batch_i))
        #torch.save(temp_feature.data.cpu(), 'trim_torch_data/test/data_{}.pt'.format(batch_i))
        #print('save trim_torch_data/test/label_{}.pt'.format(batch_i))
        y = 0
        
    preds = torch.Tensor(preds)
    return preds

def testCRNN_VarLen_fromData(cnn_model, rnn_model, dataloader, batch_size, guid):

    assert batch_size == 1
    frame_batch = 8 # simulataneous prediction of frame_batch frames
    
    preds = list()
    
    num_example = 0.0
    num_acc = 0.0

    cnn_model.eval()
    rnn_model.eval()

    for batch_i in range(len(dataloader)):
        
        # (1, num_frame, 2048)
        temp_feature = torch.load('trim_torch_data/valid/data_{}.pt'.format(batch_i))
        temp_feature = Variable(temp_feature).cuda(guid)
        
        label = torch.load('trim_torch_data/valid/label_{}.pt'.format(batch_i))
        label = Variable(label).cuda(guid)

        #print('Temp feature shape:', temp_feature.shape)
        
        y = rnn_model.forward(temp_feature)
    
        
        _, pred = torch.max(y.data, 1)
        
        num_acc = num_acc + torch.sum(pred == label.data)
        num_example = num_example + label.shape[0]
        
        #print('Acc:', num_acc/num_example*100)
        
        preds.extend(pred)

        update_progress_acc(batch_i/len(dataloader), num_acc/num_example*100)
        y = 0
        
    preds = torch.Tensor(preds)
    return preds
    

if __name__ == '__main__':

    batch_size = 1
    guid = 0
    save_dir = 'torch_model/baseline'
    
    vd_dir =  sys.argv[1]
    csv_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    print('Video dir:', vd_dir)
    print('CSV dir:', csv_dir)
    print('Output dir:', output_dir)
    
    cnn_model = torch.load('crvr_cnn.pt')
    rnn_model = torch.load('crvr_rnn.pt')
    cnn_model = cnn_model.cuda(guid)
    rnn_model = rnn_model.cuda(guid)
    
    # create dataset
    dataset = Vd2ImgDataset(vd_dir, max_frame=1000, num_classes=11, file_list=csv_dir)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    
    preds = testCRNN_VarLen(cnn_model, rnn_model, dataloader, batch_size, guid)
    #preds = testCRNN_VarLen_fromData(cnn_model, rnn_model, dataloader, batch_size, guid)
    writePreds(os.path.join(output_dir, 'p2_result.txt'), preds)
    