from inception_model import *
from img_dataloader import  *
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
import pandas as pd
import skvideo.io
from skimage import io, transform
import numpy as np
import os

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

def convert2MultipleFrames(dataloader):
    for batch_i, sample in enumerate(dataloader):
        x = (sample['X'])
        label = (sample['Y'])
        x = x.squeeze_()
        label = label.squeeze_()
        torch.save(label, 'trimFrame_torch_data/label_{}.pt'.format(batch_i))
        torch.save(x, 'trimFrame_torch_data/data_{}.pt'.format(batch_i))
        
        print('save trimFrame_torch_data/label_{}.pt'.format(batch_i))
        update_progress(batch_i/len(dataloader), round(0), (0))

#from Vd2Img_dataset import * 
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
            #print('Trim video from:', video.shape[0], 'to', self.max_frame)
            sample_index = sample_interval(video.shape[0], self.max_frame)
            video = video[sample_index]
        
        video = torch.Tensor(video)
        
        Label = torch.LongTensor([int(self.label_data[idx])])
        
        #print(idx)
        
        return {'X': video, 'Y': Label}
    
''' load from .pt '''
class ptDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = os.listdir(self.dir) 
        self.len = int(len(self.files)/2 )                     
        
    def __len__(self):
        return (self.len)
    
    def __getitem__(self, idx):
        video = torch.load(os.path.join(self.dir, 'data_{}.pt'.format(idx)))
        Label = torch.load(os.path.join(self.dir, 'label_{}.pt'.format(idx)))
        return {'X': video, 'Y': Label}
    
def update_progress(progress, loss1, loss2):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}%:  loss = {2:.3f}, acc = {3:.2f}%".format( "#"*block + "-"*(barLength-block), round(progress*100, 3), loss1, loss2*100)
    sys.stdout.write(text)
    sys.stdout.flush()


class actRecogFC_classifier(nn.Module):
    def __init__(self):
        # input 1000*16
        super(actRecogFC_classifier, self).__init__()
        self.num_z1 = 4096
        self.num_z2 = 4096
        self.num_z3 = 21
        self.num_frame = 16
        #self.drop_rate = 0.2
        #self.avgpool1d_1 = nn.AvgPool1d(self.num_frame)
        #self.conv1d_1 = nn.Conv1d(2048, 2048, 3, padding=1)
        #self.maxpool1d_1 = nn.MaxPool1d(self.num_frame)
        self.avgpool1d_2 = nn.AvgPool1d(self.num_frame)
        self.fc1 = nn.Sequential(nn.Linear(2048, self.num_z1), nn.ReLU())
        #self.drop1 = nn.Dropout(p=self.drop_rate)
        self.fc2 = nn.Sequential(nn.Linear(self.num_z1, self.num_z2), nn.ReLU())
        #self.drop2 = nn.Dropout(p=self.drop_rate)
        self.fc3 = nn.Sequential(nn.Linear(self.num_z2, self.num_z3))
        
        
    def forward(self, x):
        #out = self.avgpool1d(x)
        #print('AvgPool shape:', out.shape)
        #out_pool = self.conv1d_1(x)
        #out_pool = self.avgpool1d_1(out_pool)
        out = self.avgpool1d_2(x)
        #out = torch.cat((out_avg, out_pool), 1)
        out = self.fc1(out.view(out.size(0), -1))
        #out = self.drop1(out)
        out = F.dropout(out, training=self.training, p=0.2)
        out = self.fc2(out)
        #out = self.drop2(out)
        out = F.dropout(out, training=self.training, p=0.2)
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)
    
def testFCNN(cnn_model, fc_model, train_loader, batch_size, guid):

    epoch_acc  = 0.
    num_examples = 0

    preds = []

    for batch_i, sample in enumerate(train_loader):
        x = Variable(sample['X'])
        label = Variable(sample['Y'])
        if guid >= 0:   # move to GPU
            x = x.cuda(guid)
            label = label.cuda(guid)
        mini_batch_size = x.shape[0]

        # x: (nL, 16L, 3L, 299L, 299L)
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        # x: (16L, nL, 3L, 299L, 299L)
        split_x = torch.split(x, 1, dim=0)
        x00 = cnn_model.forward( split_x[0].view(mini_batch_size, 3, 299, 299))
        x01 = cnn_model.forward( split_x[1].view(mini_batch_size, 3, 299, 299))
        x02 = cnn_model.forward( split_x[2].view(mini_batch_size, 3, 299, 299))
        x03 = cnn_model.forward( split_x[3].view(mini_batch_size, 3, 299, 299))
        x04 = cnn_model.forward( split_x[4].view(mini_batch_size, 3, 299, 299))
        x05 = cnn_model.forward( split_x[5].view(mini_batch_size, 3, 299, 299))
        x06 = cnn_model.forward( split_x[6].view(mini_batch_size, 3, 299, 299))
        x07 = cnn_model.forward( split_x[7].view(mini_batch_size, 3, 299, 299))
        x08 = cnn_model.forward( split_x[8].view(mini_batch_size, 3, 299, 299))
        x09 = cnn_model.forward( split_x[9].view(mini_batch_size, 3, 299, 299))
        x10 = cnn_model.forward(split_x[10].view(mini_batch_size, 3, 299, 299))
        x11 = cnn_model.forward(split_x[11].view(mini_batch_size, 3, 299, 299))
        x12 = cnn_model.forward(split_x[12].view(mini_batch_size, 3, 299, 299))
        x13 = cnn_model.forward(split_x[13].view(mini_batch_size, 3, 299, 299))
        x14 = cnn_model.forward(split_x[14].view(mini_batch_size, 3, 299, 299))
        x15 = cnn_model.forward(split_x[15].view(mini_batch_size, 3, 299, 299))
        
        #print(type(x00))
        
        merge_result = torch.stack((x00,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11, x12, x13, x14, x15), 2)
        #print('Merge result shape:', merge_result.shape)
        
        
        y = fc_model.forward(merge_result)
        #print('Predcited label shape:', y.shape)
        #print('Label shape:', label.shape)
        label = label.squeeze_()
        #print('Label shape:', label.shape)
        
        _, pred = torch.max(y.data, 1)

        # statistics
        epoch_acc  = epoch_acc  + torch.sum(pred == label.data)
        num_examples = num_examples + label.shape[0]
        
        update_progress(batch_i/len(train_loader), 0, (epoch_acc/num_examples))
        preds.extend(pred)
      
    preds = torch.Tensor(preds)
    return preds


if __name__ == '__main__':
    batch_size = 2
    epochs = 60
    guid = 1
    
    vd_dir =  sys.argv[1]
    csv_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    print('Video dir:', vd_dir)
    print('CSV dir:', csv_dir)
    print('Output dir:', output_dir)
    
    
    cnn_model = torch.load("cnn_epoch_{}.pt".format(40))
    #inception_v3(pretrained=True)
    fc_model  = torch.load("fc_epoch_{}.pt".format(40))
    #actRecogFC_classifier()
    

    cnn_model = cnn_model.cuda(guid)
    fc_model  =  fc_model.cuda(guid)
    

    # create dataset
    dataset = Vd2ImgDataset(vd_dir, max_frame=16, num_classes=11, file_list=csv_dir)
    
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    cnn_model.eval()
    fc_model.eval()
    
    plot_step_loss = []
    plot_epoch_loss = []
    plot_epoch_acc = []

    preds = testFCNN(cnn_model, fc_model, dataloader, batch_size, guid)
 
    writePreds(os.path.join(output_dir, 'p1_valid.txt'), preds)