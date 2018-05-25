from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import pandas as pd 
from skimage import io
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# https://github.com/pytorch/examples/blob/master/dcgan/main.py

class FaceDataset(Dataset):
    def __init__(self, dir, len_data, csv_filename='hw4_data/train.csv'):
        self.dir = dir
        
        self.csv_filename = csv_filename
        self.csv_data = pd.read_csv(self.csv_filename, sep=',', encoding='ISO-8859-1', 
                        usecols=['image_name', 'Bangs', 'Big_Lips', 'Black_Hair',
                                 'Blond_Hair', 'Brown_Hair', 'Heavy_Makeup',
                                 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                                 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                                 'Wearing_Lipstick'])
        self.len_data = len(self.csv_data)
        
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        img_idx = str(idx).zfill(5)
        img_name = os.path.join(self.dir, img_idx + '.png')
        img = io.imread(img_name)
        
        # numpy: (H, W, C) -> torch: (C, H, W)
        img = img.astype(np.float32)
        img = img/255
        img_np = img
        img = img.transpose((2, 0, 1))
        img_tor = torch.from_numpy(img)
        
        # label
        label = self.csv_data.iloc[idx, :].values
        label = label[1::]
        label = np.float32(label)
        label = torch.from_numpy(label)
        
        return {'np': img_np, 'tor': img_tor, 'label': label}

def update_progress(progress, Loss_D, Loss_G):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress logvar must be float\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}%  Loss_D = {2:.4f}, Loss_G = {3:.4f}".format( "#"*block + "-"*(barLength-block), round(progress*100, 3), Loss_D, Loss_G)
    sys.stdout.write(text)
    sys.stdout.flush()


class generator(nn.Module): 
    def __init__(self, nz=128, ngf=64, nc=3, num_class=13):
        super(generator, self).__init__()
        self.nz  = nz
        self.num_class = num_class
        self.ngf = ngf
        self.nc  = nc
        self.fc = nn.Sequential(
                nn.Linear(self.nz + self.num_class, self.nz*2),
                nn.BatchNorm1d(self.nz*2),
                nn.ReLU(True),
                nn.Linear(self.nz*2, self.nz*2),
                nn.BatchNorm1d(self.nz*2),
                nn.ReLU(True)
                )
        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz*2, self.ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(  self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    self.ngf,  self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, self.nz*2, 1, 1)
        out = self.deconv(out)
        return out

class discriminator(nn.Module): 
    def __init__(self, ndf=64, nc=3, num_class=13):
        super(discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.num_class = num_class
        self.conv_n = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(     self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(  self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf*8, self.ndf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.ndf*8, self.ndf*4),
            nn.BatchNorm1d(self.ndf*4),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(self.ndf*4, 1),
            nn.Sigmoid()
        )
        self.cl = nn.Sequential(
            nn.Linear(self.ndf*4, self.num_class),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.conv_n(x)
        output = output.view(-1, self.ndf*8)
        output = self.fc(output)
        dis = self.dc(output)
        dis = dis.view(-1, 1).squeeze(1)
        aux = self.cl(output)
        return dis, aux

def testACGAN(netG, guid, fixed_noise):

    gen_vec = fixed_noise
    gen_vec[:, 128+7] = 0
    gen_vec = gen_vec.cuda(guid)
    #gen_vec = Variable(gen_vec).cuda(guid)
    img_0 = netG(gen_vec)
    img_0 = img_0.view(-1, 3, 64, 64)
    img_0 = img_0.data.cpu().numpy()
    img_0 = img_0.transpose([0, 2, 3, 1])
    
    gen_vec[:, 128+7] = 1
    #gen_vec = Variable(gen_vec).cuda(guid)
    img_1 = netG(gen_vec)
    img_1 = img_1.view(-1, 3, 64, 64)
    img_1 = img_1.data.cpu().numpy()
    img_1 = img_1.transpose([0, 2, 3, 1])

    return img_0, img_1, gen_vec

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TrainGAN')    
    parser.add_argument('--seed', type=int, default=8, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--gpuid', type=int, default=0, metavar='G',
                    help='gpu id (default: 0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--out_dir', type=str, default='./', metavar='S',
                    help='output dir')
    
    args = parser.parse_args()
    out_dir = args.out_dir
    
    torch.manual_seed(args.seed)
  
    guid = args.gpuid
    nz = 128
        
    netG = torch.load('gen_26.pt')
    
    if guid >= 0:
        netG = netG.cuda(guid)
        
    netG.eval()
    '''
    noise1 = torch.load('generate_vector_0.pt')
    noise2 = torch.load('generate_vector_1.pt')
    fixed_noise = torch.cat((noise1, noise2), dim=0)
    img_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
    fixed_noise = fixed_noise[img_list]
    '''

    fixed_noise = torch.load('generate_vector_ACGAN.pt')

    img_0, img_1, _ = testACGAN(netG, guid, fixed_noise)
    
    img_0[np.where(img_0[:] < 0)] = 0
    img_0[np.where(img_0[:] > 1)] = 1
    img_1[np.where(img_1[:] < 0)] = 0
    img_1[np.where(img_1[:] > 1)] = 1

    fig = plt.figure(figsize=(25, 5))
    #plt.title('Female')
    for i in range(10):
        fig.add_subplot(2, 10, i+1)
        plt.imshow(img_0[i, :, :, :])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    #fig.savefig('img_0_.png')
    
    #fig = plt.figure(figsize=(25, 4))
    #plt.title('Male')
    for i in range(10):
        fig.add_subplot(2, 10, 10+i+1)
        plt.imshow(img_1[i, :, :, :])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.figtext(0.5, 0.95, 'Female', fontsize=20)
    plt.figtext(0.5, 0.5,  'Male',   fontsize=20)
    #plt.axis([0, 1, 0, 1])
    fig.savefig(os.path.join(out_dir, 'fig3_3.png'))
    

    #torch.save(fixed_noise, 'generate_vector_all.pt')
    
    
        
    