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
import sys
import numpy as np

#from torchvision.utils import save_image

# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# https://github.com/pytorch/examples/blob/master/dcgan/main.py

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

class FaceDataset(Dataset):
    def __init__(self, dir, len_data):
        self.dir = dir
        self.len_data = len_data
    def __len__(self):
        return self.len_data
    def __getitem__(self, idx):
        img_idx = str(idx).zfill(5)
        img_name = self.dir + img_idx + '.png' #os.path.join(self.dir, img_idx + '.png')
        img = io.imread(img_name)
        img = img.astype(np.float32)
        img = img/255
        img_np = img
        # numpy: (H, W, C) -> torch: (C, H, W)
        img = img.transpose((2, 0, 1))
        img_tor = torch.from_numpy(img)
        return {'np': img_np, 'tor': img_tor}

class generator(nn.Module): 
    def __init__(self, nz=128, ngf=64, nc=3):
        super(generator, self).__init__()
        self.nz  = nz
        self.ngf = ngf
        self.nc  = nc
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(    self.nz, self.ngf*8, 4, 1, 0, bias=False),
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
        out = self.main(x)
        return out

class discriminator(nn.Module): #similar to decoder of VAE
    def __init__(self, ndf=64, nc=3):
        super(discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.main = nn.Sequential(
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
            nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


def genGAN(netG, num_img, guid):
    noise = Variable(torch.randn(batch_size, nz, 1, 1)).cuda(guid)
    gen_img = netG.forward(noise)
    return gen_img

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TrainGAN')    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
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
  
    batch_size = 32
    epochs = 30
    guid = args.gpuid
    nz = 128
    generator_steps = 1
    num_img = 32
    
    
    netG = torch.load("gen_27.pt")
    netG.eval()
    
    if guid >= 0:
        netG = netG.cuda(guid)
    gen_img = genGAN(netG, num_img, guid)
    
    
    gen_img = gen_img.data.cpu().numpy()
    gen_img = gen_img.transpose([0, 2, 3, 1])
    gen_img[np.where(gen_img[:] < 0)] = 0
    gen_img[np.where(gen_img[:] > 1)] = 1
        
    
    fig = plt.figure(figsize=(15, 8))

    for i in range(32):
        fig.add_subplot(4, 8, i+1)
        plt.imshow(gen_img[i, :, :, :])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    fig.savefig(os.path.join(out_dir, 'fig2_3.jpg'))
    