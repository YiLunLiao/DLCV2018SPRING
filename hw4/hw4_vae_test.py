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
import sys
#from tensorboardX import SummaryWriter 
import matplotlib.pyplot as plt


# compute MSE on testing data

def update_progress(progress, loss1):
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
    text = "\rPercent: [{0}] {1:.2f}% MSE loss = {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 3), loss1)
    sys.stdout.write(text)
    sys.stdout.flush()

class FaceDataset(Dataset):
    def __init__(self, dir, len_data):
        self.dir = dir
        self.len_data = len_data
    def __len__(self):
        return self.len_data
    def __getitem__(self, idx):
        idx = idx + 40000
        img_idx = str(idx).zfill(5)
        img_name = os.path.join(self.dir, img_idx + '.png')
        img = io.imread(img_name)
        img = img.astype(np.float32)
        img = img/255
        img_np = img
        
        # numpy: (H, W, C) -> torch: (C, H, W)
        img = img.transpose((2, 0, 1))
        img_tor = torch.from_numpy(img)
        return {'np': img_np, 'tor': img_tor}

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.guid = 0 # operate on GPU

        #self.batch_size = 32    # for sampling size
        self.num_z = 128
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.maxpooling1 = nn.MaxPool2d(2, padding=0)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 5, 1, 2), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), nn.ReLU())
        self.maxpooling2 = nn.MaxPool2d(2, padding=0)
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 5, 1, 2), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU())
        self.maxpooling3 = nn.MaxPool2d(2, padding=0)

        self.fc_encode1 = nn.Linear(8*8*512, self.num_z)
        self.fc_encode2 = nn.Linear(8*8*512, self.num_z)
        
        
        self.fc_decode = nn.Linear(self.num_z, 8*8*512)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(512,   3, 4, 2, 1), nn.Sigmoid())

    def encoder(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpooling1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpooling2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.maxpooling3(out)
        #print(out)
        return self.fc_encode1(out.view(out.size(0), -1)), self.fc_encode2(out.view(out.size(0), -1))

    def decoder(self, x):
        out = self.fc_decode(x)
        out = self.deconv1(out.view(x.size(0), 512, 8, 8))
        out = self.deconv2(out)
        out = self.deconv3(out)
        return out
    
    def reparameter(self, mu, var):
        #print('Reparameter:', mu.shape)
        if self.training:
            e = Variable( torch.from_numpy(
                    np.random.normal(0, 1, (mu.shape[0], mu.shape[1]))).float())
            e = e.cuda(self.guid)
            z = mu + var*e
        else:
            z = mu
        return z

    def forward(self, x):
        mu, var = self.encoder(x)
        code = self.reparameter(mu, var)
        out = self.decoder(code)
        return out, mu, var


# test VAE for MSE error
def testVAE(net, dataloader, batch_size, guid):

    mse_acc_loss = 0.

    for batch_i, x in enumerate(dataloader):
    # datapoints (resize + converting into pytorch data array)
        x = Variable(x['tor'])
        if guid >= 0:   # move to GPU
            x = x.cuda(guid)
        x_bar, mu, var = net(x)

        mse = F.mse_loss(x_bar, x, size_average=True)
        mse_acc_loss = mse_acc_loss + mse.data.tolist()[0]

        update_progress(batch_i/len(dataloader), round(mse.data.tolist()[0], 4))
    print(" ")
    print("finished: MSE error = {}\n".format(mse_acc_loss/batch_i))
    return x_bar, x
    
        
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='VAE')    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument("-z", "--n_z", type=int, default=128, help="n_z")
    parser.add_argument("-f", "--n_f", type=int, default=64*64*3, help="n_f")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="epochs")
    parser.add_argument("-g", "--guid", type=int, default=0, help="gpu id")
    parser.add_argument('--out_dir', type=str, default='./', metavar='S',
                    help='output dir')
    parser.add_argument('--input_dir', type=str, default='hw4_data/', metavar='S',
                    help='input dir')
    args = parser.parse_args()
    out_dir = args.out_dir
    input_dir = args.input_dir

    torch.manual_seed(args.seed)
  
    net = torch.load('vae_20.pt')

    if args.guid >= 0:
        net = net.cuda(args.guid)
    guid = args.guid
        
    # create dataset
    dataset = FaceDataset(dir=os.path.join(input_dir, 'test'), len_data=2621)
    
    # create dataloader
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net.eval()
    x_bar, x = testVAE(net, dataloader, batch_size, guid)
    x_bar = x_bar.view(-1, 3, 64, 64)
    x_bar = x_bar.data.cpu().numpy()
    x_bar = x_bar.transpose([0, 2, 3, 1])
    x = x.view(-1, 3, 64, 64)
    x = x.data.cpu().numpy()
    x = x.transpose([0, 2, 3, 1])
    
    fig = plt.figure(figsize=(25, 5))
    #plt.title('Female')
    for i in range(10):
        fig.add_subplot(2, 10, i+1)
        plt.imshow(x[i, :, :, :])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    for i in range(10):
        fig.add_subplot(2, 10, 10+i+1)
        plt.imshow(x_bar[i, :, :, :])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.figtext(0.5, 0.95, 'Input',          fontsize=20)
    plt.figtext(0.5, 0.5,  'Reconstucted',   fontsize=20)
    #plt.axis([0, 1, 0, 1])
    fig.savefig(os.path.join(out_dir, 'fig1_3.jpg'))
    

