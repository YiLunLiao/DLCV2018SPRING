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
from sklearn.manifold import TSNE


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
    def __init__(self, dir, len_data=0, csv_filename='hw4_data/test.csv'):
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
        img_idx = idx + 40000
        img_idx = str(img_idx).zfill(5)
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
    
    def generate(self, z):
        out = self.decoder(z)
        return out
    
    def encode(self, x):
        mu, std = self.encoder(x)
        out = torch.cat((mu, std), dim=1)
        return out


# encode images
def encodeVAE(net, batch_size, guid, dataloader, num_feautre):
    
    for batch_i, x in enumerate(dataloader):
        img = Variable(x['tor'])
        label = Variable(x['label'])
        label = label[:, 7]
        if guid >= 0:   # move to GPU
            img = img.cuda(guid)
            label = label.cuda(guid)
        mu_feature = net.encode(img)
        
        label = label.data.cpu().numpy()
        mu_feature = mu_feature.data.cpu().numpy()
        img = 0
        
        if batch_i == 0:
            label_stk = label
            mu_stk = mu_feature
        else: 
            label_stk = np.concatenate((label_stk, label), 0)
            mu_stk    = np.concatenate((mu_stk, mu_feature), 0)
        if (batch_i + 1)*batch_size >= num_feautre:
            break
        print('Batch', batch_i)

    return mu_stk, label_stk

    
    
        
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='VAE')    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument("-z", "--n_z", type=int, default=128, help="n_z")
    parser.add_argument("-f", "--n_f", type=int, default=64*64*3, help="n_f")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch_size")
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
    batch_size = args.batch_size
    
    
    dataset = FaceDataset(dir=os.path.join(input_dir, 'test'), csv_filename=os.path.join(input_dir, 'test.csv'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net.eval()
    num_feature = 2048
    
    mu_stk, label_stk = encodeVAE(net, batch_size, guid, dataloader, num_feature)
    print('mu stack shape:', mu_stk.shape)
    
    vis_data = TSNE(n_components=2, init='pca', verbose=1, random_state=0, n_iter=1000).fit_transform(mu_stk)
    vis_data_x = vis_data[:,0]
    vis_data_y = vis_data[:,1]
    
    fig = plt.figure(figsize=(10, 7))
    fig.text(0.45, 0.95, "Femal", ha="center", va="bottom", fontsize="large",color="red")
    fig.text(0.5, 0.95, "&", ha="center", va="bottom", fontsize="large")
    fig.text(0.55,0.95,"Male", ha="center", va="bottom", fontsize="large", color="blue")
    cm = plt.cm.get_cmap('Spectral')
    sc = plt.scatter(vis_data_x, vis_data_y, c= label_stk, cmap = cm)
    plt.colorbar(sc)
    #plt.figtext(0, 30, 'Female', fontsize=20)
    #plt.figtext(0, -30, 'Male', fontsize=20)
    #plt.show()

    fig.savefig(os.path.join(out_dir, 'fig1_5.jpg'))
    

