from inception_model_noBN import *
from vd_dataloader import  *
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
from Vd2Img_dataset import *
import numpy as np

def update_progress(progress, loss1, loss2):
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

'''
class actRecog_RNN_VarLen_classifier(nn.Module):
    def __init__(self):
        # input: batch_size, num_frame, 2048
        self.drop_rate = 0
        super(actRecog_RNN_VarLen_classifier, self).__init__()
        self.gru = nn.GRU(2048, 256, 1, batch_first=True, dropout=self.drop_rate, bidirectional=True)
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
'''

class actRecog_RNN_VarLen_classifier(nn.Module):
    def __init__(self):
        # input: batch_size, num_frame, 2048
        self.drop_rate = 0.2
        super(actRecog_RNN_VarLen_classifier, self).__init__()
        self.gru = nn.GRU(2048, 512, 1, batch_first=True, dropout=self.drop_rate, bidirectional=True)
        self.fc = nn.Linear(1024, 11)
        
    def forward(self, x, hidden=None):
        out, _ = self.gru(x, hidden)
        #print('LSTM output shape:', out.shape)
        out = out[:, -1, :]
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    
    def forward_gru(self, x, hidden=None):
        _, hidden_out = self.gru(x, hidden)
        return hidden_out
    
    
def trainCRNN_VarLen(cnn_model, rnn_model, opt_cnn, opt_rnn, train_loader, epoch, batch_size, guid):

    epoch_loss = 0.
    epoch_acc  = 0.
    num_examples = 0
    
    assert batch_size == 1
    frame_batch = 8 # simulataneous prediction of frame_batch frames

    for batch_i, sample in enumerate(train_loader):
        x = Variable(sample['X'])
        label = Variable(sample['Y'])
        if guid >= 0:   # move to GPU
            x = x.cuda(guid)
            label = label.cuda(guid)

        # (1, num_frame, 3, H, W) -> (num_frame, 3, H, W)
        x.squeeze_(0)
        #print('x shape:', x.shape)

        # split to different sub-batch
        split_x = torch.split(x, frame_batch, dim=0) # tuple
        feature_list = list()
        for i in range(len(split_x)):
            CNN_feature = cnn_model(split_x[i])
            #CNN_feature = CNN_feature.unsqueeze_(0)
            #if i == 0:
            #    hidden = rnn_model.forward_gru(CNN_feature)
            #elif i == len(split_x)-1:
            #    y = rnn_model.forward(CNN_feature, hidden)
            #else:
            #    hidden = rnn_model.forward_gru(CNN_feature, hidden)
            feature_list.append(CNN_feature)
            
        # (num_frame, 2048)
        temp_feature = torch.cat(feature_list, dim=0) 
        
        # (1, num_frame, 2048)
        temp_feature = temp_feature.unsqueeze_(0)

        #print('Temp feature shape:', temp_feature.shape)
        
        y = rnn_model.forward(temp_feature)
        #print('Predcited label shape:', y.shape)
        #print('Label shape:', label.shape)
        label = label.squeeze_()
        #print('Label shape:', label.shape)
        
        criterion = nn.NLLLoss()
        loss = criterion(y, label)
        
        
        # backward and optimize parameters
        #opt_cnn.zero_grad()
        opt_rnn.zero_grad()
        loss.backward()
        #opt_cnn.step()
        opt_rnn.step()
        
        torch.save(label.data.cpu(), 'label_{}.pt'.format(batch_i))
        torch.save(temp_feature.data.cpu(), 'data_{}.pt'.format(batch_i))
        
        feature_list = list()
        temp_feature = 0
        split_x = 0
        
        _, preds = torch.max(y.data, 1)

        # statistics
        epoch_loss = epoch_loss + loss.data[0]
        epoch_acc  = epoch_acc  + torch.sum(preds == label.data)
        num_examples = num_examples + label.shape[0]
        
        update_progress(batch_i/len(train_loader), round(loss.data.tolist()[0], 7), (epoch_acc/num_examples))
       
    
    print(" ")
    print("finished epoch {}, loss: {}, accuracy: {}\n".format(
                epoch,
                epoch_loss/(1+batch_i),
                epoch_acc/num_examples
                ))


def trainCRNN_VarLen_fromData(rnn_model, opt_rnn, dataset, epoch, batch_size, guid):

    epoch_loss = 0.
    epoch_acc  = 0.
    num_examples = 0
    
    order = list(range(len(dataset)))
    np.random.shuffle(order)
    
    assert batch_size == 1
    frame_batch = 8 # simulataneous prediction of frame_batch frames

    for batch_i in range(len(dataset)):
        temp_feature = torch.load('trim_torch_data/test/data_{}.pt'.format(order[batch_i]))
        label = torch.load('trim_torch_data/test/label_{}.pt'.format(order[batch_i]))
        if guid >= 0:   # move to GPU
            temp_feature = Variable(temp_feature).cuda(guid)
            label = Variable(label).cuda(guid)
            
        y = rnn_model.forward(temp_feature)

        criterion = nn.NLLLoss()
        loss = criterion(y, label)
        
        opt_rnn.zero_grad()
        loss.backward()
        opt_rnn.step()
        
        _, preds = torch.max(y.data, 1)

        # statistics
        epoch_loss = epoch_loss + loss.data[0]
        epoch_acc  = epoch_acc  + torch.sum(preds == label.data)
        num_examples = num_examples + label.shape[0]
        
        update_progress(batch_i/len(dataset), round(loss.data.tolist()[0], 7), (epoch_acc/num_examples))
       
    
    print(" ")
    print("finished epoch {}, loss: {}, accuracy: {}\n".format(
                epoch,
                epoch_loss/(1+batch_i),
                epoch_acc/num_examples
                ))
    
def convert2CNNFeature(cnn_model, train_loader, batch_size, guid):

    epoch_loss = 0.
    epoch_acc  = 0.
    num_examples = 0
    
    assert batch_size == 1
    frame_batch = 8 # simulataneous prediction of frame_batch frames

    for batch_i, sample in enumerate(train_loader):
        x = Variable(sample['X'])
        label = Variable(sample['Y'])
        if guid >= 0:   # move to GPU
            x = x.cuda(guid)
            label = label.cuda(guid)

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

        label = label.squeeze_()
        
        torch.save(label.data.cpu(), 'trim_torch_data/label_{}.pt'.format(batch_i))
        torch.save(temp_feature.cpu(), 'trim_torch_data/data_{}.pt'.format(batch_i))
        
        print('save trim_torch_data/label_{}.pt'.format(batch_i))
        
        update_progress(batch_i/len(train_loader), round(0), (0))
       


if __name__ == '__main__':
    save_dir = 'torch_model/baseline'
    save_interval = int(1)
    batch_size = 1
    epochs = 60
    guid = 0
    
    vd_dir =  'HW5_data_TA_0518/TrimmedVideos/video/test'
    csv_dir = 'HW5_data_TA_0518/TrimmedVideos/label/gt_test.csv'
    
    #cnn_model = inception_v3(pretrained=True)
    cnn_model = torch.load('crvr_cnn_9.pt')
    #rnn_model = actRecog_RNN_VarLen_classifier()
    rnn_model = torch.load("{}/crvr_rnn_v3_{}.pt".format(save_dir, 64))

    cnn_model = cnn_model.cuda(guid)
    rnn_model = rnn_model.cuda(guid)
    
    #opt_cnn = optim.Adam(cnn_model.parameters(), lr=1e-5)
    opt_cnn = 0
    opt_rnn = optim.Adam(rnn_model.parameters(),  lr=1e-4)

    # create dataset
    dataset = Vd2ImgDataset(vd_dir, max_frame=1000, num_classes=11, file_list=csv_dir)
    
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    cnn_model.eval()
    rnn_model.train()
    
    #convert2CNNFeature(cnn_model, dataloader, batch_size, guid)

    for epoch in range(1, epochs + 1):
        print('Epoch:', epoch,'/', epochs)
        #trainCRNN_VarLen(cnn_model, rnn_model, opt_cnn, opt_rnn, dataloader, epoch, batch_size, guid)
        trainCRNN_VarLen_fromData(rnn_model, opt_rnn, dataset, epoch, batch_size, guid)
        print('========================================================')
        
        # save parameters
        if epoch % save_interval == 0:
            #torch.save(
            #    cnn_model,
            #    "{}/crvr_cnn_v3_{}.pt".format(save_dir, epoch+60),
            #)
        
            torch.save(
                rnn_model,
                "{}/crvr_rnn_v4_{}.pt".format(save_dir, epoch),
            )

    # save parameters
    #torch.save(
    #  cnn_model,
    #  "{}/crvr_cnn_v3_{}.pt".format(save_dir, epoch+60),
    #)
    
    torch.save(
      rnn_model,
      "{}/crvr_rnn_v3_{}.pt".format(save_dir, epoch+60),
    )

    