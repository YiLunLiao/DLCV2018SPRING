wget https://www.dropbox.com/s/yxjvjoifktvr4wp/cnn_epoch_40.pt?dl=1 -O cnn_epoch_40.pt
wget https://www.dropbox.com/s/vq40f5hr99kydb7/fc_epoch_40.pt?dl=1 -O fc_epoch_40.pt
python3 hw5_CNN_test.py $1 $2 $3
