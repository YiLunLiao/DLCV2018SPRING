import pandas as pd
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import os

if __name__ == '__main__':
    csv_files = ['VAE_train_log.csv', 'GAN_train_log.csv', 'ACGAN_train_log.csv']
    loss_types = [('KL_', 'MSE_'), ('D_', 'G_'), ('D_', 'G_')]    
    #csv_data = pd.read_csv(csv_file, sep=',', encoding='ISO-8859-1')
    
    parser = argparse.ArgumentParser(description='plotLog')    
    parser.add_argument('--out_dir', type=str, default='./', metavar='S',
                    help='output dir')
    args = parser.parse_args()
    out_dir = args.out_dir
    
    for idx in range(3):
        csv_file = csv_files[idx]
        loss_type = loss_types[idx]
        csv_data = pd.read_csv(csv_file, sep=',', encoding='ISO-8859-1')
        fig = plt.figure(figsize=(24, 12))
        for i in range(2):
            fig.add_subplot(1, 2, i+1)
          
            data = csv_data.iloc[:, i].values
            plt.plot(data)
            loss = loss_type[i]
            plt.ylabel(loss + ' Loss', fontsize=32)
            plt.xlabel('steps (scaled by 0.01)', fontsize=32)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
                   
            plt.grid(True)
       # plt.show()
        fig.savefig(os.path.join(out_dir, 'fig' + str(idx+1) + '_2.jpg'))
        
