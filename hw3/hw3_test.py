import sys
import numpy as np
import csv
import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Dropout, Add
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Model
import os
from PIL import Image

#2313 training


def readValidDataLabel(dataPath):
    file_list2 = [file2 for file2 in os.listdir(dataPath) if file2.endswith('.jpg')]
    file_list2.sort()
    n_masks = len(file_list2)
    sat_image = []

    for i, file2 in enumerate(file_list2):
        sat_im_tmp = Image.open(os.path.join(dataPath,file2))
        sat_image.append(np.array(sat_im_tmp))
        sat_im_tmp.close()
    
    sat_image_arr = np.array(sat_image)
    print(sat_image_arr.shape)
    return sat_image_arr, file_list2



def parseMask(msk_image):
    msk_image_category = np.zeros((msk_image.shape[0], msk_image.shape[1], msk_image.shape[2], 7), dtype='uint8')
    for n in range(msk_image.shape[0]):
        mask0 = np.all(msk_image[n,:,:,:] == (0, 255, 255), axis=-1)
        #print('mask shape', mask.shape)
        msk_image_category[n, mask0, 0] = 1
        mask1 = np.all(msk_image[n,:,:,:] == (255, 255, 0), axis=-1)
        #print('mask shape', mask.shape)
        msk_image_category[n, mask1, 1] = 1
        mask2 = np.all(msk_image[n,:,:,:] == (255, 0, 255), axis=-1)
        #print('mask shape', mask.shape)
        msk_image_category[n, mask2, 2] = 1
        mask3 = np.all(msk_image[n,:,:,:] == (0, 255, 0), axis=-1)
        #print('mask shape', mask.shape)
        msk_image_category[n, mask3, 3] = 1
        mask4 = np.all(msk_image[n,:,:,:] == (0, 0, 255), axis=-1)
        #print('mask shape', mask.shape)
        msk_image_category[n, mask4, 4] = 1
        mask5 = np.all(msk_image[n,:,:,:] == (255, 255, 255), axis=-1)
        #print('mask shape', mask.shape)
        msk_image_category[n, mask5, 5] = 1
        mask6 = np.all(msk_image[n,:,:,:] == (0, 0, 0), axis=-1)
        #print('mask shape', mask.shape)
        msk_image_category[n, mask6, 6] = 1

    return msk_image_category

def writePredict(predictMsk, dataPath, file_list):
    msk_image = np.zeros((predictMsk.shape[0],predictMsk.shape[1],predictMsk.shape[2],3))
    for n in range(predictMsk.shape[0]):
        mask0 = (np.argmax(predictMsk[n,:,:,:], axis = 2) == 0)
        #print('mask shape', mask.shape)
        msk_image[n, mask0, :] = (0, 255, 255)
        mask1 = (np.argmax(predictMsk[n,:,:,:], axis = 2) == 1)
        msk_image[n, mask1, :] = (255, 255, 0)
        mask2 = (np.argmax(predictMsk[n,:,:,:], axis = 2) == 2)
        msk_image[n, mask2, :] = (255, 0, 255)
        mask3 = (np.argmax(predictMsk[n,:,:,:], axis = 2) == 3)
        msk_image[n, mask3, :] = (0, 255, 0)
        mask4 = (np.argmax(predictMsk[n,:,:,:], axis = 2) == 4)
        msk_image[n, mask4, :] = (0, 0, 255)
        mask5 = (np.argmax(predictMsk[n,:,:,:], axis = 2) == 5)
        msk_image[n, mask5, :] = (255, 255, 255)
        mask6 = (np.argmax(predictMsk[n,:,:,:], axis = 2) == 6)
        msk_image[n, mask6, :] = (0, 0, 0)
    for n, file in enumerate(file_list):
        file_num, b = file.split('_')
        file_num = file_num+'_mask.png'
        img_tmp = Image.fromarray(np.uint8(msk_image[n,:,:,:]))
        img_tmp.save(os.path.join(dataPath, file_num))
        img_tmp.close()
    return msk_image

    
#######################

#dataPath_test = 'hw3-train-validation/validation'
#dataPath_write_test = 'predictMsk_valid_liao'
dataPath_test = sys.argv[2]
dataPath_write_test  = sys.argv[3]

x_valid, msk_file_list = readValidDataLabel(dataPath_test)

print('x_valid shape ',x_valid.shape)

########################################### set parameters here
batch_size = 4
num_classes = 7#0-6
epochs = 20
lr = 0.0001
decay=1e-5
############################################
input_shape = (512, 512, 3)

x_valid = x_valid.astype('float32')
x_valid /= 255



model_name = sys.argv[1] # model name
model = load_model(model_name)

print("model loaded")


predictions = model.predict(x_valid, batch_size=4)

print('finish prediction')

print("convert to RGB images")
writePredict(predictions, dataPath_write_test, msk_file_list)

print("done")



