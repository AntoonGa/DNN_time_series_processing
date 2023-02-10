"""
Created on Thu Oct 22 15:51:08 2020

@author: grab
Compare unity output performance vs beta hyperparameter.
Using only histories.
"""
# -*- coding: utf-8 -*-

import os
path = r"D:\Dropbox\EPFL\Machine Learning\2021_01_paperRework\experiments\betaVary"
os.chdir(path)
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm
import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense, Softmax,  Flatten, Dropout, Reshape, Lambda, Concatenate
from keras.constraints import max_norm
from keras import metrics
import matplotlib.pyplot as plt
import math
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape

from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


# Load and concatenate train/val/test data.
def unityPrepData( dataFolder, trainDataFile):
    
    data = np.load(dataFolder + trainDataFile)
    inputs_unity   = data['inputs']
    targets        = data['targets']
    targets_latent = data['metadata']
    del data.f
    data.close()
    
    target_unity = np.concatenate(  ( np.squeeze(targets), targets_latent ), axis=1)
    
    return inputs_unity, target_unity


#%% load and plot target histories.
# histpath = [r"D:\Dropbox\EPFL\Machine Learning\2021_01_paperRework\trainModels\1610562995_TrainedModels_Latent7/",
#             r"D:\Dropbox\EPFL\Machine Learning\2021_01_paperRework\experiments\betaVary\1610562995_TrainedModels_Latent7/",
#             r"D:\Dropbox\EPFL\Machine Learning\2021_01_paperRework\trainModels\1610562995_TrainedModels_Latent5/",
#             r"D:\Dropbox\EPFL\Machine Learning\2021_01_paperRework\trainModels\1610562995_TrainedModels_Latent5/"]


# histlist = ["-train_Beta0.001_FM_10x20x1_8bs-_hist_.npz",
#             "-train_Beta0.001_FM_varyBeta_10x12x1_8bs-_hist_.npz",
#             "-train_Beta0.001_AMFM_10x20x1_8bs-_hist_.npz",
#             "-train_Beta0.001_AMFM_10x20x1_8bs--train_Beta0.001_AMFM_10x20x1_8bs-_hist_.npz"]
histpath = [r"D:\Dropbox\EPFL\Machine Learning\2021_01_paperRework\trainModels\1610562995_TrainedModels_Latent5/"]


histlist = ["-train_Beta0.001_AMFM_10x20x1_8bs-_hist_.npz"]

nbModels = len(histpath)
#%% plot histories together
fs = 9       
plt.figure()
xlims = (0,55)
for jj in range(nbModels):

    histFullPath = histpath[jj]  + histlist[jj]
    hist = np.load(histFullPath,allow_pickle=True)
    
    # History
    loss          =  hist['loss']        
    val_loss      =  hist['val_loss']    
    mseSignal     =  hist['mseSignal']    
    val_mseSignal =  hist['val_mseSignal'] 
    mseLatent     =  hist['mseLatent']      
   
    plt.plot(mseSignal, label=histlist[jj], linewidth = 3)
    plt.legend( fontsize = fs)
    plt.yscale('log')
    plt.xlim(xlims)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('$MSE_{dec}$ [-]', fontsize=15)
plt.tight_layout()
plt.show()
    

plt.figure()
for jj in range(nbModels):

    histFullPath = histpath[jj]  + histlist[jj]
    hist = np.load(histFullPath)
    
    
    # History
    loss          =  hist['loss']        
    val_loss      =  hist['val_loss']    
    mseSignal     =  hist['mseSignal']    
    val_mseSignal =  hist['val_mseSignal'] 
    mseLatent     =  hist['mseLatent']      
   
    plt.plot(mseLatent, label=histlist[jj], linewidth = 3)
    plt.legend( fontsize = fs)
    plt.yscale('log')
    plt.xlim(xlims)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('$MSE_{reg}$ [-]', fontsize=15)
plt.tight_layout()
plt.show()


