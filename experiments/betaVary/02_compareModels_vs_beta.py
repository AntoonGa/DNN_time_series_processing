# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:51:08 2020

@author: grab
Compare unity output performance vs beta hyperparameter.
Using only histories.
"""
# -*- coding: utf-8 -*-

import os

path = r"./"
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
from keras.layers import Input, Dense, Softmax, Flatten, Dropout, Reshape, Lambda, Concatenate
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
def unityPrepData(dataFolder, trainDataFile):

    data = np.load(dataFolder + trainDataFile)
    inputs_unity = data["inputs"]
    targets = data["targets"]
    targets_latent = data["metadata"]
    del data.f
    data.close()

    target_unity = np.concatenate((np.squeeze(targets), targets_latent), axis=1)

    return inputs_unity, target_unity


#%% colorblind-friendly qualitative color palette
colors = [
    "#88CE33",
    "#CC6677",
    "#DDCC77",
    "#117733",
    "#332288",
    "#AA4499",
    "#999933",
    "#882255",
    "#661100",
    "#6688CC",
]

# sorting colors intensity
colors.sort()


#%% models histories list
modelPaths = [r"./1610562995_TrainedModels_Latent7/"]

timeTags = ["1610562995"]

trainedPaths = ["_TrainedModels/"]

modelNames = ["Unity_Latent7_"]

trainTags = [
    "-train_Beta0_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta0.0001_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta0.0005_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta0.001_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta0.005_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta0.01_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta0.05_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta0.1_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta0.5_FM_varyBeta_10x12x1_8bs-",
    "-train_Beta1_FM_varyBeta_10x12x1_8bs-",
]


histExtension = ["_hist_.npz"]

modelExtension = [".keras"]

betas = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

nbModels = len(trainTags)

#%% plot histories
for jj in range(nbModels):
    modelPath = modelPaths[0]
    trainedPath = trainedPaths[0]
    modelName = modelNames[0]
    timeTag = timeTags[0]
    trainTag = trainTags[jj]
    beta = str(betas[jj])

    histPath = modelPath
    histName = trainTag + histExtension[0]
    histFullPath = histPath + histName
    hist = np.load(histFullPath)

    # History
    loss = hist["loss"]
    val_loss = hist["val_loss"]
    mseSignal = hist["mseSignal"]
    val_mseSignal = hist["val_mseSignal"]
    mseLatent = hist["mseLatent"]
    MSE_test = hist["MSE_test"]
    MSElatent_test = hist["mseLatent_test"]
    MSEsignal_test = hist["mseSignal_test"]

    fs = 8
    plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(loss, "--", label="beta_loss")
    plt.plot(val_loss, label="beta_val_loss")
    plt.legend(fontsize=fs)
    plt.xticks([])
    plt.yscale("log")
    plt.subplot(312)
    plt.plot(mseSignal, label="mseSignal \n" + "test_data: " + "{:.2e}".format(MSEsignal_test))
    # plt.plot(val_mseSignal  , label='val_mseSignal')
    plt.legend(fontsize=fs)
    plt.yscale("log")
    plt.xticks([])
    plt.subplot(313)
    plt.plot(mseLatent, label="mseLatent \n" + "test_data: " + "{:.2e}".format(MSElatent_test))
    # plt.plot(val_mseLatent  , label='val_mseLatent')
    plt.yscale("log")
    plt.legend(fontsize=fs)
    plt.suptitle("model beta : " + beta)
    plt.tight_layout()
plt.show()

#%% plot histories together
fs = 9
plt.figure()
for jj in range(nbModels):
    modelPath = modelPaths[0]
    trainedPath = trainedPaths[0]
    modelName = modelNames[0]
    timeTag = timeTags[0]
    trainTag = trainTags[jj]
    beta = str(betas[jj])

    histPath = modelPath
    histName = trainTag + histExtension[0]
    histFullPath = histPath + histName
    hist = np.load(histFullPath)

    # History
    loss = hist["loss"]
    val_loss = hist["val_loss"]
    mseSignal = hist["mseSignal"]
    val_mseSignal = hist["val_mseSignal"]
    mseLatent = hist["mseLatent"]

    plt.plot(mseSignal, label=beta, linewidth=3, color=colors[jj])
    plt.legend(fontsize=fs)
    plt.yscale("log")
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("$MSE_{dec}$ [-]", fontsize=15)
plt.tight_layout()
plt.show()


plt.figure()
for jj in range(nbModels):
    modelPath = modelPaths[0]
    trainedPath = trainedPaths[0]
    modelName = modelNames[0]
    timeTag = timeTags[0]
    trainTag = trainTags[jj]
    beta = str(betas[jj])

    histPath = modelPath
    histName = trainTag + histExtension[0]
    histFullPath = histPath + histName
    hist = np.load(histFullPath)

    # History
    loss = hist["loss"]
    val_loss = hist["val_loss"]
    mseSignal = hist["mseSignal"]
    val_mseSignal = hist["val_mseSignal"]
    mseLatent = hist["mseLatent"]

    plt.plot(mseLatent, label=beta, linewidth=3, color=colors[jj])
    plt.legend(fontsize=fs)
    plt.yscale("log")
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("$MSE_{reg}$ [-]", fontsize=15)
plt.tight_layout()
plt.show()
#%% plot MSE in a tab


MSE = []
MSElatent = []
MSEsignal = []
colsNames = []
# colors    = []
for jj in range(nbModels):
    modelPath = modelPaths[0]
    trainedPath = trainedPaths[0]
    modelName = modelNames[0]
    timeTag = timeTags[0]
    trainTag = trainTags[jj]
    histPath = modelPath
    histName = trainTag + histExtension[0]
    histFullPath = histPath + histName
    hist = np.load(histFullPath)

    MSE.append(hist["MSE_test"])
    MSElatent.append(hist["mseLatent_test"])
    MSEsignal.append(hist["mseSignal_test"])
    colsNames.append(str(betas[jj]))
    # colors.append((random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)))

# ax1 = plt.subplot(311)
# plt.bar(colsNames, MSE, color = colors)
# plt.xticks([])
# plt.ylabel("MSE")
# plt.yscale('log')
ax1 = plt.subplot(311)
plt.bar(colsNames, MSE, color=colors)
plt.xticks([])
plt.ylabel(r"$\beta$" + "$MSE$ [-]", fontsize=15)
plt.yscale("log")
ax1 = plt.subplot(312)
plt.bar(colsNames, MSEsignal, color=colors)
plt.xticks([])
plt.ylabel("$MSE_{dec}$ [-]", fontsize=15)
plt.yscale("log")
ax1 = plt.subplot(313)
plt.bar(colsNames, MSElatent, color=colors)
plt.ylabel("$MSE_{reg}$ [-]", fontsize=15)
plt.xlabel(r"$\beta$", fontsize=15)
plt.yscale("log")
plt.tight_layout()
plt.show()
