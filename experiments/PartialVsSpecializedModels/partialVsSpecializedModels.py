# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:51:08 2020

@author: grab
model evaluation tool


This script load the AM/FM and the AM trained DNNs.
We then compare the DNN performance on AM sets.

"""
# -*- coding: utf-8 -*-
# color Palette
targetcolor = "#661100"
fitcolor = "#6699CC"
fitcolor = "#332288"
dnncolor = "orange"
noisycolor = "#888888"


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

#%% Load and prepare AMFM DNN

# Custom loss function. Weighted MSE over signal vs latent. Beta = 1 means we focus on latent.
betaTag = 0.001
latentDim = 5


def customLoss(yTrue, yPred):
    latentSize = latentDim
    SignalSize = 512
    beta = betaTag
    mseSignal = K.square(yTrue[:, 0:SignalSize] - yPred[:, 0:SignalSize])
    mseSignal = K.abs(mseSignal)
    mseSignal = K.sum(mseSignal, axis=-1)
    mseSignal = mseSignal / SignalSize

    mseLatent = K.square(yTrue[:, SignalSize:] - yPred[:, SignalSize:])
    mseLatent = K.abs(mseLatent)
    mseLatent = K.sum(mseLatent, axis=-1)
    mseLatent = mseLatent / latentSize

    weighted_mse = (1 - beta) * mseSignal + beta * mseLatent
    return weighted_mse


# Custom metric: latent param loss only
def mseLatent(yTrue, yPred):
    latentSize = latentDim
    SignalSize = 512
    mseLatent = K.square(yTrue[:, SignalSize:] - yPred[:, SignalSize:])
    mseLatent = K.abs(mseLatent)
    mseLatent = K.sum(mseLatent, axis=-1)
    mseLatent = mseLatent / latentSize
    return mseLatent


# Custom metric: signal loss only
def mseSignal(yTrue, yPred):
    SignalSize = 512
    mseSignal = K.square(yTrue[:, 0:SignalSize] - yPred[:, 0:SignalSize])
    mseSignal = K.abs(mseSignal)
    mseSignal = K.sum(mseSignal, axis=-1)
    mseSignal = mseSignal / SignalSize
    return mseSignal


# Load and concatenate train/val/test data.
def unityPrepData(dataFolder, trainDataFile, latentDimension):

    data = np.load(dataFolder + trainDataFile, allow_pickle=True)
    inputs_unity = data["inputs"]
    targets = data["targets"]
    targets_latent = data["metadata"]
    tags = data["setTag"]

    del data.f
    data.close()

    # Delete useless latent variable for smaller regressors
    if latentDimension == 5:
        targets_latent = np.delete(targets_latent, [3, 4], axis=1)  # No mod regressor
    if latentDimension == 3:
        targets_latent = np.delete(
            targets_latent, [3, 4, 5, 6], axis=1
        )  # No mod,tau,noise regressor

    target_unity = np.concatenate((np.squeeze(targets), targets_latent), axis=1)

    return inputs_unity, target_unity, tags


dataFolders = [r"../data\Storage\00_AM_Randomized_1e5Batches/"]

modelsFolders = [r"../trainModels\1610562995_TrainedModels_Latent5/"]

modelNames = ["Unity_Latent5_1610562995-train_Beta0.001_AMFM_20x20x1_8bs-.keras"]


###########Load models
# AMFM
modelLocation = modelsFolders[0] + modelNames[0]
unityAMFM = keras.models.load_model(
    modelLocation,
    custom_objects={"customLoss": customLoss, "mseSignal": mseSignal, "mseLatent": mseLatent},
)
opt = unityAMFM.optimizer
unityAMFM.compile(optimizer=opt, loss=customLoss, metrics=[mseSignal, mseLatent])

###########Load Data
dataLocation = dataFolders[0]
testDataFile = "TestData.npz"
inputs, targets, tag = unityPrepData(dataLocation, testDataFile, latentDim)
print(np.shape(inputs))
print(np.shape(targets))

# Full losses for both models and data
lossAMFM = unityAMFM.evaluate(inputs, targets, batch_size=256)
unityOutputsAMFM = unityAMFM.predict(inputs)

#%% Load and prepare AM DNN

# Custom loss function. Weighted MSE over signal vs latent. Beta = 1 means we focus on latent.
latentDim = 7


def customLoss(yTrue, yPred):
    latentSize = latentDim
    SignalSize = 512
    beta = betaTag
    mseSignal = K.square(yTrue[:, 0:SignalSize] - yPred[:, 0:SignalSize])
    mseSignal = K.abs(mseSignal)
    mseSignal = K.sum(mseSignal, axis=-1)
    mseSignal = mseSignal / SignalSize

    mseLatent = K.square(yTrue[:, SignalSize:] - yPred[:, SignalSize:])
    mseLatent = K.abs(mseLatent)
    mseLatent = K.sum(mseLatent, axis=-1)
    mseLatent = mseLatent / latentSize

    weighted_mse = (1 - beta) * mseSignal + beta * mseLatent
    return weighted_mse


# Custom metric: latent param loss only
def mseLatent(yTrue, yPred):
    latentSize = latentDim
    SignalSize = 512
    mseLatent = K.square(yTrue[:, SignalSize:] - yPred[:, SignalSize:])
    mseLatent = K.abs(mseLatent)
    mseLatent = K.sum(mseLatent, axis=-1)
    mseLatent = mseLatent / latentSize
    return mseLatent


# Custom metric: signal loss only
def mseSignal(yTrue, yPred):
    SignalSize = 512
    mseSignal = K.square(yTrue[:, 0:SignalSize] - yPred[:, 0:SignalSize])
    mseSignal = K.abs(mseSignal)
    mseSignal = K.sum(mseSignal, axis=-1)
    mseSignal = mseSignal / SignalSize
    return mseSignal


# Load and concatenate train/val/test data.
def unityPrepData(dataFolder, trainDataFile, latentDimension):

    data = np.load(dataFolder + trainDataFile, allow_pickle=True)
    inputs_unity = data["inputs"]
    targets = data["targets"]
    targets_latent = data["metadata"]
    tags = data["setTag"]

    del data.f
    data.close()

    # Delete useless latent variable for smaller regressors
    if latentDimension == 5:
        targets_latent = np.delete(targets_latent, [3, 4], axis=1)  # No mod regressor
    if latentDimension == 3:
        targets_latent = np.delete(
            targets_latent, [3, 4, 5, 6], axis=1
        )  # No mod,tau,noise regressor

    target_unity = np.concatenate((np.squeeze(targets), targets_latent), axis=1)

    return inputs_unity, target_unity, tags


dataFolders = [r"../data\Storage\00_AM_Randomized_1e5Batches/"]

modelsFolders = [r"../trainModels\1610562995_TrainedModels_Latent7/"]

modelNames = ["Unity_Latent7_1610562995-train_Beta0.001_AM_10x20x1_8bs-.keras"]


###########Load models
# AM
modelLocation = modelsFolders[0] + modelNames[0]
unityAM = keras.models.load_model(
    modelLocation,
    custom_objects={"customLoss": customLoss, "mseSignal": mseSignal, "mseLatent": mseLatent},
)
opt = unityAM.optimizer
unityAM.compile(optimizer=opt, loss=customLoss, metrics=[mseSignal, mseLatent])

###########Load Data
dataLocation = dataFolders[0]
testDataFile = "TestData.npz"
inputs, targets, tag = unityPrepData(dataLocation, testDataFile, latentDim)
print(np.shape(inputs))
print(np.shape(targets))

# Full losses for both models and data
lossAM = unityAM.evaluate(inputs, targets, batch_size=256)
unityOutputsAM = unityAM.predict(inputs)

#%% Display one evaluation on a random sample.

index = 26
fontsizes = 14
# prepare data
inputDNN = np.reshape(inputs[index, 0:512], (1, 512, 1))
ydata = inputs[index, 0:512]
true = targets[index, 0:512]
true_latent = targets[index, 512:]

# DNN prediction and residu
predDNNAM = unityOutputsAM[index, 0:512]
predDNNAMFM = unityOutputsAMFM[index, 0:512]

latent_predDNNAM = unityOutputsAM[index, 512:]
latent_predDNNAMFM = unityOutputsAMFM[index, 512:]
latent_predDNNAMFM = np.insert(latent_predDNNAMFM, 3, 0)
latent_predDNNAMFM = np.insert(latent_predDNNAMFM, 3, 0)

fig = plt.figure()
plt.subplot(311)
plt.plot(ydata, color=noisycolor, linewidth=3, label="Input - noisy")
plt.plot(true, color=targetcolor, linewidth=2, label="Target - noiseless")
plt.xticks([])
plt.ylim(0.35, 0.65)
plt.legend(loc="upper left", fontsize=fontsizes - 5)
plt.subplot(312)
plt.plot(true, color=targetcolor, linewidth=6, label="Target - noiseless")
plt.plot(predDNNAM, color=dnncolor, linewidth=3, label="Specialized DNN pred.")
plt.plot(predDNNAMFM, color=fitcolor, linewidth=2, label="Partial DNN pred.")
plt.ylim(0.35, 0.65)
plt.xticks([])
plt.legend(loc="upper left", fontsize=fontsizes - 5)

plt.subplot(313)
colorsbar = (
    targetcolor,
    targetcolor,
    targetcolor,
    targetcolor,
    targetcolor,
    targetcolor,
    targetcolor,
)
xbar = ["$F_c$", "$\sin (\phi)$ ", "$\cos (\phi)$", "$F_m$", "$I_m$", "$T$", "$\sigma$"]

plt.bar(xbar, true_latent[:], width=0.55, label="Target - latent param.", color=colorsbar)
plt.bar(xbar, latent_predDNNAM, width=0.41, label="Specialized DNN pred.", color=dnncolor)
plt.bar(xbar, latent_predDNNAMFM, width=0.26, label="Partial DNN pred.", color=fitcolor)
plt.legend(loc="upper left", fontsize=fontsizes - 5)
plt.tick_params(axis="x", labelsize=fontsizes - 1)
plt.tight_layout()
plt.show()

#%% Compute perLatentParameter RMSE.
def getRMSE(true, pred):
    residu = true - pred
    resNorm = np.abs(true - pred)
    resNorm = resNorm**2
    resNorm = np.mean(resNorm, axis=0)
    RMSE = np.sqrt(resNorm)
    return RMSE


def getRMSESignal(true, pred):
    SignalSize = 512
    redisu = true[:, 0:SignalSize] - pred[:, 0:SignalSize]
    resNorm = np.abs(redisu)
    resNorm = resNorm**2
    resNorm = np.mean(resNorm, axis=1)
    resNorm = np.mean(resNorm, axis=0)
    RMSE = np.sqrt(resNorm)
    return RMSE


# RMSE vectors
DNNAMRMSE = np.zeros(7)
DNNAMFMRMSE = np.zeros(7)

# AM calculation
for ii in range(7):
    true = targets[:, 512 + ii]
    pred = unityOutputsAM[:, 512 + ii]
    DNNAMRMSE[ii] = getRMSE(true, pred)

# AMFM calculation
for ii in (0, 1, 2, 5, 6):

    true = targets[:, 512 + ii]

    if ii == 0 or ii == 1 or ii == 2:
        pred = unityOutputsAMFM[:, 512 + ii]
        DNNAMFMRMSE[ii] = getRMSE(true, pred)
    elif ii == 3 or ii == 4:
        DNNAMFMRMSE[ii] = 0
    elif ii == 5 or ii == 6:
        pred = unityOutputsAMFM[:, 512 + ii - 2]
        DNNAMFMRMSE[ii] = getRMSE(true, pred)


# Signal RMSE
true = targets[:, 0:512]
pred = unityOutputsAM[:, 0:512]
signalAMRMSE = getRMSESignal(true, pred)
DNNAMRMSE = np.insert(DNNAMRMSE, 0, signalAMRMSE)

pred = unityOutputsAMFM[:, 0:512]
signalAMFMRMSE = getRMSESignal(true, pred)
DNNAMFMRMSE = np.insert(DNNAMFMRMSE, 0, signalAMFMRMSE)


xbar = [
    "Signal",
    "$F_c$",
    "$\sin (\phi)$ ",
    "$\cos (\phi)$",
    "$F_m$",
    "$I_m$",
    "$T$",
    "$\sigma$",
]

plt.bar(xbar, DNNAMRMSE, width=0.6, label="Specialized DNN pred.", color=dnncolor)
plt.bar(xbar, DNNAMFMRMSE, width=0.4, label="Partial DNN pred.", color=fitcolor)
plt.legend(loc="upper left", fontsize=fontsizes - 5)
plt.tick_params(axis="x", labelsize=fontsizes - 1)
plt.tight_layout()
plt.title("Validation-set mean RMSE")
plt.show()
