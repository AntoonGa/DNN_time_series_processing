# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:51:08 2020

@author: grab
model evaluation tool


This script tests the DNN outputs and compares them to a fit with true initial gueeses.
Select the correct "mode" for the script to load the proper DNN and data.

"""
# -*- coding: utf-8 -*-
# color Palette
targetcolor = "#661100"
fitcolor = "#6699CC"
fitcolor = "#332288"
dnncolor = "orange"
noisycolor = "#888888"


mode = "MONO"


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


dataFolders = [r"../data\Storage\00_MONO_Randomized_1e5Batches/"]

modelsFolders = [r"../trainModels\1610562995_TrainedModels_Latent5/"]

modelNames = ["Unity_Latent5_1610562995-train_Beta0.001_MONO_10x20x1_8bs-.keras"]


modelLocation = modelsFolders[0] + modelNames[0]
dataLocation = dataFolders[0]


###########Load models
unity = keras.models.load_model(
    modelLocation,
    custom_objects={"customLoss": customLoss, "mseSignal": mseSignal, "mseLatent": mseLatent},
)
opt = unity.optimizer
unity.compile(optimizer=opt, loss=customLoss, metrics=[mseSignal, mseLatent])


###########Load Data
testDataFile = "TestData.npz"
inputs, targets, tag = unityPrepData(dataLocation, testDataFile, latentDim)
print(np.shape(inputs))
print(np.shape(targets))


# Full losses for both models and data
loss = unity.evaluate(inputs, targets, batch_size=256)
unityOutputs = unity.predict(inputs)
print(np.shape(unityOutputs))


#%% single sample MSE vs noise
print("_____Test set MSE vs noise_______")


def getresidu(true, pred, latent_true, latent_pred):
    residu = true - pred
    resNorm = np.abs(true - pred)
    resNorm = resNorm**2
    resNorm = np.mean(resNorm)

    latentResidu = latent_true[:] - latent_pred[:]
    resNormLatent = np.abs(latentResidu)
    resNormLatent = resNormLatent**2
    resNormLatent = np.mean(resNormLatent)
    return residu, resNorm, latentResidu, resNormLatent


# nbSamples = np.shape(inputs)[0]
nbSamples = 1000
noiseLevels = targets[0:nbSamples, -1]
MSEunitySignal = np.zeros(nbSamples)
MSEunityLatent = np.zeros(nbSamples)


# pass through all testSet samples and compute residues:
# note that we remove the noise prediction as the fit cannot perform it.

for dataTarget in tqdm(range(nbSamples)):
    true = targets[dataTarget, 0:512]
    latent_true = targets[dataTarget, 512:-1]  # Noise param removed

    pred = unityOutputs[dataTarget, 0:512]
    latent_pred = unityOutputs[dataTarget, 512:-1]  # Noise param removed

    _, resNorm, _, resNormLatent = getresidu(true, pred, latent_true, latent_pred)

    MSEunitySignal[dataTarget] = resNorm
    MSEunityLatent[dataTarget] = resNormLatent

# sort by noiselevel
sortedNoise = np.sort(noiseLevels)
sortedMSEunitySignal = MSEunitySignal[np.argsort(noiseLevels)]
sortedMSEunityLatent = MSEunityLatent[np.argsort(noiseLevels)]

fontsizes = 14
fig, axs = plt.subplots(2, sharex="all")

axs[0].scatter(
    sortedNoise, sortedMSEunitySignal, marker="*", label="DNN", c="orange", alpha=0.8
)
axs[0].set_yscale("log")
axs[0].legend()
axs[0].set_ylabel("$MSE_{dec}$ [-]", fontsize=fontsizes)

axs[1].scatter(
    sortedNoise, sortedMSEunityLatent, marker="*", label="DNN", c="orange", alpha=0.8
)
axs[1].set_yscale("log")
axs[1].legend()
axs[1].set_xlabel("Noise Level [a.u]", fontsize=fontsizes)
axs[1].set_ylabel("$MSE_{reg}$ [-]", fontsize=fontsizes)
plt.tight_layout()
plt.show()

#%% single sample MSE vs noise with FIT with true initial guesses.
# FIT FUNCTION
print("_____FIT test set MSE vs noise_______")

import scipy
from scipy.optimize import curve_fit


def doFit(xdata, ydata, initial, normalizationRange):

    # def models AM/FM
    def MONOfunc(x, freq, phi, Tau):

        ys = 1 * np.cos(2 * 3.14 * x * freq + phi) * np.exp(-x / (Tau * 511))
        ms = 1
        out = np.multiply(ys, ms)
        # generator normalization inverse
        theMin = -1
        theMax = 1
        out = (out - theMin) / (theMax - theMin) + -0.5
        out = out / 5 + 0.5
        return out

    # Normalization of latent parameters.
    fmin = normalizationRange[0][0]
    fmax = normalizationRange[0][1]
    # phimin   = metaTags['NormalizationRanges'][1][0] #unused: sin/cos mapping
    # phimax   = metaTags['NormalizationRanges'][1][1]
    # fmmin    = normalizationRange[2][0]
    # fmmax    = normalizationRange[2][1]
    # Imin     = normalizationRange[3][0]
    # Imax     = normalizationRange[3][1]
    taumin = normalizationRange[4][0]
    taumax = normalizationRange[4][1]

    # Produce initial guesses with true values ("denormalize them beforehand)
    freq0 = initial[0] * (fmax - fmin) + fmin
    x = initial[1]
    y = initial[2]
    phase = np.arctan2(x * 2 - 1, y * 2 - 1)

    if phase < 0:
        phase = phase + 2 * 3.14

    phi0 = phase
    # Fmod0 = initial[3] * (fmmax-fmmin) + fmmin
    # Imod0 = initial[4] * (Imax-Imin) + Imin
    Tau0 = initial[3] * (taumax - taumin) + taumin
    initGuess = [freq0, phi0, Tau0]

    # fit
    bounds = ([fmin * 0.5, 0, -taumin * 0.5], [fmax * 1.5, 3 * 3.15, taumax * 1.5])
    popt, pcov = curve_fit(MONOfunc, xdata, ydata, p0=initGuess, bounds=bounds, maxfev=5500)
    fitEval = MONOfunc(xdata, *popt)

    # Rescaling outputs to 0-1 as in the generator to accomodate for DNN comparison
    freqRescaled = (popt[0] - fmin) / (fmax - fmin)
    sinphiRescaled = (np.sin(popt[1]) + 1) / 2
    cosphiRescaled = (np.cos(popt[1]) + 1) / 2
    # fmodRescaled   = ( popt[2] - fmmin  )  / (fmmax-fmmin)
    # ImodRescaled   = ( popt[3] - Imin   )  / (Imax-Imin)
    tauRescaled = (popt[2] - taumin) / (taumax - taumin)

    normalpopt = [freqRescaled, sinphiRescaled, cosphiRescaled, tauRescaled]

    return fitEval, popt, np.transpose(normalpopt), np.transpose(initGuess)


normalizationRange = tag[5]["NormalizationRanges"]

# single fit
# nbSamples = np.shape(inputs)[0]
noiseLevels = targets[0:nbSamples, -1]
MSEfitSignal = np.zeros(nbSamples)
MSEfitLatent = np.zeros(nbSamples)

freqsFIT = np.zeros(nbSamples)
sinsFIT = np.zeros(nbSamples)
cossFIT = np.zeros(nbSamples)
TausFIT = np.zeros(nbSamples)

# pass through all testSet samples and compute residues:
for dataTarget in tqdm(range(nbSamples)):

    xdata = np.linspace(0, 511, 512)
    ydata = np.squeeze(inputs[dataTarget, 0:512])

    initialGuesses = targets[dataTarget, 512:-1]
    fitEval, popt, normalpopt, initGuess = doFit(
        xdata, ydata, initialGuesses, normalizationRange
    )

    freqsFIT[dataTarget] = normalpopt[0]
    sinsFIT[dataTarget] = normalpopt[1]
    cossFIT[dataTarget] = normalpopt[2]
    TausFIT[dataTarget] = normalpopt[3]

    true = targets[dataTarget, 0:512]
    latent_true = targets[dataTarget, 512:-1]

    pred = fitEval
    latent_pred = normalpopt

    _, resNorm, _, resNormLatent = getresidu(true, pred, latent_true, latent_pred)

    MSEfitSignal[dataTarget] = resNorm
    MSEfitLatent[dataTarget] = resNormLatent

    # xbar = np.linspace(0,latentDim-2,latentDim-1)
    # plt.subplot(311)
    # plt.plot(ydata)
    # plt.plot(fitEval, alpha=0.9)
    # plt.subplot(312)
    # plt.plot(true,'k--')
    # plt.plot(fitEval, alpha=0.9)
    # plt.plot(true-fitEval + 0.5, alpha=0.5)
    # plt.subplot(313)
    # plt.bar(xbar,latent_true)
    # plt.bar(xbar,latent_pred, width=0.5)
    # plt.show()

# sort by noiselevel
sortedNoise = np.sort(noiseLevels)
sortedMSEfitSignal = MSEfitSignal[np.argsort(noiseLevels)]
sortedMSEfitLatent = MSEfitLatent[np.argsort(noiseLevels)]

fontsizes = 14
fig, axs = plt.subplots(2, sharex="all")

axs[0].scatter(sortedNoise, sortedMSEfitSignal, marker="*", label="fit", c="orange", alpha=0.8)
axs[0].set_yscale("log")
axs[0].legend()
axs[0].set_ylabel("$MSE_{dec}$ [-]", fontsize=fontsizes)

axs[1].scatter(sortedNoise, sortedMSEfitLatent, marker="*", label="fit", c="orange", alpha=0.8)
axs[1].set_yscale("log")
axs[1].legend()
axs[1].set_xlabel("Noise Level [a.u]", fontsize=fontsizes)
axs[1].set_ylabel("$MSE_{reg}$ [-]", fontsize=fontsizes)
plt.tight_layout()
plt.show()

#%% DNNs assisted fits : use DNN latent output as init guessees for fit.
# FIT FUNCTION
print("_____DNN-assisted FIT test set MSE vs noise_______")


# single fit
# nbSamples = np.shape(inputs)[0]
MSEDNNassistedfitSignal = np.zeros(nbSamples)
MSEDNNassistedfitLatent = np.zeros(nbSamples)

freqsDNNassistedFIT = np.zeros(nbSamples)
sinsDNNassistedFIT = np.zeros(nbSamples)
cossDNNassistedFIT = np.zeros(nbSamples)
TausDNNassistedFIT = np.zeros(nbSamples)

# pass through all testSet samples and compute residues:
for dataTarget in tqdm(range(nbSamples)):

    xdata = np.linspace(0, 511, 512)
    ydata = np.squeeze(inputs[dataTarget, 0:512])

    initialGuesses = unityOutputs[dataTarget, 512:-1]
    # targets[dataTarget,512:-1]
    fitEval, popt, normalpopt, initGuess = doFit(
        xdata, ydata, initialGuesses, normalizationRange
    )

    freqsDNNassistedFIT[dataTarget] = normalpopt[0]
    sinsDNNassistedFIT[dataTarget] = normalpopt[1]
    cossDNNassistedFIT[dataTarget] = normalpopt[2]
    TausDNNassistedFIT[dataTarget] = normalpopt[3]

    true = targets[dataTarget, 0:512]
    latent_true = targets[dataTarget, 512:-1]

    pred = fitEval
    latent_pred = normalpopt

    _, resNorm, _, resNormLatent = getresidu(true, pred, latent_true, latent_pred)

    MSEDNNassistedfitSignal[dataTarget] = resNorm
    MSEDNNassistedfitLatent[dataTarget] = resNormLatent

    # xbar = np.linspace(0,latentDim-2,latentDim-1)
    # plt.subplot(311)
    # plt.plot(ydata)
    # plt.plot(fitEval, alpha=0.9)
    # plt.subplot(312)
    # plt.plot(true,'k--')
    # plt.plot(fitEval, alpha=0.9)
    # plt.plot(true-fitEval + 0.5, alpha=0.5)
    # plt.subplot(313)
    # plt.bar(xbar,latent_true)
    # plt.bar(xbar,latent_pred, width=0.5)
    # plt.show()

# sort by noiselevel
sortedMSEDNNassistedfitSignal = MSEDNNassistedfitSignal[np.argsort(noiseLevels)]
sortedMSEDNNassistedfitLatent = MSEDNNassistedfitLatent[np.argsort(noiseLevels)]

fontsizes = 14
fig, axs = plt.subplots(2, sharex="all")

axs[0].scatter(
    sortedNoise, sortedMSEDNNassistedfitSignal, marker="*", label="fit", c="orange", alpha=0.8
)
axs[0].set_yscale("log")
axs[0].legend()
axs[0].set_ylabel("$MSE_{dec}$ [-]", fontsize=fontsizes)

axs[1].scatter(
    sortedNoise, sortedMSEDNNassistedfitLatent, marker="*", label="fit", c="orange", alpha=0.8
)
axs[1].set_yscale("log")
axs[1].legend()
axs[1].set_xlabel("Noise Level [a.u]", fontsize=fontsizes)
axs[1].set_ylabel("$MSE_{reg}$ [-]", fontsize=fontsizes)
plt.tight_layout()
plt.show()

#%% COMPARE fit and DNN-assited FITS

normallizerToOne1 = np.amax(
    [np.amax(sortedMSEfitSignal), np.amax(sortedMSEDNNassistedfitSignal)]
)
normallizerToOne2 = np.amax(
    [np.amax(sortedMSEfitLatent), np.amax(sortedMSEDNNassistedfitLatent)]
)

fontsizes = 14
fig, axs = plt.subplots(2)

axs[0].scatter(
    sortedNoise,
    sortedMSEfitSignal / normallizerToOne1,
    label="Fit w/ true guess",
    c=fitcolor,
    alpha=1,
    s=50,
)
axs[0].scatter(
    sortedNoise,
    sortedMSEDNNassistedfitSignal / normallizerToOne1,
    marker="*",
    label="DNN",
    c=dnncolor,
    alpha=0.75,
    s=25,
)
axs[0].set_yscale("log")
axs[0].legend(fontsize=fontsizes)
axs[0].set_ylabel("$MSE_{dec}$ [rel.]", fontsize=fontsizes)

axs[1].scatter(
    sortedNoise,
    sortedMSEfitLatent / normallizerToOne2,
    label="Fit w/ true guess",
    c=fitcolor,
    alpha=1,
    s=50,
)
axs[1].scatter(
    sortedNoise,
    sortedMSEDNNassistedfitLatent / normallizerToOne2,
    marker="*",
    label="DNN",
    c=dnncolor,
    alpha=0.75,
    s=25,
)
axs[1].set_yscale("log")
axs[1].legend(fontsize=fontsizes)
axs[1].set_xlabel("Noise Level [a.u]", fontsize=fontsizes)
axs[1].set_ylabel("$MSE_{reg}$ [rel.]", fontsize=fontsizes)

axs[1].tick_params(axis="both", which="major", labelsize=fontsizes - 1)
axs[0].tick_params(axis="both", which="major", labelsize=fontsizes - 1)
axs[0].set_xticks([])

plt.tight_layout()
plt.show()

plt.scatter(freqsFIT, sinsFIT, label="Fit w/ true guess", c=fitcolor, alpha=1, s=50)
plt.scatter(
    freqsDNNassistedFIT,
    sinsDNNassistedFIT,
    label="DNN-assisted fit",
    marker="*",
    c=dnncolor,
    alpha=0.75,
    s=25,
)
# plt.legend(fontsize = fontsizes, loc=2)
plt.xlabel("$F_c$ [a.u]", fontsize=fontsizes)
plt.ylabel("$ ( \sin (\phi) + 1 )$" + " " + "$/$" + " " + "$2$ [-]", fontsize=fontsizes)
plt.xticks(fontsize=fontsizes - 1)
plt.yticks(fontsize=fontsizes - 1)

plt.show()
