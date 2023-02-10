# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:11:24 2020

@author: grab
This script shows how to generate batches of datasets
Purely for display, no saving.

"""
import os

path = r"./"
os.chdir(path)
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# This is the signal generation class
from lib import generator_AMFMPOLY


def dataMosaic(noisyData, pureData, randomized=False):
    # Display a 100x100 subplot with random selections of validation data. nice mosaic
    colAndRow = 10
    cc = 0
    if randomized == True:
        sampled_list = random.sample(range(len(noisyData[:, 0])), colAndRow * colAndRow)
    else:
        sampled_list = [ii for ii in range(colAndRow * colAndRow)]

    fig, axs = plt.subplots(colAndRow, colAndRow)
    for ii in range(colAndRow):
        for jj in range(colAndRow):

            index = sampled_list[cc]
            axs[ii, jj].plot(noisyData[index, :])
            axs[ii, jj].plot(pureData[index, :], "k", lw=0.25)

            axs[ii, jj].axis("off")
            cc = cc + 1
    plt.show()


#%% generate train/validation/test data
# Data is randomized
# Meta data phase is mapped to cos/sin
freq = [0.01953125, 0.125]
phi = [0, 2 * 3.14]
tau = [6, 6]
fmod = [0.001953125, 0.009765625]
Imod = [0, 1]
noiseParams = [0, 0.5]
drifts = 0
sampleLength = 512
signalParams = [freq, phi, tau, fmod, Imod]
randomized = True
latentVariables = 7
numSamples = 1e3
batchNb = 0
seed = round(time.time())

outputShape = (numSamples, sampleLength, 1)

#%% AM data
test = generator_AMFMPOLY.generator_AMFMPOLY(
    outputShape=outputShape,
    randomized=randomized,
    display=10,
    signalParams=signalParams,
    noiseParams=noiseParams,
    driftParam=drifts,
    seed=seed,
    modTypes=["AM"],
)
carrier, modulation, modulated, real, noise, drift, metadata, metaTags = test.dataGen()

print("AM")
print(np.amax(modulated))
print(np.amin(modulated))
print(np.amax(real))
print(np.amin(real))
print(np.mean(modulated))
print(np.mean(real))

print(np.shape(carrier))

#%% FM data
test = generator_AMFMPOLY.generator_AMFMPOLY(
    outputShape=outputShape,
    randomized=randomized,
    display=10,
    signalParams=signalParams,
    noiseParams=noiseParams,
    driftParam=drifts,
    seed=seed,
    modTypes=["FM"],
)
carrier, modulation, modulated, real, noise, drift, metadata, metaTags = test.dataGen()

print("AM")
print(np.amax(modulated))
print(np.amin(modulated))
print(np.amax(real))
print(np.amin(real))
print(np.mean(modulated))
print(np.mean(real))

print(np.shape(carrier))
