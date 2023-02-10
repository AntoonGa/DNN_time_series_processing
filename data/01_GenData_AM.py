# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:11:24 2020

@author: grab
Generate amplitude modulated data

All latent parameters are normalized from 0 to 1 using the constraints of the generator
Noisy data is normalized from 0 to 1 in the generator
Noiseless data is normalized the same way
"""
print("------------------------------------------")
print("------------------------------------------")
print("-----------------AM---------------------")
print("------------------------------------------")
print("------------------------------------------")
import os

path = r"./"
os.chdir(path)
import matplotlib.pyplot as plt
import numpy as np
import random
import time
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
tau = [0.2, 8]
fmod = [0.001953125, 0.009765625]
Imod = [0, 1]
noiseParams = [0, 0.5]
drifts = 0
sampleLength = 512
signalParams = [freq, phi, tau, fmod, Imod]
randomized = True
latentVariables = 7

nbSample = 1e5
nbValSample = 5e4
batchNb = 22
seed = round(time.time())


timeStamp = "time_" + str(round(time.time()))
# saveDir = r"./temp/"
saveDir = r"./Storage/00_AM_Randomized_1e5Batches/"

try:
    os.stat(saveDir)
except:
    os.mkdir(saveDir)

# data generation
for ii in range(batchNb):
    seed = round(time.time())

    #% Save data  name and size.
    if ii == 0:
        saveName = "ValidationData.npz"
        numSamples = int(nbValSample)

    elif ii == 1:
        saveName = "TestData.npz"
        numSamples = int(nbSample)

    else:
        saveName = "TrainData_" + str(ii - 2) + ".npz"
        numSamples = int(nbSample)

    outputShape = (numSamples, sampleLength, 1)

    print("_______________________________")
    print("DATA SET " + str(ii + 1) + " of " + str(batchNb))
    print("_______________________________")

    outputShape = (numSamples, sampleLength, 1)

    test = generator_AMFMPOLY.generator_AMFMPOLY(
        outputShape=outputShape,
        randomized=randomized,
        display=1,
        signalParams=signalParams,
        noiseParams=noiseParams,
        driftParam=drifts,
        seed=seed,
        modTypes=["AM"],
    )
    _, _, modulated, real, _, _, metadata, metaTags = test.dataGen()

    print("AM set informations:")
    print("----max/min pure signals----")
    print(np.amax(modulated))
    print(np.amin(modulated))
    print("----max/min noisy signals----")
    print(np.amax(real))
    print(np.amin(real))
    print("----mean pure/noisy signals----")
    print(np.mean(modulated))
    print(np.mean(real))

    # Targets (fit params.), normalized
    # metaData has the following shape: see range to get proper normalization
    # metadata = [freq,phi,fmod,Imod,tau, noiseSigma, noiseDrift])
    # latentParamaters are scaled from 0 to 1, each type is rescaled independently
    # We use the global constrains of the generator as normalizers

    fmin = metaTags["NormalizationRanges"][0][0]
    fmax = metaTags["NormalizationRanges"][0][1]
    # phimin   = metaTags['NormalizationRanges'][1][0] #unused: sin/cos mapping
    # phimax   = metaTags['NormalizationRanges'][1][1]
    fmmin = metaTags["NormalizationRanges"][2][0]
    fmmax = metaTags["NormalizationRanges"][2][1]
    Imin = metaTags["NormalizationRanges"][3][0]
    Imax = metaTags["NormalizationRanges"][3][1]
    taumin = metaTags["NormalizationRanges"][4][0]
    taumax = metaTags["NormalizationRanges"][4][1]
    noisemin = metaTags["NormalizationRanges"][5][0]
    noisemax = metaTags["NormalizationRanges"][5][1]

    targets_latent = np.zeros((numSamples, latentVariables))
    targets_latent[:, 0] = (metadata[:, 0] - fmin) / (fmax - fmin)
    targets_latent[:, 1] = (np.sin(metadata[:, 1]) + 1) / 2
    targets_latent[:, 2] = (np.cos(metadata[:, 1]) + 1) / 2
    targets_latent[:, 3] = (metadata[:, 2] - fmmin) / (fmmax - fmmin)
    targets_latent[:, 4] = (metadata[:, 3] - Imin) / (Imax - Imin)
    targets_latent[:, 5] = (metadata[:, 4] - taumin) / (taumax - taumin)
    targets_latent[:, 6] = (metadata[:, 5] - noisemin) / (noisemax - noisemin)

    # enter whatever information you want here.
    meta = [
        randomized,
        "Normal 0 to 1",
        "512x1",
        "Conv1D input shape",
        "SinCos mapping",
        metaTags,
        timeStamp,
    ]

    # Data is saved as float32 in this dictionnary.
    DictToSave = {
        "inputs": np.float32(real),
        "targets": np.float32(modulated),
        "metadata": np.float32(targets_latent),
        "setTag": meta,
    }

    fullDir = saveDir + saveName
    np.savez(fullDir, **DictToSave)

    del modulated, real, metadata, metaTags, DictToSave

print("DONE")
