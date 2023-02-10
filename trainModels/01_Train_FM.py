# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:51:08 2020

@author: grab
Training the unity model
We add custom loss (betamse) in which the decoder and regressor losses are B-averaged
Beta a hyper parameter balacing the regressor and decoder loss.
We train both a small (no mod) and the full regressor on AM and FM data
the regressor and decoder losses are also stored separatly in the metrics functions.
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


# Load and concatenate train/val/test data.
def unityPrepData(dataFolder, trainDataFile, latentDimension):

    data = np.load(dataFolder + trainDataFile)
    inputs_unity = data["inputs"]
    targets = data["targets"]
    targets_latent = data["metadata"]
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

    return inputs_unity, target_unity


#%%  Frist training round: AutoEncoder only.
# model locations
timeStamp = "1610562995"
# Training parameters
bs = 8
superCycles = 10
cycles = 20
epochs = 1
betaTags = [0.001]
# data Folders
dataFolder = r"../data\Storage\00_FM_Randomized_1e5Batches/"
# AMFM tags
AMFMtag = "FM"
# Select thenumber of latent dimension the regressor has
latentDims = [7]


# Loop through latent dimension models
for betaTag in betaTags:
    for latentDim in latentDims:
        print("___________________________________________")
        print("---------------LATENT DIM: " + str(latentDim) + "---------------")
        print("___________________________________________")

        # LOAD MODEL dirs
        modelpath = r"../models/" + timeStamp + "_InitialModels_Latent" + str(latentDim)
        # SAVE MODEL dirs
        dirSaveModels = r"./" + timeStamp + "_TrainedModels_Latent" + str(latentDim) + "/"
        # SAVING TAGS
        trainTag0 = ""
        trainTag1 = (
            "-train_Beta"
            + str(betaTag)
            + "_"
            + AMFMtag
            + "_"
            + str(superCycles)
            + "x"
            + str(cycles)
            + "x"
            + str(epochs)
            + "_"
            + str(bs)
            + "bs-"
        )

        # LOAD VALIDATION DATA
        validationDataFile = "ValidationData.npz"
        inputs_unity_val, target_unity_val = unityPrepData(
            dataFolder, validationDataFile, latentDim
        )
        print(np.shape(inputs_unity_val))
        print(np.shape(target_unity_val))

        # Custom loss function. Weighted MSE over signal vs latent. Beta = 1 means we focus on latent.
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

        # LOAD MODEL and prepare names
        dirModels = modelpath + "/"
        modelslist = np.load(dirModels + timeStamp + ".npy", allow_pickle=True)
        uniName = str(modelslist) + trainTag0
        unity = keras.models.load_model(dirModels + uniName + ".keras")

        # COMPILE MODEL
        lr = 0.0001
        opt = keras.optimizers.Adam(learning_rate=lr)
        unity.compile(optimizer=opt, loss=customLoss, metrics=[mseSignal, mseLatent])

        print("___________________________________________")
        print("_________------- TRAINING -------__________")
        print("---------------TRAIN MODEL: " + str(betaTag) + "---------------")
        print(uniName)
        unity.summary()
        print("___________________________________________")
        print("___________________________________________")

        # CYCLE TRAINING
        unHistory = []
        initTime = time.time()
        for superCycle in range(superCycles):
            print("___________________________________________")
            print("___________Train Unity___________")
            print("ENTERING SUPERCYCLE: " + str(superCycle + 1) + " of : " + str(superCycles))
            print("___________________________________________")
            for cycle in range(cycles):
                print("___________________________________________")
                print("ENTERING CYCLE: " + str(cycle + 1) + " of : " + str(cycles))
                print("___________________________________________")

                # Load train data
                trainDataFile = "TrainData_" + str(cycle) + ".npz"
                inputs_unity, target_unity = unityPrepData(dataFolder, trainDataFile, latentDim)

                print("___________Train Unity___________")
                h_temp = unity.fit(
                    inputs_unity,
                    target_unity,
                    validation_data=(inputs_unity_val, target_unity_val),
                    epochs=epochs,
                    batch_size=bs,
                )
                unHistory.append(h_temp.history)

                # History
                a = [unHistory[ii]["loss"] for ii in range(len(unHistory))]
                b = [unHistory[ii]["val_loss"] for ii in range(len(unHistory))]
                c = [unHistory[ii]["mseSignal"] for ii in range(len(unHistory))]
                d = [unHistory[ii]["val_mseSignal"] for ii in range(len(unHistory))]
                e = [unHistory[ii]["mseLatent"] for ii in range(len(unHistory))]
                f = [unHistory[ii]["val_mseLatent"] for ii in range(len(unHistory))]
                loss = np.concatenate(a, axis=0)
                val_loss = np.concatenate(b, axis=0)
                mseSignal = np.concatenate(c, axis=0)
                val_mseSignal = np.concatenate(d, axis=0)
                mseLatent = np.concatenate(e, axis=0)
                val_mseLatent = np.concatenate(f, axis=0)

                # Plot histories
                if (cycle + 1) % 2 == 0 and cycle != 0:
                    ax1 = plt.subplot(311)
                    plt.plot(loss, "--", label="beta_loss")
                    plt.plot(val_loss, label="beta_val_loss")
                    plt.legend()
                    plt.xticks([])
                    plt.yscale("log")
                    plt.subplot(312)
                    plt.plot(mseSignal, "--", label="mseSignal")
                    plt.plot(val_mseSignal, label="val_mseSignal")
                    plt.legend()
                    plt.yscale("log")
                    plt.xticks([])
                    plt.subplot(313)
                    plt.plot(mseLatent, "--", label="mseLatent")
                    plt.plot(val_mseLatent, label="val_mseLatent")
                    plt.yscale("log")
                    plt.suptitle("model Beta: " + str(betaTag))
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

        # Evaluate trained model
        print("___________Evaluate model___________")
        trainDataFile = "TestData.npz"
        inputs_test, targets_test = unityPrepData(dataFolder, trainDataFile, latentDim)
        MSE_test, mseSignal_test, mseLatent_test = unity.evaluate(inputs_test, targets_test)
        print("MSE_test " + str(MSE_test))
        print("mseLatent_test " + str(mseLatent_test))
        print("mseSignal_test " + str(mseSignal_test))

        # SAVE MODEL AND RESULTS
        print("___________Saving model___________")
        trainTags = {
            "superCycles": superCycles,
            "cycles": cycles,
            "epochs": epochs,
            "bs": bs,
            "dataFolder": dataFolder,
            "trainTime": round(time.time() - initTime),
            "beta": betaTag,
            "AMFMtag": AMFMtag,
        }

        histSave = {
            "loss": loss,
            "val_loss": val_loss,
            "mseSignal": mseSignal,
            "val_mseSignal": val_mseSignal,
            "mseLatent": mseLatent,
            "val_mseLatent": val_mseLatent,
            "MSE_test": MSE_test,
            "mseSignal_test": mseSignal_test,
            "mseLatent_test": mseLatent_test,
            "trainTags": trainTags,
        }

        try:
            os.stat(dirSaveModels)
        except:
            os.mkdir(dirSaveModels)

        unity.save(dirSaveModels + uniName + trainTag1 + ".keras")
        np.savez(dirSaveModels + trainTag0 + trainTag1 + "_hist_", **histSave)

        print("_____Display some results of final model_______")
        unityOutputs = unity.predict(inputs_test)
        for ii in range(100):
            xbar = np.linspace(0, latentDim - 1, latentDim)
            plt.subplot(211)
            plt.plot(inputs_test[ii, 0:512])
            plt.plot(targets_test[ii, 0:512], "k")
            plt.plot(unityOutputs[ii, 0:512], alpha=0.9)
            plt.subplot(212)
            plt.bar(xbar, targets_test[ii, 512:])
            plt.bar(xbar, unityOutputs[ii, 512:], width=0.5)
            plt.suptitle(
                "Beta:"
                + str(betaTag)
                + " | MSE reg. (testD):"
                + str(round(mseLatent_test, 4))
                + " | MSE dec. (testD):"
                + str(round(mseSignal_test, 4))
            )
            plt.show()
