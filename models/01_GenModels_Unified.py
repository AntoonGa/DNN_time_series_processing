# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:36:49 2020

@author: grab
"""
# -*- coding: utf-8 -*-
"""

Generate the neuralnet models.
We generate two initial models with 5 and 7 latent parameters respectively.


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
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import Input, Dense, Softmax, Flatten, Dropout, Reshape, Lambda, Concatenate
from keras.constraints import max_norm
from keras import metrics
import matplotlib.pyplot as plt
import math
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import keras

from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K

#%% load train and validation data
dataPath = r"../data\Storage\00_AMFM_Randomized_1e5Batches/"
fileName = "TestData.npz"
dataName = dataPath + fileName
data = np.load(dataName)
inputs = data["inputs"]
targets = data["targets"]
targets_latent = data["metadata"]

print(np.shape(inputs))
print(np.shape(targets))
print(np.shape(targets_latent))

# Looping through regressor latent dimension.
latentVariables = [7, 5]
timeStamp = str(round(time.time()))

for latentVariable in latentVariables:

    #% AUTOENCODER AND AUTOREGRESSOR ARCHITECTURE
    saving = True
    activation = "relu"
    initializer = "he_uniform"
    latent_dim = latentVariable
    #% Generate and save models
    bottleneck_dim = 64
    time.sleep(1)

    # ENCODER
    img_width = 512
    input_shape = (img_width, 1)
    maxpoolSize = 4

    EncoderName = "Enc_1conv1poolX2-2dense_" + timeStamp

    i = Input(shape=input_shape, name="encoder_input")
    x = Conv1D(64, kernel_size=64, activation=activation, padding="same")(i)
    x = MaxPooling1D(maxpoolSize, padding="same")(x)
    x = Conv1D(64, kernel_size=32, activation=activation, padding="same")(x)
    x = MaxPooling1D(maxpoolSize, padding="same")(x)
    x = Flatten()(x)
    x = Dense(128, activation=activation, kernel_initializer=initializer)(x)
    bottleneck = Dense(
        bottleneck_dim, activation=activation, kernel_initializer=initializer, name="bottleneck"
    )(x)

    encoder = Model(i, bottleneck, name="encoder")
    encoder.summary()

    # REGRESSOR
    RegressorName = "Reg_ConvPoolX2conv_denseX3_" + timeStamp

    r_i = Input(shape=(bottleneck_dim, 1), name="regressor_input")
    x = Reshape((bottleneck_dim, 1))(r_i)
    x = Conv1D(64, kernel_size=64, activation=activation, padding="same")(x)
    x = MaxPooling1D(maxpoolSize, padding="same")(x)
    x = Conv1D(64, kernel_size=32, activation=activation, padding="same")(x)
    x = MaxPooling1D(maxpoolSize, padding="same")(x)
    x = Conv1D(64, kernel_size=32, activation=activation, padding="same")(x)
    x = Flatten()(x)
    x = Dense(256, activation=activation, kernel_initializer=initializer)(x)
    x = Dense(128, activation=activation, kernel_initializer=initializer)(x)
    x = Dense(64, activation=activation, kernel_initializer=initializer)(x)
    latent = Dense(latent_dim, name="latentState")(x)

    regressor = Model(r_i, latent, name="regressor")
    regressor.summary()

    # DECODER adding a dense layer on the input just to be able to maintain dimension... This might change a few things !
    DecoderName = "Dec_1Dense_1conv1pool1upX3_bicephal" + timeStamp

    d_i = Input(shape=(bottleneck_dim + latent_dim,), name="decoder_input")
    # d_i = Concatenate(name='decoder_input')([regressor.output,encoder.output])

    # x = Reshape((bottleneck_dim+latent_dim,1))(d_i)
    # x = Flatten()(x)
    x = Dense(128, activation=activation, kernel_initializer=initializer)(d_i)
    x = Reshape((128, 1))(x)
    x = Conv1D(64, kernel_size=32, activation="relu", padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(64, kernel_size=32, activation="relu", padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(64, kernel_size=32, activation="relu", padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, kernel_size=32, activation="sigmoid", padding="same")(x)

    decoder = Model(d_i, decoded, name="decoder")
    decoder.summary()

    # Instantiate Aregressor
    arName = "AutoRegressor_" + timeStamp
    ar_outputs = regressor(encoder(i))
    ar = Model(i, ar_outputs, name="ar")
    ar.summary()

    # instantiate autoEncoder
    # Instantiate AE
    aeName = "AutoEncoder_" + timeStamp
    concat = Concatenate()([encoder(i), regressor(encoder(i))])
    ae_outputs = decoder(concat)
    ae = Model(i, ae_outputs, name="ae")
    ae.summary()

    # Unity model
    uniName = "Unity_Latent" + str(latent_dim) + "_" + timeStamp

    flattenAE = Reshape((512,))(ae_outputs)

    # Concatenate AutoEncoder and Regressor variables.
    concat2 = Concatenate()([flattenAE, regressor(encoder(i))])
    unity_outputs = concat2
    unity = Model(i, unity_outputs, name="unity")
    unity.summary()

    # Save models and model list in timeStampedFolder
    if saving == True:
        modelNames = uniName
        archiNames = (timeStamp, modelNames)
        print(archiNames)
        try:
            dirName = archiNames[0] + "_InitialModels_Latent" + str(latent_dim)
            os.mkdir(dirName, mode=0o777, dir_fd=None)
        except:
            print("dir not created")

        # ae.save(dirName    + "./"+archiNames[1][0]+ ".keras")
        # ar.save(dirName    + "./"+archiNames[1][1]+ ".keras")
        unity.save(dirName + "./" + archiNames[1] + ".keras")
        np.save(dirName + "./" + archiNames[0], archiNames[1])
