# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:19:26 2020
Attempt at normalizing
@author: grab
listwise waveform generator
You can nosie input a drift level!

call dataGen() to output:
    carrier, modulation, modulated, real, noise, drift, metadata, tag, dataRanges_ouputs, normalRanges_ouputs
# params are as following:
    
    #these are constrained in a certain range by the integration time and dt.
    #inputing values outside de range with narrow them down to the range
    #NOTE: FMOD IS NOT USED IN THE COMPUTATION. ONLY IMOD IS.
    #TURN OFF DECAY by setting TAU to 0
    #TURN OFF MODULATION by setting IMOD to 0
    
    #The signal is a sine-modulated-sine with a decay
    #On top of which Gaussian noise and drifts are linearly added
    #The normalization is strong enough that most values live between 0.3 to 0.7
    #The gaussian noise variance is picked up from the range each iteration of a signal
    #The drift is somewhat constant along the entire databatch, however this does not mean it is the same !
    
    #Randomized = True will insure all the variables are picked randomly from the widest range of parameters, usefull for general training
    
    #Fs is 1 sec, dt is 1sec, therefore the signal length is the size of the signal
    #The outputShape input insures automatic conversion to the neural network shape : outputShape = ( nBsamples, sampleLength, virtual dim = 1, virtual dim= 1  )
       
    # metaData has the following shape:
    # [freq,phi,fmod,Imod,tau, noiseSigma, noiseDrift]

"""
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import bottleneck as bn
from tqdm import tqdm
import math
from numpy import random
import scipy.special
import traceback


class generator_AMFMPOLY:
    def __init__(
        self,
        outputShape=(1000, 512, 1, 1),
        randomized=True,
        display=0,
        signalParams=[[0.05, 0.1], [0, 2 * 3.14], [0, 0.5], [100, 100], [0, 1]],
        noiseParams=[0.1, 0.1],
        driftParam=0.1,
        seed=None,
        modTypes=["AM"],
    ):
        # seed value
        self.seed = seed

        # Intersample configuration
        self.modTypes = modTypes

        # Full set configuration
        self.outputShape = outputShape
        self.total_num_samples = int(self.outputShape[0])
        self.randomized = randomized
        self.display = int(display)

        # Intrasample configuration
        self.sample_size = int(outputShape[1])
        self.starting_point = 0
        self.maxTime = self.sample_size
        self.dt = 1
        self.Fs = 1

        # signal and noise params physical limitations
        self.amp = 1  # amplitude is always 1, noise is the scale parameters
        self.fMax = (
            self.Fs / 8
        )  # 8 samples per oscillation, more stringent than the usual Nqyst
        self.fMin = 10 / self.maxTime  # 10 full oscillation per measurement
        self.tauMin = 0.2  # this is in units of total time! (exponential decay)
        self.tauMax = 8
        self.fmodMin = (
            1 / self.maxTime
        )  # 1 full oscillation per measurement        #fmod ranges
        self.fmodMax = self.fMin / 2  # Always smaller than the carrier
        self.ImodMin = 0  # modulation index, unitless.
        self.ImodMax = 1
        self.noiseMin = 0  # Gaussian noise (linear combinaison with signal, this is the coeff)
        self.noiseMax = 0.5
        self.driftMin = 0  # NOT USED; you can input any value ! random walk added to the noise linearly with that coef.
        self.driftMax = 1

        # assign signal generation param range: this insures the input range is withing physical boundaries
        signalParams = np.sort(signalParams)
        self.freqRange = [
            min(max(signalParams[0][0], self.fMin), self.fMax),
            max(min(signalParams[0][1], self.fMax), self.fMin),
        ]
        self.phiRange = [
            min(max(signalParams[1][0], 0), 3.14 * 2),
            max(min(signalParams[1][1], 3.14 * 2), 0),
        ]
        self.tauRange = [
            min(max(signalParams[2][0], self.tauMin), self.tauMax),
            max(min(signalParams[2][1], self.tauMax), self.tauMin),
        ]
        self.fmodRange = [
            min(max(signalParams[3][0], self.fmodMin), self.fmodMax),
            max(min(signalParams[3][1], self.fmodMax), self.fmodMin),
        ]
        self.ImodRange = [
            min(max(signalParams[4][0], self.ImodMin), self.ImodMax),
            max(min(signalParams[4][1], self.ImodMax), self.ImodMin),
        ]

        # Correct meta actual Values, zero turns off the effect
        if self.ImodRange == [0, 0]:
            self.fmodRange = [0, 0]

        if signalParams[2][0] == 0 and signalParams[2][1] == 0:
            self.tauRange = [0, 0]

        # noise param param range: this insures the input range is withing boundaries
        noiseParams = np.sort(noiseParams)
        self.noiseRange = [
            min(max(noiseParams[0], self.noiseMin), self.noiseMax),
            max(min(noiseParams[1], self.noiseMax), self.noiseMin),
        ]
        self.drift = driftParam

        # Display options
        self.num_samples_visualize = int(min(self.display, self.total_num_samples))

        # Display signal and noise ranges during data generation
        if randomized == True:
            # assign signal param range: this insures the input range is withing boundaries
            signalParams = np.sort(signalParams)
            self.freqRange = [self.fMin, self.fMax]
            self.phiRange = [0, 3.14 * 2]
            self.tauRange = [self.tauMin, self.tauMax]
            self.fmodRange = [self.fmodMin, self.fmodMax]
            self.ImodRange = [self.ImodMin, self.ImodMax]
            print("___________Default Ranges__________")
        else:
            print("_______________Ranges______________")
        print("freqRange: " + str(self.freqRange))
        print("phiRange:  " + str(self.phiRange))
        print("tauRange:  " + str(self.tauRange))
        print("fmodRange: " + str(self.fmodRange))
        print("ImodRange: " + str(self.ImodRange))
        print("noiseRange:" + str(self.noiseRange))
        print("drift:      " + str(self.drift))
        print("_______________________________________")

        time.sleep(0.5)

        # Generate range arrays of output, and normalization ranges (physical boundaries), these an output
        self.dataRanges_ouputs = [
            self.freqRange,
            self.phiRange,
            self.fmodRange,
            self.ImodRange,
            self.tauRange,
            self.noiseRange,
        ]
        self.normalRanges_ouputs = [
            [self.fMin, self.fMax],
            [0, 2 * 3.14],
            [self.fmodMin, self.fmodMax],
            [self.ImodMin, self.ImodMax],
            [self.tauMin, self.tauMax],
            [self.noiseMin, self.noiseMax],
        ]

    # Generating pure waveforms by calling respective function and rescale
    def waveforms(self):
        # Containers for samples and subsamples
        carrier = []
        modulation = []
        modulated = []
        metadata = []
        tag = []

        numSamplePerModType = int(self.total_num_samples / len(self.modTypes))

        if "AM" in self.modTypes:
            (
                carrier_temp,
                modulation_temp,
                modulated_temp,
                metadata_temp,
                tag_temp,
            ) = self.AMforms(numSamplePerModType)

            # append data
            carrier.append(carrier_temp)
            modulation.append(modulation_temp)
            modulated.append(modulated_temp)
            metadata.append(metadata_temp)
            tag.append(tag_temp)

        if "FM" in self.modTypes:
            (
                carrier_temp,
                modulation_temp,
                modulated_temp,
                metadata_temp,
                tag_temp,
            ) = self.FMforms(numSamplePerModType)

            # append data
            carrier.append(carrier_temp)
            modulation.append(modulation_temp)
            modulated.append(modulated_temp)
            metadata.append(metadata_temp)
            tag.append(tag_temp)

        if "POLY" in self.modTypes:
            (
                carrier_temp,
                modulation_temp,
                modulated_temp,
                metadata_temp,
                tag_temp,
            ) = self.POLYforms(numSamplePerModType)

            # append data
            carrier.append(carrier_temp)
            modulation.append(modulation_temp)
            modulated.append(modulated_temp)
            metadata.append(metadata_temp)
            tag.append(tag_temp)

        # Reshape data
        a = np.shape(carrier)[0]
        b = np.shape(carrier)[1]
        c = np.shape(carrier)[2]
        carrier = np.squeeze(np.reshape(carrier, (1, a * b, c)))
        modulation = np.squeeze(np.reshape(modulation, (1, a * b, c)))
        modulated = np.squeeze(np.reshape(modulated, (1, a * b, c)))
        metadata = np.concatenate(metadata, axis=0)
        tag = np.squeeze(np.reshape(tag, (1, a * b, 3)))

        # Generate Gaussian Noise over all batch
        noise = self.gaussianNoise(carrier, metadata)
        # Generate Drifts over all batch
        drift = self.driftNoise(noise)

        # real samples
        real = np.add(modulated, np.add(drift, noise))

        # Rescale data
        # this insures the noisy data is largely between 0 and 1. All other fields are normalized the same way
        scale = 5
        carrier = np.array(carrier) / scale + 0.5
        modulation = np.array(modulation) / scale + 0.5
        modulated = np.array(modulated) / scale + 0.5
        real = np.array(real) / scale + 0.5
        noise = np.array(noise) / scale + 0.5
        drift = np.array(drift) / scale + 0.5

        # Final reshape
        carrier = np.expand_dims(carrier, axis=-1)
        modulation = np.expand_dims(modulation, axis=-1)
        modulated = np.expand_dims(modulated, axis=-1)
        real = np.expand_dims(real, axis=-1)
        noise = np.expand_dims(noise, axis=-1)
        drift = np.expand_dims(drift, axis=-1)

        return carrier, modulation, modulated, real, noise, drift, metadata, tag

    # Call waveforms and noise generation function. Then call displays.
    def dataGen(self):
        timeStamp1 = time.time()

        carrier, modulation, modulated, real, noise, drift, metadata, tag = self.waveforms()

        # Shuffle and display
        self.shuffle(carrier, modulation, modulated, noise, drift, real, metadata, tag)
        self.displayFunc(carrier, modulation, modulated, noise, drift, real, metadata, tag)

        try:
            plt.hist(modulated.flatten(), bins=100)
            plt.title("Noiseless modulated data distribution")
            plt.show()
            plt.hist(real.flatten(), bins=100)
            plt.title("Noisy modulated data distribution")
            plt.show()
        except:
            print("Cannot print histograms")

        timeStamp2 = time.time()
        dispstring = str(round(timeStamp2 - timeStamp1, 2))
        print("__________________________________")
        print("Data ready, in: " + dispstring + " seconds")
        print("Shape data: " + str(np.shape(carrier)))
        print("Shape Meta: " + str(np.shape(metadata)))
        print("__________________________________")

        metaTags = {
            "ModTypes": self.modTypes,
            "ModTags": tag,
            "UserRanges": self.dataRanges_ouputs,
            "NormalizationRanges": self.normalRanges_ouputs,
            "RandomSeed": self.seed,
            "MetadataShape": "[freq,phi,fmod,Imod,tau, noiseSigma, noiseDrift]",
        }
        print("metaTags fields:" + str(metaTags.keys()))
        return carrier, modulation, modulated, real, noise, drift, metadata, metaTags

    # AM Modulated sinewaves
    def AMforms(self, numSamplePerModType):
        random.seed(self.seed)

        # placeholder for signals and modulations
        carrier_out = []
        modulation_out = []
        modulated_out = []
        metadata_out = []
        tag_out = []

        for j in tqdm(range(0, numSamplePerModType), desc="Generating Carriers and AM mods: "):
            # draw single instance of latent param
            freq, phi, Imod, fmod, tau, noiseSigma, noiseDrift = self.drawLatentParams(j)

            # Generate sinewave
            if tau > 0:
                carrier_temp = [
                    1
                    * math.cos(2 * 3.14 * ii * freq + phi)
                    * math.exp(-ii / (tau * self.maxTime))
                    for ii in range(self.sample_size)
                ]
            else:
                carrier_temp = [
                    1 * math.cos(2 * 3.14 * ii * freq + phi) for ii in range(self.sample_size)
                ]
            # generate modulation
            modulation_temp = [
                Imod * math.cos(2 * 3.14 * ii * fmod) for ii in range(self.sample_size)
            ]

            carrier_temp = np.array(carrier_temp)
            modulation_temp = np.array(modulation_temp)
            # generate AM modulated function
            modulated_temp = np.multiply(carrier_temp, (1 + modulation_temp))

            # Rescale to 1! modulated function is rescaled from 0 to 1, then DC shifted to zero mean
            thMin = -2  # sum of two sines can go up to two
            thMax = 2
            carrier_temp = (carrier_temp - thMin) / (thMax - thMin) - 0.5
            modulation_temp = (modulation_temp - thMin) / (thMax - thMin) - 0.5
            modulated_temp = (modulated_temp - thMin) / (thMax - thMin) - 0.5

            carrier_out.append(carrier_temp)
            modulation_out.append(modulation_temp)
            modulated_out.append(modulated_temp)
            metadata_out.append([freq, phi, fmod, Imod, tau, noiseSigma, noiseDrift])
            tag_out.append([1, 0, 0])

        return carrier_out, modulation_out, modulated_out, metadata_out, tag_out

    # FM Modulated sinewaves
    def FMforms(self, numSamplePerModType):
        random.seed(self.seed)

        # placeholder for signals and modulations
        carrier_out = []
        modulation_out = []
        modulated_out = []
        metadata_out = []
        tag_out = []

        for j in tqdm(range(0, numSamplePerModType), desc="Generating Carriers and FM mods: "):
            # draw single instance of latent param
            freq, phi, Imod, fmod, tau, noiseSigma, noiseDrift = self.drawLatentParams(j)

            # Generate sinewave
            if tau > 0:
                carrier_temp = [
                    1
                    * math.cos(2 * 3.14 * ii * freq + phi)
                    * math.exp(-ii / (tau * self.maxTime))
                    for ii in range(self.sample_size)
                ]
            else:
                carrier_temp = [
                    1 * math.cos(2 * 3.14 * ii * freq + phi) for ii in range(self.sample_size)
                ]
            # generate modulation
            modulatorSens = 0.01
            mindex = modulatorSens * Imod / fmod
            modulation_temp = [
                Imod * math.cos(2 * 3.14 * ii * fmod) for ii in range(self.sample_size)
            ]
            # generate FM signal
            modulated_temp = [
                1
                * math.cos(
                    2 * 3.14 * ii * freq
                    + 2 * 3.14 * mindex * math.cos(2 * 3.14 * ii * fmod)
                    + phi
                )
                * math.exp(-ii / (tau * self.maxTime))
                for ii in range(self.sample_size)
            ]

            # Reshape
            carrier_temp = np.array(carrier_temp)
            modulation_temp = np.array(modulation_temp)
            modulated_temp = np.array(modulated_temp)

            # Rescale to 1! modulated function is rescaled from 0 to 1, then DC shifted to zero mean
            thMin = -1  # modulation of 1 sine can go up to 1
            thMax = 1
            carrier_temp = (carrier_temp - thMin) / (thMax - thMin) - 0.5
            modulation_temp = (modulation_temp - thMin) / (thMax - thMin) - 0.5
            modulated_temp = (modulated_temp - thMin) / (thMax - thMin) - 0.5

            carrier_out.append(carrier_temp)
            modulation_out.append(modulation_temp)
            modulated_out.append(modulated_temp)
            metadata_out.append([freq, phi, fmod, Imod, tau, noiseSigma, noiseDrift])
            tag_out.append([0, 1, 0])

        return carrier_out, modulation_out, modulated_out, metadata_out, tag_out

    # polychromatic wave
    def POLYforms(self, numSamplePerModType):
        random.seed(self.seed)

        # placeholder for signals and modulations
        carrier_out = []
        modulation_out = []
        modulated_out = []
        metadata_out = []
        tag_out = []

        for j in tqdm(range(0, numSamplePerModType), desc="Generating polychromatic: "):
            # draw single instance of latent param
            freq, phi, Imod, fmod, tau, noiseSigma, noiseDrift = self.drawLatentParams(j)

            # Generate sinewave
            if tau > 0:
                carrier_temp = [
                    1
                    * math.cos(2 * 3.14 * ii * freq + phi)
                    * math.exp(-ii / (tau * self.maxTime))
                    for ii in range(self.sample_size)
                ]
            else:
                carrier_temp = [
                    1 * math.cos(2 * 3.14 * ii * freq + phi) for ii in range(self.sample_size)
                ]
            # generate modulation
            modulation_temp = [
                Imod * math.cos(2 * 3.14 * ii * fmod) for ii in range(self.sample_size)
            ]
            # generate polychromatic signal
            modulated_temp = [
                (math.cos(2 * 3.14 * ii * freq + phi) + Imod * math.cos(2 * 3.14 * ii * fmod))
                * math.exp(-ii / (tau * self.maxTime))
                for ii in range(self.sample_size)
            ]

            # Reshape
            carrier_temp = np.array(carrier_temp)
            modulation_temp = np.array(modulation_temp)
            modulated_temp = np.array(modulated_temp)

            # Rescale to 1! modulated function is rescaled from 0 to 1, then DC shifted to zero mean
            thMin = -2  # sum of two sines can go up to 2
            thMax = 2
            carrier_temp = (carrier_temp - thMin) / (thMax - thMin) - 0.5
            modulation_temp = (modulation_temp - thMin) / (thMax - thMin) - 0.5
            modulated_temp = (modulated_temp - thMin) / (thMax - thMin) - 0.5

            carrier_out.append(carrier_temp)
            modulation_out.append(modulation_temp)
            modulated_out.append(modulated_temp)
            metadata_out.append([freq, phi, fmod, Imod, tau, noiseSigma, noiseDrift])
            tag_out.append([0, 0, 1])

        return carrier_out, modulation_out, modulated_out, metadata_out, tag_out

    def drawLatentParams(self, j):
        random.seed(self.seed + j)
        # Sinewave, modulation and decay parameters
        freq = random.uniform(self.freqRange[0], self.freqRange[1])
        phi = random.uniform(self.phiRange[0], self.phiRange[1])
        Imod = random.uniform(self.ImodRange[0], self.ImodRange[1])
        tau = random.uniform(self.tauRange[0], self.tauRange[1])
        noiseSigma = random.uniform(self.noiseRange[0], self.noiseRange[1])
        noiseDrift = self.drift

        # fmod is always slower than the carrier
        fmod = random.uniform(self.fmodRange[0], self.fmodRange[1])

        return freq, phi, Imod, fmod, tau, noiseSigma, noiseDrift

    def gaussianNoise(self, pure_samples, metadata):
        random.seed(self.seed)

        timeStamp1 = time.time()
        y_out = []
        for ii in range(np.shape(pure_samples)[0]):
            sigma = metadata[ii][5]
            y_out.append(np.random.normal(0, sigma, np.shape(pure_samples)[1]))

        timeStamp2 = time.time()
        dispstring = str(round(timeStamp2 - timeStamp1, 2))
        print("Generating Gaussian Noise: " + dispstring + " seconds")
        time.sleep(0.5)
        return y_out

    def driftNoise(self, noiseSamples):
        random.seed(self.seed)

        # This is a random walk on even indexes, smoothed via moving average and then lineraly added to noise.
        theShape = np.shape(noiseSamples)

        # drift multiplier parameters
        driftToNoiseRatio = self.drift  # multiplier when adding drift to noise
        if driftToNoiseRatio > 0:
            # init arrays
            xwalk = np.zeros(theShape)
            smoothWalk = []

            # random walk Wiergner model
            for ii in tqdm(range(theShape[0]), desc="Generating Drifts:     "):

                steps = random.choice((-1, 0, +1), theShape[1])

                for jj in range(theShape[1]):
                    if jj % 2 == 0:  # Random everyEvenPoints
                        xwalk[ii, jj] = xwalk[ii, jj - 1] + steps[jj]
                    else:
                        xwalk[ii, jj] = xwalk[ii, jj - 1]

                # Smoothing and normalizing
                temp = bn.move_mean(xwalk[ii, :], window=int(theShape[1] / 2), min_count=1)
                temp = temp / np.max(abs(temp))
                temp = temp * driftToNoiseRatio
                smoothWalk.append(temp)
        else:
            print("No drifts...")
            smoothWalk = np.zeros(theShape)

        return smoothWalk

    def displayFunc(self, carrier, modulation, modulated, noise, drift, real, metadata, tag):

        carrier = np.squeeze(carrier)[0 : self.num_samples_visualize]
        modulation = np.squeeze(modulation)[0 : self.num_samples_visualize]
        modulated = np.squeeze(modulated)[0 : self.num_samples_visualize]
        noise = np.squeeze(noise)[0 : self.num_samples_visualize]
        drift = np.squeeze(drift)[0 : self.num_samples_visualize]
        real = np.squeeze(real)[0 : self.num_samples_visualize]
        metadata = metadata[0 : self.num_samples_visualize]
        print("Displaying " + str(self.num_samples_visualize) + " samples")

        for ii in range(self.num_samples_visualize):
            try:
                freq = " | f =" + str(round(metadata[ii, 0], 4))
                phi = " | phi =" + str(round(metadata[ii, 1], 4))
                fmod = " | fmod =" + str(round(metadata[ii, 2], 4))
                Imod = " | Imod =" + str(round(metadata[ii, 3], 4))
                tau = " | tau =" + str(round(metadata[ii, 4], 4))
                sigma = " | sigma =" + str(round(metadata[ii, 5], 4))
                driftval = " | drift =" + str(round(metadata[ii, 6], 4))

                if (tag[ii] == [1, 0, 0]).all():
                    typefunc = "AM"

                if (tag[ii] == [0, 1, 0]).all():
                    typefunc = "FM"

                if (tag[ii] == [0, 0, 1]).all():
                    typefunc = "POLY"

                signalTitle = freq + phi + fmod + Imod + tau
                noiseTitle = sigma + driftval

                index = ii
                # maxim=np.max(real_samples[index,:])
                # minim=np.min(real_samples[index,:])
                # lims= [minim*0.9, maxim*1.1]
                fig, axs = plt.subplots(4, sharex="all")
                axs[0].plot(real[index, :], "b")
                axs[0].plot(modulated[index, :], "k", lw=2)

                axs[1].plot(modulated[index, :], "k", lw=2)

                axs[2].plot(carrier[index, :], "r")
                axs[2].plot(modulation[index, :], "c")

                axs[3].plot(noise[index, :], "k")
                axs[3].plot(drift[index, :], "orange")

                axs[0].set_title(signalTitle, fontsize=10)
                axs[1].set_title("Noiseless", fontsize=10)
                axs[2].set_title(typefunc + " function and carrier", fontsize=10)
                axs[3].set_title(noiseTitle, fontsize=10)

                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                plt.show()
            except Exception:
                traceback.print_exc()
                break

    def fastDisp(self, ydata, nB=5, labels=["xdata", "ydata", "title"]):
        ydata = np.squeeze(ydata)
        for ii in range(nB):
            try:
                plt.plot(ydata[ii])
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                plt.title(labels[2])
                plt.show()

            except:
                print("No more to display")

    def shuffle(self, *argv):
        timeStamp1 = time.time()
        rng_state = np.random.get_state()
        for target in argv:
            np.random.set_state(rng_state)
            np.random.shuffle(target)

        timeStamp2 = time.time()
        dispstring = str(round(timeStamp2 - timeStamp1, 2))
        print("Shuffling data: " + dispstring + " seconds")
