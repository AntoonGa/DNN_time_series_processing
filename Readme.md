"""
This repo is the raw code used in the paper "Deep neural networks to recover unknown physical parameters from oscillating time series" (https://doi.org/10.1371/journal.pone.0268439).

![ezcv logo](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0268439.g002)
Architecture used in the paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0268439
"The Encoder produces a reduced representation of the input noisy signals. The Encoder output is passed to the Regressor, which outputs the latent parameters’ prediction. The Encoder and Regressor outputs are passed to the Decoder, which produces a noiseless prediction of the inputs. "


The code generate huge amount of frequency-modulated, amplitude-modulated and pure decaying sinewave. To which gaussian noise, spurious event and drifts are added.
A custom DNN architecture is then produced as a base model.
The training task is to reproduce denoised time-series, in addition to an approximation of the latent parameters (frequency, amplitude, phase, noise level, decay rate etc).
After training, a series of experiments are run to evaluate performance and study possible applications of the models:

-1 raw performance (denoising and regression)
-2 performance vs a non-linear least-square fit (with true latent parameters as initial guesses)
-3 performance for a multi-model trained DNN
-4 application of the DNN to estimate initial guesses for a fit
-5 an interesting discussion on hyperparamters is also done

What comes out:
1- After training, the DNN outperform least-square fit (with true latent parameters as initial guesses) in almost all cases. 
2- When used as a preprocessing tool to informe least-square fit initial guesses, the outcome is always the optimal solution. This remove the need for initial guesses exploration when fitting a large number of signals.
3- The DNN is able to deal with multiple models at once, which is obviously never the case for a least-square fit. This is espcially usefull in processing unknown or varying signal-models.


This code is currently uncommented.
How to proceed with the current version of the repo:

1- Generate datasets:
Run all scripts in ./data

2- Instanciate DNN model:
Run script in ./models

3- Train models:
Run all scripts in ./trainModels
/!\ These scripts are path dependent: look at the generated-folders names in ./models and ./data and adjust for yourself.
/!\ Any decent training will take ~12/24h on a 1080GTX

4- Evaluate models and run experiments:
Run all scripts in ./experiments

"""




![ezcv logo](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0268439.g005)
DNN architecture output example on a noisy amplitude-modulated sinewave.
"Performance comparison of the specialized DNN (trained on AM-sine waves, tasked to denoise signals and recover all latent parameters of AM-sine waves) and of the partial DNN (trained on monochromatic, AM- and FM-sine waves, tasked to denoise signals and recover the carrier frequency, phase, coherence time and noise level only).

Top: Randomly selected example of noisy input AM-signal, alongside specialized- and partial-DNN denoised and latent predictions. Bottom: Individual latent parameters and signal denoising root mean squared error (RMSE), averaged over the whole AM-sinewave test set (100′000 samples) for both DNNs."