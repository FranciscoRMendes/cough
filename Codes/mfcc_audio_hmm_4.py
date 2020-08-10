# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:43:03 2020

@author: frmendes
"""

import librosa
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir('C:/Users/frmendes/Documents/Merck Cough Accelerator/dat')
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir() if isfile(join(f))]
from hmmlearn import hmm
import signal
from librosa import display
from scipy.signal import butter, sosfilt, sosfreqz


fs = 8000
ceps = []
for f in onlyfiles:
    #f = onlyfiles[9]
    y,fs = librosa.load(f,fs)
#   librosa.display.waveplot(y,fs)
    y2 = y / np.sqrt(sum(y**2)/len(y))
#   librosa.display.waveplot(y2,fs)
#   plt.plot(y2)
    y_filter = butter_bandpass_filter(y2,0.00001,3000,fs)
#   librosa.display.waveplot(y_filter,fs)
#    sd.play(y_filter,fs)
    mfcc = librosa.core.stft(y_filter,n_fft = 100)
    mfcc = np.abs(mfcc)
    ceps.append(mfcc.transpose())
    
    
    
#
#fs=8000
#ceps = []
#for f in onlyfiles:
#    #f = onlyfiles[0]
#    y,fs = librosa.load(f,fs)
#    y_filter = butter_bandpass_filter(y,0.00001,3000,fs)
##    sd.play(y_filter,fs)
#    y = np.asfortranarray(y_filter)
#    mfcc = librosa.core.stft(y,n_fft = 100)
#    mfcc = np.abs(mfcc)
##    y_inv = librosa.core.istft(mfcc)
##    sd.play(y_inv,fs)
#    ceps.append(mfcc.transpose())
#   

#fit model
lengths = [len(x) for x in ceps]
X = np.concatenate(ceps)

X.shape, lengths

n_components = 5
patient_model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=1000)
patient_model.transmat_ = np.zeros((n_components,n_components)) + 1/n_components
patient_model.fit(X = X, lengths=lengths)

#simulate model
mfcc_simul = patient_model.sample(int(np.mean(lengths)))[0].transpose()
y_hat = librosa.core.istft(mfcc_simul)
#sd.play(y_hat,fs)
#y_norm = y_hat - y_hat.min()/y_hat.max()-y_hat.min()
plt.plot(y)
plt.plot(y_hat)

sd.play(y_hat,fs)
librosa.display.waveplot(y_hat)



fs = 8000
ceps = []

for f in onlyfiles:
#    f = onlyfiles[1]
    y,fs = librosa.load(f,fs)
#    librosa.display.waveplot(y,fs)
    y2 = y / np.sqrt(sum(y**2)/len(y))
#    librosa.display.waveplot(y2,fs)
    truth = smoother(y2, lf=1600, mode='var', var_thres=0.075)
#    plt.plot(y2);plt.plot(truth)
    y3 = y2[truth==1]
#    librosa.display.waveplot(y3,fs)
    y_filter = butter_bandpass_filter(y3,0.00001,3000,fs)
    y_filter = y_filter[:5000]
#    sd.play(y_filter,fs)
    mfcc = librosa.core.stft(y_filter,n_fft = 100)
    mfcc = np.abs(mfcc)
    ceps.append(mfcc.transpose())
    
    
#fit model
lengths = [len(x) for x in ceps]
X = np.concatenate(ceps)

X.shape, lengths

n_components = 5
patient_model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=100)
patient_model.transmat_ = np.zeros((n_components,n_components)) + 1/n_components
patient_model.fit(X = X, lengths=lengths)

#simulate model
mfcc_simul = patient_model.sample(int(np.mean(lengths)))[0].transpose()
y_hat = librosa.core.istft(mfcc_simul)
sd.play(y_hat,fs)  
sd.play(y_filter,fs)

plt.plot(y_hat)
plt.plot(y_filter)


#ref sound
f = onlyfiles[4]
y,fs = librosa.load(f,fs)
y2 = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y2, lf=1600, mode='var', var_thres=0.075)
y3 = y2[truth==1]
y_filter = butter_bandpass_filter(y3,0.00001,3000,fs)


y_hat = butter_bandpass_filter(y_hat,0.00001,3000,fs)
y_hat_1 = moving_average(y_hat,n=100)
librosa.display.waveplot(y_filter)
librosa.display.waveplot(y_hat)


sd.play(y,fs)
sd.play(y_hat,fs)  

k = plt.specgram(x = y_filter  , Fs=fs,cmap='plasma',vmax=1)
k = plt.specgram(x = y_hat, Fs=fs,cmap='plasma',vmax=1)

from scipy import *
smooth(y_hat,window_len=100)

plt.psd(y_filter,fs)
plt.psd(y_hat,fs)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


moving_average(arange(0,10),n=3)
a = arange(10)
ret = np.cumsum(a, dtype=float)






