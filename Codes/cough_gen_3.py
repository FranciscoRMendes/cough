# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:36:01 2020

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

#onlyfiles= [onlyfiles[x] for x in [2,4,7,9]] 


count = 9
fs = 44000
f = onlyfiles[count]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]
start,end = an.start.loc[count],an.end.loc[count]
y_cough = y_cough[start:end]

#y_cough1 = clean_signal(y_cough)
librosa.display.waveplot(y_cough)

mfcc_1 = librosa.feature.mfcc(y_cough,sr=fs,n_mfcc=21)


compare_signals(y_cough,y_cough1)
sd.play(y_cough)

count = 4
fs = 44000
f = onlyfiles[count]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]
start,end = an.start.loc[count],an.end.loc[count]
y_cough = y_cough[start:end]
y_cough = clean_signal(y_cough)

mfcc_2 = librosa.feature.mfcc(y_cough,sr=fs,n_mfcc=21)

compare_signals(mfcc_1[0,:],mfcc_2[0,:])

x= 0.5*mfcc_1[:,:39]+0.5*mfcc_2
#x = x/2

np.random.shuffle(x.T)

y_hat = librosa.feature.inverse.mfcc_to_audio(x,sr=fs)
librosa.display.waveplot(y_hat)
sd.play(y_hat)

compare_signals(y_hat,y_cough)

ph = hilphase(y_hat[:18000],y_cough)[0]
plt.plot(ph)

y_hat = butter_bandpass_filter(y_hat,0.00001,(0.4*fs/2),fs)

k = plt.specgram(x = y_cough_filt_1, Fs=fs,cmap='plasma',vmax=1)
k = plt.specgram(x = y_cough_filt_2, Fs=fs,cmap='plasma',vmax=1)
k = plt.specgram(x = y_hat, Fs=fs,cmap='plasma',vmax=1)


librosa.display.waveplot(y)
sd.play(y,fs)

#y_hat_1 =  y_hat[::-1]
#plt.psd(y_cough_filt_1,fs)
#plt.psd(y_cough_filt_2,fs)
plt.psd(y_hat,fs)

sd.play(y_hat,fs)
#sd.play(y_cough_filt_1,21000)
#2,4,7,9



def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")
    
    
with open filename, 'rb' as fd:
    contents = fd.read('C:/Users/frmendes/Music/Colors.mp3')