# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:43:32 2020

@author: frmendes
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:59:14 2020

@author: frmendes
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:49:42 2020

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

onlyfiles= [onlyfiles[x] for x in [2,4,7,9]] 
import pandas as pd
an = pd.read_excel('../annotation.xlsx')
#0 Frequency
#1 MFCC
#2 Spectrogram
mode = 0
mode_print = ['Frequency','MFCC','Spectrogram']
print(mode_print[mode])


fs = 44000
top = 0.5 #this is in seconds, how much of the signal to start with
top = int(top*fs) #convert it to samples by time * sample rate

ceps = []
count = 0
for f in onlyfiles:
#    f = onlyfiles[1]
    y,fs = librosa.load(f,fs)
    y_norm = y / np.sqrt(sum(y**2)/len(y))
    truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
    y_cough = y_norm[truth==1]
#    if top!=0:
#        y_cough = y_cough[:top]
    
    start,end = an.start.loc[count],an.end.loc[count]
    
    y_cough = y_cough[start:end]
    y_cough_filt = clean_signal(y_cough)
    
    print(len(y_cough_filt))
    if mode == 0:
        mfcc = librosa.core.stft(y_cough_filt,n_fft = 100)
        mfcc = np.abs(mfcc)
    if mode == 1:
        mfcc = librosa.feature.mfcc(y_cough_filt,sr=fs,n_mfcc=21,dct_type=1\
                                    ,hop_length=int(0.010*fs), n_fft=int(0.025*fs))
    if mode == 2:
        mfcc = librosa.feature.melspectrogram(y_cough_filt,sr=fs,n_mels=21\
                                              ,hop_length=int(0.010*fs), n_fft=int(0.025*fs))
    ceps.append(mfcc.transpose())
    count = count+1


#fs = 44000
f = onlyfiles[0]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y_norm[truth==1]
start = 1500
end = 24500


y_cough_ref = clean_signal(y_cough[start:end])
plt.plot(y_cough[start:end])
plt.plot(y_cough_ref)

sd.play(y_cough_ref,fs)






#fit model
lengths = [len(x) for x in ceps]
X = np.concatenate(ceps)
X.shape, lengths


n_components = 3
patient_model = hmm.GaussianHMM(n_components=n_components, covariance_type='tied', n_iter=1000)
patient_model.transmat_ = np.zeros((n_components,n_components)) + 1/n_components
patient_model.fit(X = X, lengths=lengths)



#simulate model
#mfcc_simul = patient_model.sample(int(np.mean(lengths)))

mfcc_simul,states = patient_model.sample(int(np.mean(lengths)))
mfcc_simul = mfcc_simul.transpose()
print(states)
#y_hat = mfcc_simul


#simulate

if mode == 0:
    y_hat = librosa.core.istft(mfcc_simul)
if mode == 1:
    y_hat = librosa.feature.inverse.mfcc_to_audio(mfcc_simul,sr=fs)
if mode == 2:
    y_hat = librosa.feature.inverse.mel_to_audio(mfcc_simul,sr=fs)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

y_hat = moving_average(y_hat,3)




#compare visually
librosa.display.waveplot(y_hat);
librosa.display.waveplot(y_cough_ref)

#compare spectrogram
k = plt.specgram(x = y_cough_ref, Fs=fs,cmap='plasma',vmax=1)
k = plt.specgram(x = y_hat, Fs=fs,cmap='plasma',vmax=1)

y_hat = y_hat[::-1]

y_hat = librosa.effects.preemphasis(y_hat)


sd.play(y_hat,44000)  
sd.play(y_cough_ref,fs)

#HZ content
plt.psd(y_cough_ref,fs)
plt.psd(y_hat,fs)


y_hat = butter_bandpass_filter(y_hat[:int((0.1*44000))],0.00001,17600,fs)










#state check
f = onlyfiles[1]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y_norm[truth==1]
if top!=0:
    y_cough = y_cough[:top]
y_cough_filt = butter_bandpass_filter(y_cough,0.00001,3000,fs)

if mode == 0:
    mfcc = librosa.core.stft(y_cough_filt,n_fft = 100)
    mfcc = np.abs(mfcc)
if mode == 1:
    mfcc = librosa.feature.mfcc(y_cough_filt,sr=fs,n_mfcc=21)
if mode == 2:
    mfcc = librosa.feature.melspectrogram(y_cough_filt,sr=fs,n_mels=21)
    
    
states = patient_model.predict(mfcc.transpose())
plt.plot(states)
plt.plot()
