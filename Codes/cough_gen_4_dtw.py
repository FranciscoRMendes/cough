# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:01:20 2020

@author: frmendes
"""

def smoother(sample, lf=1600, 
             var_thres=0.00003, ss_thres=0.00003,
             mode='var'):
    assert mode in ['ss', 'var']
    
    truth = sample.copy()
    len_sample = len(truth)
    l = int((len(truth))/lf + 1)
    
    
    if mode=='var':
        for i in range(l):
            if i*lf >= len_sample:
                break
                
            variance = np.var(truth[i*lf : i*lf + lf])
            if variance > var_thres:
                truth[i*lf : min(i*lf + lf, len_sample)] = 1
            else:
                truth[i*lf : min(i*lf + lf, len_sample)] = 0
    
    if mode=='ss':
        for i in range(l):
            if i*lf >= len_sample:
                break
                
            sum_squares = sum(abs(truth[i*lf:i*lf+lf]) ** 2)
            if sum_squares > ss_thres * lf:
                truth[i*lf : min(i*lf + lf, len_sample)] = 1
            else:
                truth[i*lf : min(i*lf + lf, len_sample)] = 0
    print(i*lf + lf, end=' | ')
    return truth

import pandas as pd
import scipy
import librosa
import sounddevice as sd
#import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir('C:/Users/frmendes/Documents/Merck Cough Accelerator/dat')
from os import listdir
from hmmlearn import hmm
from os.path import isfile, join
onlyfiles = [f for f in listdir() if isfile(join(f))]

an = pd.read_excel('../annotation.xlsx')
fs = 44000

#from librosa import display
#from scipy.signal import butter, sosfilt, sosfreqz


i = 2
f = onlyfiles[i]
y,fs = librosa.load(f,fs)

#--------------------------------------------------------------#
y = scipy.signal.filtfilt([1,-1],[1 , -0.98],y);
y_norm = y / np.sqrt(sum(y**2)/len(y))
y_cough = y[smoother(y_norm, lf=1600, mode='var', var_thres=0.075)==1]
y_cough = y_cough[an.start.loc[i]:an.end.loc[i]]
#--------------------------------------------------------------#

librosa.display.waveplot(y_cough)
#plt.figure()
#plt.psd(y_cough)
#plt.figure()
#plt.phase_spectrum(y_cough)

gold_y = y_cough
gold_mfcc = librosa.feature.mfcc(gold_y,sr=fs,n_mfcc=21)

def warp_to_gold(gold_mfcc,mfcc):
    D, wp = librosa.sequence.dtw(gold_mfcc, mfcc)
    unique_keys, indices = np.unique(wp[:,0], return_index=True)
    wp = wp[np.sort(indices)[::-1]]
    mfcc  = mfcc[:,wp[:,1]]
    return(mfcc)
    
out = warp_to_gold(gold_mfcc,mfcc)




ceps = []
count = 0

for f in onlyfiles[1:]:
#   f = onlyfiles[0]
    y,fs = librosa.load(f,fs)
            
    #--------------------------------------------------------------#
    y = scipy.signal.filtfilt([1,-1],[1 , -0.98],y);
    y_norm = y / np.sqrt(sum(y**2)/len(y))
    y_cough = y[smoother(y_norm, lf=1600, mode='var', var_thres=0.075)==1]
    y_cough = y_cough[an.start.loc[count]:an.end.loc[count]]
    #--------------------------------------------------------------#
    librosa.display.waveplot(y_cough)
    mfcc = librosa.feature.mfcc(y_cough,sr=fs,n_mfcc=21)
    mfcc = warp_to_gold(gold_mfcc,mfcc)
    ceps.append(mfcc.transpose())
    count = count+1
    

output_mfcc = sum(ceps)/9

 y_hat = librosa.feature.inverse.mfcc_to_audio(output_mfcc.transpose(),sr=fs)
 librosa.display.waveplot(y_hat,fs)
 
sd.play(y_hat,fs)

#fit model
lengths = [len(x) for x in ceps]
X = np.concatenate(ceps)
X.shape, lengths

#
#n_components = 3
#patient_model = hmm.GaussianHMM(n_components=n_components, covariance_type='tied', n_iter=1000)
#patient_model.transmat_ = np.zeros((n_components,n_components)) + 1/n_components
#patient_model.fit(X = X, lengths=lengths)
#
#
#mfcc_simul = patient_model.sample(int(np.mean(lengths)))[0].transpose()
#mfcc_simul = patient_model.sample(1000)[0].transpose()
#y_hat = librosa.feature.inverse.mfcc_to_audio(mfcc_simul,sr=fs)
#
#sd.play(y_hat,fs)
##    
##sd.play(y_cough,fs)
##
#compare_signals(y_hat,y_cough)
##
##k = plt.specgram(y_hat,Fs=fs)
##k = plt.specgram(y_cough,Fs=fs)
#
#
#
# import numpy as np
#
# chroma = librosa.feature.chroma_cqt(y=y_cough, sr=fs)
# # Use time-delay embedding to reduce noise
# chroma_stack = librosa.feature.stack_memory(chroma, n_steps=3)
# # Build recurrence, suppress self-loops within 1 second
# rec = librosa.segment.recurrence_matrix(chroma_stack, width=2,
#                             mode='affinity',
#                             metric='cosine')
# 
# 
## using infinite cost for gaps enforces strict path continuation
#L_score, L_path = librosa.sequence.rqa(rec, np.inf, np.inf,
#                             knight_moves=False)
#
# 
#plt.figure(figsize=(10, 4))
#plt.subplot(1,2,1)
#librosa.display.specshow(rec, x_axis='frames', y_axis='frames')
#plt.title('Recurrence matrix')
#plt.colorbar()
#plt.subplot(1,2,2)
#librosa.display.specshow(L_score, x_axis='frames', y_axis='frames')
#plt.title('Alignment score matrix')
#plt.colorbar()
#plt.plot(L_path[:, 1], L_path[:, 0], label='Optimal path', color='c')
#plt.legend()
#plt.show()
#
#
#
#import matplotlib.pyplot as plt
#
#R = librosa.segment.recurrence_matrix(output_mfcc.transpose(), width=1,
#                             mode='affinity',
#                             metric='cosine')
#
#R_aff = librosa.segment.recurrence_matrix(output_mfcc.transpose(), mode='affinity')
# 
# 
#plt.figure(figsize=(8, 4))
#plt.subplot(1, 2, 1)
#librosa.display.specshow(R, x_axis='time', y_axis='time')
#plt.title('Binary recurrence (symmetric)')
#plt.subplot(1, 2, 2)
#librosa.display.specshow(R_aff, x_axis='time', y_axis='time',cmap='magma_r')
#plt.title('Affinity recurrence')
#plt.tight_layout()
#plt.show()
#
