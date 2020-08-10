# -*- coding: utf-81 -*-
"""
Created on Sun Jun 14 11:41:50 2020

@author: frmendes
"""

import os
os.chdir('C:/Users/frmendes/Documents/Merck Cough Accelerator/dat')
from os import listdir
from hmmlearn import hmm
from os.path import isfile, join
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
onlyfiles = [f for f in listdir() if isfile(join(f))]
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
fs = 44000
an = pd.read_excel('../annotation.xlsx')

def hmm_predict(c,n=3):
    lengths = [len(x) for x in c]
    X = np.concatenate(c)
    X.shape, lengths
    n_components = 3
    patient_model = hmm.GaussianHMM(n_components=n_components, covariance_type='tied', n_iter=1000)
    patient_model.transmat_ = np.zeros((n_components,n_components)) + 1/n_components
    patient_model.fit(X = X, lengths=lengths)
    sim = patient_model.sample(int(np.mean(lengths)))[0].transpose()
    yhat = librosa.feature.inverse.mfcc_to_audio(sim,sr=fs)
    return(yhat,sim)

def mean_predict(c):
    output = sum(c)/9
    yhat = librosa.feature.inverse.mfcc_to_audio(output,sr=fs)
    return(yhat,output)
    



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



def segment_cough(i):
    f = onlyfiles[i]
    y = librosa.load(f,fs)[0]
    y_norm = y / np.sqrt(sum(y**2)/len(y))
    y_cough = y[smoother(y_norm, lf=1600, mode='var', var_thres=0.075)==1]
    y_cough = y_cough[an.start.loc[i]:an.end.loc[i]]
    
    phase_1, phase_2, phase_3 = y_cough[:an.a.loc[i]],y_cough[an.a.loc[i]:an.b.loc[i]],y_cough[an.b.loc[i]:]
#   plt.figure()
#   plt.plot(y_cough)
#    
#   plt.figure()
#   plt.plot(phase_1)
#   plt.plot(phase_2)
#   plt.plot(phase_3)
    
    return(phase_1,phase_2,phase_3)

#Example
phase_1,phase_2,phase_3 = segment_cough(2)

#----------------------------------------------------------------#
# extract "god standard mfccs" phase wise for all coughs
#----------------------------------------------------------------#

y1,y2,y3 = segment_cough(2)
gold_m1,gold_m2,gold_m3 = librosa.feature.mfcc(y1,sr=fs,n_mfcc=21),librosa.feature.mfcc(y2,sr=fs,n_mfcc=21),librosa.feature.mfcc(y3,sr=fs,n_mfcc=21)

    


#----------------------------------------------------------------#
# extract mfccs phase wise for all coughs
#----------------------------------------------------------------#
c1=[]
c2=[]
c3=[]
for i in range(0,9):
    
    y1,y2,y3 = segment_cough(i)
    m1,m2,m3 = librosa.feature.mfcc(y1,sr=fs,n_mfcc=21),librosa.feature.mfcc(y2,sr=fs,n_mfcc=21),librosa.feature.mfcc(y3,sr=fs,n_mfcc=21)
    
    
    m1 = warp_to_gold(gold_m1,m1)
    m2 = warp_to_gold(gold_m2,m2)
    m3 = warp_to_gold(gold_m3,m3)
    
    c1.append(m1.transpose())
    c2.append(m2.transpose())
    c3.append(m3.transpose())
#--------------------------------------------------------------#


#librosa.display.waveplot(y,fs)
#plt.plot(y[500:23500])

#----------------------------------------------------------------#
# simulate phases by simple mean across all ceps
#----------------------------------------------------------------#
    
output_1 = sum(c1)/9
output_2 = sum(c2)/9
output_3 = sum(c3)/9

#np.random.shuffle(output_1),np.random.shuffle(output_2),np.random.shuffle(output_3)


y1_hat_m = librosa.feature.inverse.mfcc_to_audio(output_1.transpose(),sr=fs)
y2_hat_m = librosa.feature.inverse.mfcc_to_audio(output_2.transpose(),sr=fs)
y3_hat_m = librosa.feature.inverse.mfcc_to_audio(output_3.transpose(),sr=fs)

y_hat_m = np.hstack(( y1_hat_m,y2_hat_m,y3_hat_m )).ravel()
librosa.display.waveplot(y2_hat_m,fs)
librosa.display.waveplot(y_hat_m,fs)

plt.plot(c2[0][:,1])
plt.plot(sim.transpose()[:,1])

##k = 
k = plt.specgram(y3_hat_m,Fs=fs)
k = plt.specgram(y3_hat,Fs=fs)
sim_2 = sim.copy()
sim_2 = sim_2.transpose()
sim_2[:,0] = output_2[:,0]

plt.plot(sim_2[:,0])
plt.plot(output_2[:,0])
sd.play(y2_hat_m,fs) 

librosa.display.waveplot(y2_hat,fs)
#--------------------------------------------------------------#

#----------------------------------------------------------------#
#simulate phases using markov model
#----------------------------------------------------------------#
lengths = [len(x) for x in c1]
X = np.concatenate(c1)
X.shape, lengths


    
y1_hat,m = build_predict(c3)






y_hat_1 = np.hstack(( y1_hat,y2_hat,y3_hat )).ravel()
librosa.display.waveplot(y_hat_1,fs)
librosa.display.waveplot(y_hat,fs)
sd.play(y_hat_1,fs)

k = plt.psd(y_hat_1)
k = plt.psd(y_hat)

compare_signals(y_hat,y_hat_1)
phase_1, phase_2, phase_3 = y_cough[:an.a.loc[i]],y_cough[an.a.loc[i]:an.b.loc[i]],y_cough[an.b.loc[i]:]




