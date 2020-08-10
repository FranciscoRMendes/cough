# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:00:54 2020

@author: frmendes
"""

import scipy

from dtw import dtw
ceps = []
count = 0

for f in onlyfiles[1:]:
#    f = onlyfiles[1]
    y,fs = librosa.load(f,fs)
#    a = [1 , -0.98]; b = [1,-1];
#
#    y = scipy.signal.filtfilt(b,a,y);
    y_norm = y / np.sqrt(sum(y**2)/len(y))
    truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
    y_cough = y[truth==1]
    start,end = an.start.loc[count],an.end.loc[count]
    y_cough = y[start:end]
    plt.psd(y_cough)
    count = count + 1


librosa.display.waveplot(y1)

librosa.display.waveplot(y2)

mfcc1 = librosa.feature.mfcc(y1,sr=fs,n_mfcc=21)
mfcc2 = librosa.feature.mfcc(y2,sr=fs,n_mfcc=21)
l2_norm = lambda x, y: (x - y) ** 2
from numpy.linalg import norm

dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T,dist=l2_norm)
dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))

print ('Normalized distance between the two sounds:', )
help(dtw)

from pylab import *
imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plot(path[0], path[1], 'w')
xlim((-0.5, cost.shape[0]-0.5))
ylim((-0.5, cost.shape[1]-0.5))

h = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))

D, wp = librosa.sequence.dtw(mfcc1, mfcc2,backtrack=True)

unique_keys, indices = np.unique(wp[:,0], return_index=True)

wp = wp[np.sort(indices)]

mfcc2_  = mfcc2[:,wp[:,1]]
mfcc1_ = mfcc1[:,wp[:,0]]

x = (mfcc1_+mfcc2_)/2

y_hat = librosa.feature.inverse.mfcc_to_audio(x,sr=fs)
librosa.display.waveplot(y_hat)
y_hat = y_hat[::-1] 
sd.play(y_hat)

compare_signals(y_hat,y1)
