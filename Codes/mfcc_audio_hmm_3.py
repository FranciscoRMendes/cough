# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:07:47 2020

@author: frmendes
"""
fs=16000
ceps = []
for f in onlyfiles:
    #f = onlyfiles[0]
    y,fs = librosa.load(f,fs)
    y_filter = butter_bandpass_filter(y,0.00001,3000,fs)
#    sd.play(y_filter,fs)
    y = np.asfortranarray(y_filter)
    mfcc = librosa.feature.melspectrogram(y,n_mels = 30, sr=fs)
    
#    y_inv = librosa.feature.inverse.mel_to_audio(mfcc,sr=fs)
#    sd.play(y_inv,fs)
    ceps.append(mfcc.transpose())
   

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
y_hat = librosa.feature.inverse.mel_to_audio(mfcc_simul,sr=fs)
sd.play(y_hat,fs)
plt.plot(y_hat)
plt.plot(y)
