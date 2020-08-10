# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:08:04 2020

@author: frmendes
"""

fs = 16000
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
    y3 = y3[:8000]
#    librosa.display.waveplot(y3,fs)
    y_filter = butter_bandpass_filter(y3,0.00001,3000,fs)
#    sd.play(y_filter,fs)
    mfcc = librosa.feature.mfcc(y_filter,sr=fs,n_mfcc=21)
    
    ceps.append(mfcc.transpose())
    
    
#fit model
lengths = [len(x) for x in ceps]
X = np.concatenate(ceps)

X.shape, lengths

n_components = 3
patient_model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=100)
patient_model.transmat_ = np.zeros((n_components,n_components)) + 1/n_components
patient_model.fit(X = X, lengths=lengths)

#simulate model
mfcc_simul = patient_model.sample(int(np.mean(lengths)))[0].transpose()
mfcc_simul = patient_model.sample(20)[0].transpose()
y_hat = librosa.feature.inverse.mfcc_to_audio(mfcc_simul)
y_hat = y_hat[::-1]
sd.play(y_hat,fs)  

plt.plot(y_hat)
plt.plot(y_filter)


