df = pd.read_csv('cSpace_file_diagnosis.csv')


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

#y, fs = librosa.load('C:/Users/frmendes/Documents/Merck Cough Accelerator/Alison_C_8_20_My_recording_3.wav')
ceps = []
for f in onlyfiles:
    f = onlyfiles[0]
    y,fs = librosa.load(f)
    data, samplerate = sf.read(f)
    y_filter = butter_bandpass_filter(y,0.00001,3000,fs)
#    sd.play(y_filter,fs)
    y = np.asfortranarray(y_filter)
    mfcc = librosa.feature.mfcc(y_filter, sr=fs, n_mfcc=20)
    ceps.append(mfcc)
#    y_hat = librosa.feature.inverse.mfcc_to_audio(mfcc = mfcc)
    plt.plot(y)
    plt.plot(y_hat)
    
    
ceps = []
for f in onlyfiles:
    #f = onlyfiles[0]
    y,fs = librosa.load(f)
    data, samplerate = sf.read(f)
    y_filter = butter_bandpass_filter(y,0.00001,3000,fs)
#    sd.play(y_filter,fs)
    y = np.asfortranarray(y_filter)
    mfcc = librosa.feature.mfcc(y_filter, sr=fs, n_mfcc=20)
    ceps.append(mfcc.transpose())

fs = 16000
ceps = []
for f in onlyfiles:
    #f = onlyfiles[0]
    y,fs = librosa.load(f,fs)
    y2 = y / np.sqrt(sum(y**2)/len(y))
    y_filter = butter_bandpass_filter(y2,0.00001,3000,fs)
    mfcc = librosa.feature.mfcc(y_filter,sr=fs,n_mfcc=50)
    y_hat = librosa.feature.inverse.mfcc_to_audio(mfcc = mfcc)
#    sd.play(y_hat,fs)    
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
mfcc_simul = patient_model.sample(300)[0].transpose()
y_hat = librosa.feature.inverse.mfcc_to_audio(mfcc = mfcc_simul,sr=fs)

#play sound
sd.play(y_hat)
librosa.display.waveplot(y_hat)
librosa.display.waveplot(y)
plt.plot(y)
patient_model.score(X)

score = []
for i in range(3,10):
    print(i)
    patient_model.fit(X = X, lengths=lengths)
    score.append(patient_model.score(X))
    


set(patient_model.predict(X))




k = plt.specgram(x = y_filter  , Fs=fs,cmap='plasma')
k = plt.specgram(x = y_hat, Fs=fs,cmap='plasma')

plt.plot(y)


plt.psd(y)
plt.psd(y_hat)
















f = onlyfiles[1]
y,fs = librosa.load(f)
y_filter = butter_bandpass_filter(y,0.00001,3000,fs)
k = plt.specgram(x = y_filter , Fs=fs,cmap='plasma')


#lengths = [len(x) for x in ceps]
X = ceps[0]
#X.shape, lengths

n_components = 5
patient_model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', tol=0.01)

patient_model.transmat_ = np.zeros((n_components,n_components)) + 1/n_components
patient_model.fit(X = X)
mfcc_simul = patient_model.sample(300)[0].transpose()
y_hat = librosa.feature.inverse.mfcc_to_audio(mfcc = mfcc_simul)













from scipy import signal
signal.welch(y_hat)
plt.psd(y_hat)
plt.psd(y)

    print(i, end=' | ')
    
    patient_cough.append(mfcc.T)
    
from playsound import playsound
playsound(y_filter)

patients_female = df[df.Sex=='Female'].File.to_list()
patients_male = df[df.Sex=='Male'].File.to_list()

patients = df.First_last.unique()
data_dir = 'wav/'
len(patients_female), len(patients_male)

patients

p = [x for x in patients_male if 'Joseph_F' in x]

patient_cough = []
for i in p:
    y,fs = librosa.load(data_dir+i, sr=16000)
    b,a = signal.butter(N=8, Wn=1000/8000, btype='lowpass')
    y = signal.filtfilt(b,a,y)
    y = np.asfortranarray(y)
    print(i, end=' | ')
    mfcc = librosa.feature.mfcc(y, sr=fs, n_mfcc=20)
    patient_cough.append(mfcc.T)

lengths = [len(x) for x in patient_cough]
X = np.concatenate(patient_cough)
X.shape, lengths




np.round(patient_model.transmat_, 3)

gn, gn_state = patient_model.sample(300)
y_gen = 
