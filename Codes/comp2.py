# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:52:04 2020

@author: frmendes
"""




i = 1
f = onlyfiles[i]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]

start,end = an.start.loc[i],an.end.loc[i]
y_cough = y_cough[start:end]

plt.figure()
librosa.display.waveplot(y_cough)
plt.figure()
plt.psd(y_cough)
plt.figure()
plt.phase_spectrum(y_cough)


i = 1
f = onlyfiles[i]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]

#        y_cough = y_cough[:top]

start,end = an.start.loc[i],an.end.loc[i]
y_cough = y_cough[start:end]

plt.figure()
librosa.display.waveplot(y_cough)
plt.figure()
plt.psd(y_cough)
plt.figure()
plt.phase_spectrum(y_cough)

i = 1
f = onlyfiles[i]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]

#        y_cough = y_cough[:top]

start,end = an.start.loc[i],an.end.loc[i]
y_cough = y_cough[start:end]

plt.figure()
librosa.display.waveplot(y_cough)
plt.figure()
plt.psd(y_cough)
plt.figure()
plt.phase_spectrum(y_cough)


####-----------------------------------------------------------------------------
i = 1
f = onlyfiles[i]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]
start,end = an.start.loc[i],an.end.loc[i]
y_cough = y_cough[start:end]

librosa.display.waveplot(y_cough)
plt.figure()
plt.psd(y_cough)
plt.figure()
plt.phase_spectrum(y_cough)



####-----------------------------------------------------------------------------
i = 2
f = onlyfiles[i]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]

#        y_cough = y_cough[:top]

start,end = an.start.loc[i],an.end.loc[i]
y_cough = y_cough[start:end]

plt.figure()
librosa.display.waveplot(y_cough)
plt.figure()
plt.psd(y_cough)
plt.figure()
plt.phase_spectrum(y_cough)
sd.play(y1,fs)

y1 = y_cough



####-----------------------------------------------------------------------------
i = 4
f = onlyfiles[i]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]

#        y_cough = y_cough[:top]

start,end = an.start.loc[i],an.end.loc[i]
y_cough = y_cough[start:end]

plt.figure()
librosa.display.waveplot(y_cough)
plt.figure()
plt.psd(y_cough)
plt.figure()
plt.phase_spectrum(y_cough)

y2 = y_cough

####-----------------------------------------------------------------------------

i = 1
f = onlyfiles[i]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]

#        y_cough = y_cough[:top]

start,end = an.start.loc[i],an.end.loc[i]
y_cough = y_cough[start:end]

plt.figure()
librosa.display.waveplot(y_cough)
plt.figure()
plt.psd(y_cough)
plt.figure()
plt.phase_spectrum(y_cough)

i = 1
f = onlyfiles[i]
y,fs = librosa.load(f,fs)
y_norm = y / np.sqrt(sum(y**2)/len(y))
truth = smoother(y_norm, lf=1600, mode='var', var_thres=0.075)
y_cough = y[truth==1]

#        y_cough = y_cough[:top]

start,end = an.start.loc[i],an.end.loc[i]
y_cough = y_cough[start:end]

plt.figure()
librosa.display.waveplot(y_cough)
plt.figure()
plt.psd(y_cough)
plt.figure()
plt.phase_spectrum(y_cough)
