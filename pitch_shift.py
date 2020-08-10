import librosa
import numpy as np
import os
import IPython
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
import librosa.display
import soundfile as sf

y, fs = librosa.load('C:/Users/frmendes/Documents/Merck Cough Accelerator/Alison_C_8_20_My_recording_3.wav')

y, fs = librosa.load('C:/Users/frmendes/Documents/Merck Cough Accelerator/Julie_H_8_22_My_recording_3.wav',sr=16000)


freqs, psd = signal.welch(y,fs = fs)

plt.figure(figsize=(5, 4))
plt.plot(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()


freqs, psd = signal.welch(yhat,fs = fs)

plt.figure(figsize=(5, 4))
plt.plot(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()




from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

from scipy.signal import butter, sosfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y




plt.plot(y)
plt.plot(yhat)
plt.plot(yhat+noise_hat)

k = plt.specgram(x = y  , Fs=fs,cmap='plasma')

np.histogram(noise)
plt.hist(noise,bins='auto')
plt.hist(noise_hat)
noise = y-yhat

noise_hat = librosa.effects.pitch_shift(noise,n_steps=2,sr=fs)

plt.plot(noise)
plt.plot(noise_hat)



plt.plot(y)
lowcut = 0
highcut = 1000

,sr=16000

librosa.display.waveplot(y,fs)
plt.plot(y)

librosa.display.waveplot(y,fs)

true_y = 
noise = 
librosa.effects.pitch_shift(y, sr, n_steps, bins_per_octave=12, res_type='kaiser_best', **kwargs)[source]


os.chdir('C:/Users/frmendes/Documents/Merck Cough Accelerator/')
playsound('C:/Users/frmendes/Documents/Merck Cough Accelerator/Alison_C_8_19_My_recording_.wav')
y_shifted_neg = librosa.effects.pitch_shift(y=y, sr=fs, n_steps=-4)
IPython.display.Audio(y_shifted_neg,rate=fs)

y_shifted_pos = librosa.effects.pitch_shift(y=y, sr=fs, n_steps=6)
IPython.display.Audio(y_shifted_pos,rate=fs)


b,a = signal.butter(N=8, Wn=[2000/8000] ,btype='highpass')
y_noise_high = signal.filtfilt(b,a,y) 
 
plt.plot(y)


from pyramid.arima import auto_arima
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


decomposition = seasonal_decompose(df_log) 
model = ARIMA(y, order=(1,0,0))
results = model.fit(disp=-1)
plt.plot(y)
plt.plot((y-results.fittedvalues), color='red')
plt.plot(yhat2)

error = (y-results.fittedvalues)
mu=np.mean(error)
var=np.var(error)

yhat2 = results.fittedvalues+np.random.normal(mu,var,len(y))



from playsound import playsound

yhat = results.fittedvalues
yhat= yhat.astype(np.float32)
librosa.output.write_wav('check.wav', results.fittedvalues, sr=16000) 

import scipy
scipy.io.wavfile('check.wav',yhat,sr=16000)
scipy.io.wavfile.write('check.wav', 16000, yhat)
playsound()