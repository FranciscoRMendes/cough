# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 06:47:24 2020

@author: frmendes
"""

def phase_shift(signal):
    ## Fourier transform of real valued signal
    signalFFT = np.fft.rfft(signal)
    
    ## Get Power Spectral Density
    signalPSD = np.abs(signalFFT) ** 2
    signalPSD /= len(signalFFT)**2
    
    ## Get Phase
    signalPhase = np.angle(signalFFT)
    
    ## Phase Shift the signal +90 degrees
    newSignalFFT = signalFFT * cmath.rect( 1., np.pi/2 )
    
    ## Reverse Fourier transform
    newSignal = np.fft.irfft(newSignalFFT)

    return(newSignal)