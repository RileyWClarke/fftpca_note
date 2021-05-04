import numpy as np
from matplotlib import pyplot as plt
from fftpca_config import *

def flare_temp(t, A, thalf, tpeak):

    tnew = (t-tpeak)/thalf

    tnew1 = tnew[tnew < 0]

    tnew2 = tnew[tnew >= 0]

    Frise = A * (1 + (1.941 * tnew1) - (0.175 * tnew1**2) - (2.246 * tnew1**2) - (1.125 * tnew1**4))

    Fdecay = A * (0.689 * np.exp(-1.6 * tnew2) + 0.3030 * np.exp(-0.2783 * tnew2))

    f = np.hstack([Frise, Fdecay])

    f[f < 0] = 0
    
    return f

time = np.arange(-10,10.1,0.1)

def make_temps(amin, amax, tmin, tmax, n=10, scale='log'):
    
    temps = np.zeros((n, n, len(time)))
    
    lin_arr_a = np.linspace(amin,amax,n)
    log_arr_a =  (amin + amax) - np.log2(np.linspace(2**amin,2**amax,n))[::-1]
    lin_arr_t = np.linspace(tmin,tmax,n)
    log_arr_t =  (tmin + tmax) - np.log2(np.linspace(2**tmin,2**tmax,n))[::-1]
    
    if scale == 'linear':
        amps = lin_arr_a
        thalves = lin_arr_t
        for i, a in enumerate(lin_arr_a):
            for j, thalf in enumerate(lin_arr_t): 
                temps[i,j] = flare_temp(time, a, thalf, 0)

    if scale == 'log':
        amps = log_arr_a
        thalves = log_arr_t
        for i, a in enumerate(log_arr_a):
            for j, thalf in enumerate(log_arr_t): 
                temps[i,j] = flare_temp(time, a, thalf, 0)
            
    np.save('temps.npy',temps)
    return temps, amps, thalves



