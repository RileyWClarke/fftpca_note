import numpy as np
from matplotlib import pyplot as plt
import sys
from fftpca_config import *
import convplots
import templates

###Flare Template Fitting & FFD Comparison

REGEN = True

amin = 1
amax = 10
tmin = 0.1
tmax = 10
n=10

if REGEN:

    temps, amps, thalfs = templates.make_temps(amin, amax, tmin, tmax, n=n, scale='log')
    
else:
    temps = np.load('temps.npy')
    amps = (amin + amax) - np.log2(np.linspace(2**amin,2**amax,n))[::-1]
    thalfs =  (tmin + tmax) - np.log2(np.linspace(2**tmin,2**tmax,n))[::-1]
    
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
    
#conv = np.convolve(flux1norm, temps[2,0])

#peak = np.where(conv == np.max(conv))[0][0]

normtemps = np.array([temp/temp.sum() for temp in temps.reshape(n**2,201)])

for i in range(5):
    ax1.plot(normtemps[i],'.')
    ax1.plot(normtemps[i], label='FWHM = {0:.2f}'.format(thalfs[i]))
    ax1.set_title('Sample Templates, A = {}'.format(amps[0]))
    ax1.legend()
plt.savefig('templates.png', dpi=200, bbox_inches='tight')

convplots.plot(15, normtemps.reshape((n,n,201)), thalfs)
sys.exit()

plt.figure()
plt.plot(flux1norm[peak - int(temps.shape[2]+1) : peak])
plt.vlines(100,-10,70, color='k')
plt.plot(temps[2,-1]*50)
plt.plot(temps[2,-1]*50, '.')
plt.show()



