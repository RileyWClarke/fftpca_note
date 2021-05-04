import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
from fftpca_config import *
import templates
import match_template1d

#temps, amps, thalfs = templates.make_temps(1, 10, 0.1, 7, n=10, scale='log')

def plot(nflares, temps, thalfs):

    fig, axes = plt.subplots(nflares,3,figsize=(10,30))

    for i, flare in enumerate(flare_ts[:nflares]):
        axes[i,0].scatter(time1, flux1rm, color='k', marker='.', s=5)
        axes[i,0].vlines(flare, -25, 25, color='k', linestyle='--', linewidth=0.5, alpha=0.25, label=str(flare))
        axes[i,0].set_xlim(flare-0.05,flare+0.05)
        axes[i,0].get_xaxis().set_ticks([])
        #axes[i,0].set_ylabel('Rel. Flux')
        axes[i,0].legend(fontsize=6)
        if i == 0:
            print('plotting 1st flare...')
        elif i == 1:
            print('plotting 2nd flare...')
        elif i == 2:
            print('plotting 3rd flare...')
        else:
            print('plotting {}th flare...'.format(i+1))
            
        peaks = []
        flare_ind = np.where( np.abs(time1 - flare) == np.abs(time1 - flare).min() )[0][0]
        
        for k in range(temps.shape[1]):
            conv = match_template1d.match_template1d(flux1rm, temps[0,k]/temps[0,k].max())
            axes[i,1].plot(time1, conv, linewidth=0.5, label='FWHM = {0:.2f}'.format(thalfs[k]))
            axes[i,1].set_xlim(flare-0.05,flare+0.05)
            axes[i,1].get_xaxis().set_ticks([])
            axes[i,1].legend(fontsize=3, loc=1)
            peaks.append(conv[flare_ind-10:flare_ind+10].max())
                
        axes[i,1].axvline(flare, color='k', linestyle='--', linewidth=0.5, alpha=0.25)

        peaks = np.array(peaks)
        maxpeak = np.where(peaks == peaks.max())[0][0]
        temp_x = np.arange(flare - (100*np.diff(time1)[0]), flare + (101*np.diff(time1)[0]), np.diff(time1)[0])
        axes[i,2].plot(time1,flux1rm, '.', c='k', markersize=1, label='Data')
        axes[i,2].plot(temp_x, temps[0,maxpeak]*100, linewidth=1, alpha=0.5, label='FWHM = {0:.2f} template'.format(thalfs[maxpeak]))
        axes[i,2].set_xlim(flare-0.01, flare+0.01)
        axes[i,2].get_xaxis().set_ticks([])
        axes[i,2].legend(fontsize=3)

    plt.savefig('Figures/diag_plot.png', dpi=500, bbox_inches='tight')
    plt.show()


