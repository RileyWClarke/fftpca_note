import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sys import argv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import bandpass_ifft, pc_thr, hrm_gfilter, pc_filt
from config import ROOTDIR

PLOTS = False
if len(argv) > 4 and argv[4] == 'plots':
    PLOTS = True

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

if argv[1].split('.')[-1] == 'npy':
    fluxes = np.load(argv[1])
elif argv[1].split('.')[-1] == 'csv':
    fluxes = pd.read_csv(argv[1], header=None).values
else:
    raise ValueError('Input file must be .npy or .csv format')

dfts = []
pspectra = []

for i, flux in enumerate(fluxes):
    lco, hco = 4, 6
    sample_rate = float(argv[3])
    Spectrum, frequency, Filtered_spectrum, Filtered_signal, Low_freq, High_freq = bandpass_ifft(flux=flux, low_cutoff=lco, high_cutoff=hco, sample=1./sample_rate, gf_sig = 1, Filter='Gaussian')

    dfts.append(Spectrum)
    pspectra.append(np.abs(Spectrum)**2)

pspec_dict = {}

for j, spec in enumerate(pspectra):
  pspec_dict["spec{0}".format(j)] = spec[1:]

pspec_df = pd.DataFrame(data=pspec_dict)
x = pspec_df.values
x = scale(x, axis=0)

pca = PCA()

pca.fit(x)

X = pca.transform(x)

pca_df = pd.DataFrame(data = X)

filt = hrm_gfilter(frequency, 0.2, 4.1, 2)

if PLOTS:
  fig, ax = plt.subplots()
  ax.boxplot([pca.components_[0], pca.components_[1], pca.components_[2], pca.components_[3],pca.components_[4]])
  plt.hlines(0, 0, 7, linestyle='--', alpha=0.5)
  plt.xlabel('Component #')
  plt.title('Distribution of Principal Component Coefficients')
  plt.xlim(0.5,5.5)
  plt.savefig(ROOTDIR + 'Figures/ztfcrm_box.png', dpi=100, bbox_inches='tight')
  plt.show()

  plt.plot(frequency[1:], pca_df.iloc[:,0])
  plt.vlines(frequency[41:][np.argmax(pca_df.iloc[40:,0])], 0, 30, color='k', ls='--')
  plt.title('PC1')
  plt.xlabel('Frequency (Hz)')
  plt.savefig(ROOTDIR + 'Figures/ztfcrm_pc1.png', dpi=100, bbox_inches='tight')
  plt.show()


peakf = frequency[41:][np.argmax(pca_df.iloc[40:,0])]
print(f'DFT Peak at {peakf:.3f} Hz')

d = {'pc1':pca.components_[0],'pc2':pca.components_[1],'pc3':pca.components_[2],'pc4':pca.components_[3],'pc5':pca.components_[4]}
coeff_df = pd.DataFrame(data=d)
coeff_df.to_csv(ROOTDIR + 'Data/ztf_crm/ztfcrm_coeffs.csv')

dft_dict = {}

for j, dft in enumerate(dfts):
  dft_dict["dft{0}".format(j)] = dft

dft_df = pd.DataFrame(data=dft_dict)

y = dft_df.values
y.real = scale(y.real, axis=0)
y.imag = scale(y.imag, axis=0)

pca2 = PCA()
pca2.fit(y.real)

Y = pca2.transform(y.real)

pca2_df = pd.DataFrame(data = Y)
pca2i = PCA()
pca2i.fit(y.imag)

Yi = pca2i.transform(y.imag)
pca2i_df = pd.DataFrame(data = Yi)

ss = pc_thr(pca2_df.T, frequency, fund=77.34, threshold=2)

fundamental = peakf  #Hz
nComp=len(fluxes) #Selects how many comps to use

real_new_pcs = pc_filt(pca2_df.T, frequency, fund=fundamental, std=0.5, nhrms=2, subset=None, plot=False)
im_new_pcs = pc_filt(pca2i_df.T, frequency, fund=fundamental, std=0.5, nhrms=2, subset=None, plot=False)

Yhat = np.dot(real_new_pcs.T[:,0:nComp], pca2.components_[0:nComp,:])
Yhati = np.dot(im_new_pcs.T[:,0:nComp], pca2i.components_[0:nComp,:])

Yhat1c = Yhat + (1j)*Yhati

processed = []

for i in range(nComp):
  processed.append(np.fft.irfft(Yhat1c[:,i]))

processed = np.array(processed)
outpath = argv[2]
np.save(f'{outpath}.npy', processed)
print(f'Saved processed light curves to {outpath}.npy')