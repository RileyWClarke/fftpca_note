import numpy as np
from matplotlib import pyplot as plt
import astropy.io.fits as fits


def invgaussian(a ,m, s, x):
    '''
    Compute an inverted Gaussian function.
    Parameters
    ----------
    a : float
        Amplitude of the Gaussian.
    m : float
        Mean of the Gaussian.
    s : float
        Standard deviation of the Gaussian.
    x : float
        Input value for the Gaussian function.
    Returns
    -------
    g : float
        Output value of the inverted Gaussian function.
    '''
    g = -a * np.exp(-(m-x)**2 / s**2) + 1
    return g

def urls_to_dict(urls, campaign):
    '''
    Convert a list of KIC FITS file URLs into a dictionary of time and flux data for a specified Kepler campaign.
    Parameters
    ----------
    urls : list
        List of FITS file URLs.
    campaign : str
        Campaign identifier to filter the URLs.
    Returns
    -------
    d : dict
        Dictionary containing time and flux data from the FITS files.
    '''
    d = {}
    for j, url in enumerate(urls):
        print()
        if url[51] == campaign:
            try:
                with fits.open(url, mode="readonly") as hdulist:
                    d["k2bjds{0}".format(j)] = hdulist[1].data['TIME'] 
                    d["pdcsap_flux{0}".format(j)] = hdulist[1].data['PDCSAP_FLUX']
                    d["pdcsap_flux_err{0}".format(j)] = hdulist[1].data['PDCSAP_FLUX_ERR']

            except: 
                print('Loop '+str(j)+' : File not found')

    return d

def lin_interp_nans(flux):
    '''
    Linearly interpolate over NaN values in a flux array.
    Parameters
    ----------
    flux : array-like
        1D array of flux values with NaNs to be interpolated over.
    Returns
    -------
    newflux : array-like
        1D array of flux values with NaNs replaced by linear interpolation.
    '''
    newflux = np.copy(flux)

    #if series tip/tail is nan, replace with first/last real value:

    if np.isnan(newflux[0]) == True:
      for i,j in enumerate(newflux):
        if np.isnan(j) == False:
          newflux[:i] = np.full(i, j)
          break
        else:
          pass

    if np.isnan(newflux[-1]) == True:
      for i,j in enumerate(newflux[::-1]):
        if np.isnan(j) == False:
          newflux[-i:] = np.full(i, j)
          break
        else:
          pass

    #patching:
    bools = np.isnan(newflux)
    edges = np.diff(bools)
    edgeindx = np.where(~(edges == 0))[0]+1
    
    for i, edge in enumerate(edgeindx):
    
      if i%2 == 0:
      
        a, b = np.polyfit( (edge-1, edgeindx[i+1]),
                           (flux[edge-1], flux[edgeindx[i+1]]), deg = 1)
        
        patch = a * np.arange(edge,edgeindx[i+1]) + b 
        
      else:
        continue

      newflux[edge:edgeindx[i+1]] = patch
      
    return newflux

def bandpass_ifft(flux, low_cutoff, high_cutoff, sample=1, inv_box=False, gf_sig = 1, Filter='box', Plot=''):
    """Bandpass filtering on a real signal using inverse FFT
    
    Inputs
    =======
    
    X: 1-D numpy array of floats, the real time domain signal (time series) to be filtered
    Low_cutoff: float, frequency components below this frequency will not pass the filter (physical frequency in unit of Hz)
    High_cutoff: float, frequency components above this frequency will not pass the filter (physical frequency in unit of Hz)
    sample: float, the sampling frequency of the signal (physical frequency in unit of Hz)    
    inv_box: If using box filter, setting inv=True filters out frequencies outside the box
    Filter: Default filter is box, can choose 'Gaussian' also
    
    Notes
    =====
    1. The input signal must be real, not imaginary nor complex
    2. The Filtered_signal will have only half of original amplitude. Use abs() to restore. 
    3. In Numpy/Scipy, the frequencies goes from 0 to F_sample/2 and then from negative F_sample to 0. 
    
    """        
    #perform fft
    spectrum = np.fft.rfft(flux) 
    freq = np.fft.rfftfreq(len(flux), sample)
    
    #calculate the index of the cut off points
    lc = np.abs(freq) < low_cutoff
    hc = np.abs(freq) > high_cutoff
    between = ~(lc + hc)
    
    ps = np.abs(spectrum)**2
    if ('PS' in Plot) or ('All' in Plot):
      plt.plot(freq, ps)
      plt.title("power spectrum")
      plt.xlabel('Frequency (1/day)')
      plt.ylabel('Power Spectral Density')
      #plt.xlim(0,100)
      #plt.savefig('Figures/spec.png', bbox_inches='tight', pad_inches=0.5)
      plt.show()

    if ('DFT' in Plot) or ('All' in Plot):
      plt.plot(freq, spectrum)
      #plt.plot(freq[between], spectrum[between], alpha=0.5)
      plt.title("real fourier transform ")
      plt.xlabel('Frequency (1/day)')
      plt.ylabel('Amplitude')
      #plt.xlim(0,100)
      #plt.savefig('Figures/fft.png', bbox_inches='tight', pad_inches=0.5)
      plt.show()
    
    
    
    if Filter == 'box':
    
      #filtered_spectrum = spectrum.copy()
    
      if inv_box == True:
        x_1 = np.arange(0, low_cutoff, 0.1)
        x_2 = np.arange(high_cutoff, np.max(freq), 0.1)
        plt.plot(freq, spectrum)
        plt.fill_between(x_1, [plt.ylim()[0]] * len(x_1), 
                     [plt.ylim()[1]] * len(x_1), color='r', alpha=0.3)
        plt.fill_between(x_2, [plt.ylim()[0]] * len(x_2), 
                     [plt.ylim()[1]] * len(x_2), color='r', alpha=0.3)
        plt.title("range to suppress")
        plt.figure()
        filtered_spectrum[lc] = 0.
        filtered_spectrum[hc] = 0.
      else:
        x_ = np.arange(low_cutoff, high_cutoff, 0.1)
        plt.plot(freq, spectrum)
        plt.fill_between(x_, [plt.ylim()[0]] * len(x_), 
                     [plt.ylim()[1]] * len(x_), color='r', alpha=0.3)
        plt.title("range to suppress")
        plt.figure()
        filtered_spectrum[between] = 0.
    
    if Filter == 'Gaussian':
      ig = invgaussian(1, np.median([low_cutoff,high_cutoff]), gf_sig, freq)
      filtered_spectrum = spectrum * ig
      if ('filter' in Plot) or ('All' in Plot):
        plt.plot(freq, ig)
        plt.title('Gaussian Filter')
        #plt.savefig('Figures/gfilter.png')
        #plt.xlim(0,100)
        plt.figure()

    if ('spec_filtered' in Plot) or ('All' in Plot):
      plt.plot(freq, filtered_spectrum, label="filtered spectrum")
      plt.plot(freq, spectrum, c='k', ls="--", label="spectrum", alpha=0.5)
      plt.title("Unfiltered vs. Filtered Spectrum")
      plt.xlabel('Frequency (1/day)')
      plt.ylabel('Amplitude')
      #plt.xlim(0,100)
      #plt.savefig('Figures/filter_compare.png', bbox_inches='tight', pad_inches=0.5)
      plt.figure()

    filtered_signal = np.fft.irfft(filtered_spectrum)  # Construct filtered signal

    if ('signal_filtered' in Plot) or ('All' in Plot):
      fig = plt.figure(figsize=(15,10)) 
      plt.plot(filtered_signal, label="filtered signal")
      plt.plot(flux, c='k', ls="--", label="original signal", alpha=0.5)
      plt.xlabel('Time')
      plt.ylabel('Amplitude')
      plt.title("Unfiltered vs. Filtered Signal")
      #plt.savefig('Figures/filtered_signal.png', bbox_inches='tight', pad_inches=0.5)
      plt.legend()
      #Filtered_signal = np.zeros_like(Filtered_signal)
    return spectrum, freq, filtered_spectrum, filtered_signal, low_cutoff, high_cutoff

def pc_thr(dataframe, frequency, fund, threshold):
    '''
    Identifies PCs with significant power at a given fundamental frequency
    Inputs:
    dataframe: pandas DataFrame of PCA components (PCs are rows)
    frequency: numpy array of frequencies corresponding to the PCA components
    fund: float, fundamental frequency to check for significant power
    threshold: float, multiplier for standard deviation to define significance
    Outputs:
    pc_list: list of indices of PCs with significant power at the fundamental frequency
    '''
    pcs = dataframe.values
    pc_list = []
    f_ind = np.argmin(np.abs(frequency - fund))
    fstep = np.diff(frequency).mean()
    std = 0.1
    pts_per_std = int(std/fstep)

    for i, pc in enumerate(pcs):

        local_avg = np.mean(np.concatenate((pc[(f_ind - pts_per_std*2):(f_ind - pts_per_std*1)], 
                            pc[(f_ind + pts_per_std*1):(f_ind + pts_per_std*2)])))
        local_std = np.std(np.concatenate((pc[(f_ind - pts_per_std*2):(f_ind - pts_per_std*1)], 
                            pc[(f_ind + pts_per_std*1):(f_ind + pts_per_std*2)])))

    if np.abs(pc[f_ind]) > np.abs( local_avg * threshold * local_std ):

        pc_list.append(i)

    else:
        pass

    return pc_list

def hrm_gfilter(x, s, fund, nhrm=0):
    '''
    Create a Gaussian band-stop filter to attenuate specified harmonics.
    Inputs:
    x: numpy array of frequencies
    s: float, standard deviation of the Gaussian
    fund: float, fundamental frequency
    nhrm: int, number of harmonics to include in the filter
    Outputs:
    filt: numpy array, the resulting band-stop filter
    '''

    amps=np.ones(len(x))
    fhrms = np.array([fund])

    for i in range(2, nhrm+2):
        fhrms = np.append(fhrms, i*fund)

    filt = np.ones(len(x))

    for hrm, amp in zip(fhrms, amps):
        filt *= invgaussian(amp,hrm,s,x) 
    # filt += gaussian(amp,hrm,s,x)

    return filt

def pc_filt(dataframe, frequency, fund, std, nhrms, subset=None, plot=False):

    '''
    Takes a set of Discrete Fourier Transform PCs with a common frequency domain 
    and bandpass filters them on a given 
    fundamental frequency and associated harmonics
    with a Gaussian filter of standard deviation "std"
    Inputs:
    dataframe: pandas DataFrame of PCA components (PCs are rows)
    frequency: numpy array of frequencies corresponding to the PCA components
    fund: float, fundamental frequency for filtering
    std: float, standard deviation of the Gaussian filter
    nhrms: int, number of harmonics to include in the filter
    subset: list of int, indices of PCs to filter (if None, all PCs are filtered)
    plot: bool, whether to plot the first PC before and after filtering
    Outputs:
    new_pcs: numpy array of filtered PCA components
    '''

    #Create pcs and new_pcs
    new_pcs = []
    pcs = dataframe.values

    #Select subset of pcs if applicable:
    if subset != None:
        pcs = np.take(pcs, subset, axis=0)
    else:
        pass

    #Bandpass filter pcs:
    for i, pc in enumerate(pcs):

        fstep = np.diff(frequency).mean()
        pts_per_std = int(std/fstep)
        dftpc = np.copy(pc)  
        hrm_inds = []

        for i in range(1,nhrms+2):
            hrm_inds.append(np.where(np.abs(frequency - fund*i) == np.abs(frequency - fund*i).min())[0][0])


        for k, ind in enumerate(hrm_inds):
            local_avg = np.mean(np.concatenate((dftpc[(ind - pts_per_std*2):(ind - pts_per_std*1)], 
                        dftpc[(ind + pts_per_std*1):(ind + pts_per_std*2)])))

        dftpc[5:] -= local_avg
        dftpc[5:] *= hrm_gfilter(frequency[5:], std, fund*(k+1))
        dftpc[5:] += local_avg

        new_pcs.append(dftpc)

    #Replace modified pcs in the ensemble
    if subset != None:
        prod = dataframe.values.copy()
        for n, m in zip(range(len(subset)), subset):
            prod[m] = new_pcs[n]
        return prod
    
    if plot:
        plt.plot(frequency, pcs[0], label='Original PC1')
        plt.plot(frequency, new_pcs[0], label='Filtered PC1')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('PC1 (Imaginary) Before and After Filtering')
        plt.legend()
        plt.show()

    return np.array(new_pcs)