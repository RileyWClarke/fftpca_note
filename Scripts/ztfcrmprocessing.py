import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from sys import argv

data = pd.read_csv(argv[1], header=None)
data.drop(data.index[-1], inplace=True)

temp = np.array([np.array(name.split('_')[2:]) for name in data[0].values])

data['rundate'] = temp[:,0]
data['runnumber'] = temp[:,1]
data['filter'] = temp[:,2]
data['chip'] = temp[:,3]
data['segment'] = temp[:,-4]

data['source_xy'] = [t[:-4] for t in temp[:,-2] + temp[:,-1]]

data.groupby(['rundate', 'runnumber', 'filter', 'chip', 'source_xy']).count()

data[0][(data['chip'] == 'c05') & (data['segment'] == '0000')].values

arr = np.zeros(data[(data['chip'] == 'c05') & (data['segment'] == '0000')].values.shape[0])

for i, name in enumerate(data[0][(data['chip'] == 'c05') & (data['segment'] == '0000')].values):
    source = pd.read_csv(f'Data/ztf_crm/{name}')
    flux = source['flux_hap_2'].values
    middle_flux_mean = flux[int(len(flux) / 2 - 50) : int(len(flux) / 2 + 50)].mean()
    arr[i] = middle_flux_mean

testnames = data[0][(data['chip'] == 'c05') & (data['segment'] == '0000')].values[np.argwhere(arr >= np.sort(arr)[-10]).flatten()] #get names of the 10 sources with highest mean flux
testlcs = []

for i, name in enumerate(testnames):
    source = pd.read_csv(f'Data/ztf_crm/{name}')
    testlcs.append(source['flux_hap_2'].values)

testlcs = np.array(testlcs)

for i in range(len(testlcs)):
    testlcs[i] = (testlcs[i] - np.nanmean(testlcs[i])) / np.nanstd(testlcs[i])

np.save(f'{argv[2]}', testlcs)


