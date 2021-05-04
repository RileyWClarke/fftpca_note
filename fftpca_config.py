import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Raw LCs:
bjds = np.load('Data/bjds.npy')
fluxes = np.load('Data/fluxes.npy')
fluxerrs = np.load('Data/fluxerrs.npy')

#Raw Short Cad LCs:

data1=np.loadtxt('Data/phot211046195r2_ssc.2m0335.dat')
data2=np.loadtxt('Data/phot210327027r2_ssc.2m0355.dat')

time1,flux1,xx1,yy1=data1[:,0],data1[:,1],data1[:,2],data1[:,3]
time2,flux2,xx1,yy2=data2[:,0],data2[:,1],data2[:,2],data2[:,3]

flux1norm = (flux1 - np.mean(flux1)) / np.std(flux1)
flux2norm = (flux2 - np.mean(flux2)) / np.std(flux2)

df = pd.DataFrame({'flux1':flux1})
flux1rm = df['flux1']-df['flux1'].rolling(200).mean()

print(flux1rm[36000:36200].mean())

plt.plot(df['flux1'].values)
plt.plot(flux1rm)
plt.show()

#Processed LCs:

flux_p = np.load('Data/processed_lcs.npy')

flare_ts = [2253.65107,
          2240.04075,
          2281.57290,
          2284.68258,
          2287.91009,
          2268.87127,
          2295.39965,
          2299.37770,
          2291.45770,
          2248.59421,
          2261.51045,
          2269.02792,
          2249.17243,
          2276.02709,
          2229.08574,
          2293.30131,
          2295.02575,
          2258.54786,
          2284.92844,
          2238.85706,
          2251.42265,
          2249.41148,
         ]
