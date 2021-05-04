import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import oaconvolve
from scipy import signal

def rayleigh(x, s):

    r = (x / s**2) * np.exp(-x**2 / (2 * s**2))

    return r

rtemps = np.zeros((10,1001))

x = np.arange(-50,50.1,0.1)

for i, s in enumerate(np.arange(0.5,5.5,0.5)):
    rtemps[i] = rayleigh(x, s)
    plt.plot(x, rtemps[i], label=r'$\sigma$ = {}'.format(s))
    plt.title('Templates')
    plt.legend()

test = np.ones(10001)
test[4500:5501] += rayleigh(x, 2.0)

fig, axes = plt.subplots()

axes.plot(test)
axes.set_title('Signal, $\sigma$ = 2.0')

fig, axes = plt.subplots(1,1, figsize=(10,5))

for i, s in enumerate(np.arange(0.5,5.5,0.5)):
    convtest = oaconvolve(test, rtemps[i], mode='valid')
    axes.plot(convtest, label=r'$\sigma$ = {}'.format(s))
plt.title('Convolutions')
plt.legend()
plt.show()

