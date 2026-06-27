import numpy as np
rg = np.random.default_rng(1)
import matplotlib.pyplot as plt
mu, sigma = 2, 0.5
v = rg.normal(mu, sigma, 10000)
plt.hist(v, bins=50, density=True)      
(n, bins) = np.histogram(v, bins=50, density=True) 
plt.plot(.5 * (bins[1:] + bins[:-1]), n) 

plt.show()