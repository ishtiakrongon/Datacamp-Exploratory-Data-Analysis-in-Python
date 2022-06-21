# Modeling Distributions
# This is the part of the Distributions notebook


#-------------------------------------------------------------------------------

# THE NORMAL DISTRIBUTION (Also known as Gaussian Distribution)

# Importing Libraries

from unicodedata import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from empiricaldist import Cdf
#-------------------------------------------------------------------------------


# THE NORMAL DISTRIBUTION

sample = np.random.normal( size = 1000 )

Cdf(sample).plot()

plt.show()



#-------------------------------------------------------------------------------

# The Normal CDF

xs = np.linspace( -3, 3 )       # creating equally spaced array
ys = norm(0, 1).cdf(xs)         # here norm(0, 1) creates an object that
                                # represents a normal distribution with mean 0
                                # and standard deviation 1.

#-------------------------------------------------------------------------------

# Plotting the results

plt.plot(xs, ys , color = 'gray')
Cdf(sample).plot()

plt.show()


#-------------------------------------------------------------------------------