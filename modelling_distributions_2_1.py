# Modeling Distributions
# This is the part of the Distributions notebook


#-------------------------------------------------------------------------------

# THE NORMAL DISTRIBUTION (Also known as Gaussian Distribution)

# Importing Libraries

from unicodedata import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Loading Datasets

gss = pd.read_hdf('D:/git_repositories/Datacamp-Exploratory-Data-Analysis-in-Python/Data/gss.hdf5', 'gss')

print(gss.head())



#-------------------------------------------------------------------------------


# Extract realinc and compute its log
income = gss['realinc']
log_income = np.log10(income)
log_income.head()


# Compute mean and standard deviation
mean = np.round( np.mean(log_income), 2 )
std = np.round( np.std(log_income), 2 )
print(mean, std)

# Make a norm object

dist = norm(mean, std)

# Evaluate the model CDF
xs = np.linspace(2, 5.5)
ys = dist.cdf(xs)

# Plot the model CDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Create and plot the Cdf of log_income

# Plot the data KDE
sns.kdeplot(log_income)
    
# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('CDF')
plt.show()


#-------------------------------------------------------------------------------

# Evaluate the normal PDF
xs = np.linspace(2, 5.5)
ys = dist.pdf(xs)

# Plot the model PDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Plot the data KDE
sns.kdeplot(log_income)
    
# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('PDF')
plt.show()
