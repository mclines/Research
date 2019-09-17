import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import moving_ls
import compare


x0, A, gamma = 12, 3, 5

def f(x,x0,a,gamma):
    return 1/(1+25*x**2)

n = 1000
x = np.linspace(-5, 5, n)
yexact = f(x,1,1,1)

# Add some noise with a sigma of 0.5 apart from a particularly noisy region
# near x0 where sigma is 3
sigma = np.ones(n)*0.5
sigma[np.abs(x-x0+1)<1] = 3
noise = np.random.randn(n) * sigma
y = yexact

def fL(x, x0, A, gamma):
    """ The Lorentzian entered at x0 with amplitude A and HWHM gamma. """
    return A *gamma**2 / (gamma**2 + (x-x0)**2)

def rms(y, yfit):
    return np.sqrt(np.sum((y-yfit)**2))

# Unweighted fit
p0 = 10, 4, 4
popt, pcov = curve_fit(f, x, y, p0)
yfit = f(x, *popt)
print('Unweighted fit parameters:', popt)
print('Covariance matrix:'); print(pcov)
print('rms error in fit:', rms(yexact, yfit))
print()

# Weighted fit
popt2, pcov2 = curve_fit(f, x, y, p0, sigma=sigma, absolute_sigma=True)
yfit2 = f(x, *popt2)
print('Weighted fit parameters:', popt2)
print('Covariance matrix:'); print(pcov2)
print('rms error in fit:', rms(yexact, yfit2))

# #my part
building_nodes = np.linspace(-5,5,500)
my_exact = f(x,1,1,1)
nodes = compare.create_vandermonde(x,3)
moving_values, _ = moving_ls.main(nodes,y,building_nodes)




plt.plot(x, yexact,'r', label='Exact')
#plt.scatter(x, y, color = 'black',marker = 'o', lw='.05', label='Noisy data')
#plt.plot(x, yfit,'b', label='Unweighted fit')
plt.plot(x, yfit2,'g', label='Weighted fit')
plt.plot(building_nodes,moving_values,'b',label='Moving fit')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.legend(loc='lower left')
plt.savefig('./results/comaprison.png')

# pylab.plot(x, yexact, label='Exact')
# pylab.plot(x, y, 'o', label='Noisy data')
# pylab.plot(x, yfit, label='Unweighted fit')
# pylab.plot(x, yfit2, label='Weighted fit')
