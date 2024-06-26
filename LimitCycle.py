import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import scipy.integrate as integr   

# Code used to generate a numerical sampling of the limit cycle of a van der Pol oscillator.
mu = 10**(-1)
def vdp(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0]]

freesol = integr.solve_ivp(vdp,[0,30*np.pi],[2,0],method='Radau', dense_output=True,rtol=1e-6,atol=1e-6)
def xfree(t):
    return freesol.sol(t)[0]
def vfree(t):
    return freesol.sol(t)[1]


tt = np.linspace(27*np.pi,30*np.pi,20000)
xcycleT = xfree(tt)
vcycleT = vfree(tt)
indx = xcycleT > 1
xcycle = xcycleT[indx]
vcycle = vcycleT[indx]
plt.plot(xcycle,vcycle,'.')
plt.show()


limitcycle = open('limitcycle1new.dat','a')
for i in range(len(xcycle)):
    limitcycle.write(f'{xcycle[i]} {vcycle[i]}\n')
limitcycle.close()