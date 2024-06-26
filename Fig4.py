import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import scipy.integrate as integr

# Initial and final conditions and parameters used
v0 = 0
x0 = 5
x01 = 1.5090467417292208
v01 = 0
xf = 2
vf = 0
mu = 0.1
sf = 10

# Optimal path definitions and plotting of Optimal path in the phase space, and optimal force protocol vs sf
def f(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0]]
sol = integr.solve_ivp(f,[0,100],[2,0],'RK45',dense_output=True,rtol=1e-6,atol=1e-6)
def xfree(t):
    return sol.sol(t)[0]
def vfree(t):
    return sol.sol(t)[1]
fig1,ax0 = plt.subplots()
fig2,ax1 = plt.subplots()
ax0.set_xlabel(r'$x_1$',fontsize=20)
ax0.set_ylabel(r'$x_2$',fontsize=20)
ax1.set_xlabel(r'$s_f$',fontsize=20)
ax1.set_ylabel(r'$F_{\mathrm{opt}}$',fontsize=20)

tf = sf * mu

def g(x):
    return 1/2 * x * np.sqrt(x**2 - 1) - 1/2 * np.log(np.abs(x + np.sqrt(x**2 - 1)))
c0 = g(x0)
c1 = 1/(tf) * (g(xf) - g(x0))
def xEL(t):
    return root_scalar(lambda x: 1/2*x*np.sqrt(x**2-1) - 1/2*np.log(np.abs(x+np.sqrt(x**2-1))) - c1*t - c0,x0=(xf+x0)/2).root
xEL = np.vectorize(xEL)
tt = np.linspace(0,tf,1000)
mid_xEL = xEL(tf/2)
mid_yEL = c1 / np.sqrt(mid_xEL**2 - 1)

ax0.annotate('', xy=(xEL(tf/2+0.1), c1 / np.sqrt(xEL(tf/2+0.1)**2 - 1)), xytext=(xEL(tf/2), c1 / np.sqrt(xEL(tf/2)**2 - 1)), 
             arrowprops=dict(arrowstyle='-|>', color='gray', lw=1.5))



ax0.plot(xEL(tt),c1/np.sqrt(xEL(tt)**2 - 1),color='gray', ls = '-')
mid_y0 = (0 + c1 / np.sqrt(xEL(0)**2 - 1)) / 2
ax0.plot([x0, x0], [0, c1 / np.sqrt(xEL(0)**2 - 1)], color='blue', ls = '-')
ax0.annotate('', xy=(x0, mid_y0 - 0.05), xytext=(x0, 0 + 0.05), 
             arrowprops=dict(arrowstyle='-|>', color='blue', lw=1.5))
mid_yf = (c1 / np.sqrt(xEL(tf)**2 - 1) + vf) / 2
ax0.plot([xf, xf], [c1 / np.sqrt(xEL(tf)**2 - 1), vf], color='red', ls = '-')
ax0.annotate('', xy=(xf, mid_yf + 0.05), xytext=(xf, c1 / np.sqrt(xEL(tf)**2 - 1) - 0.05), 
             arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5))

ax0.scatter(x0, v0, color='black', label='$B_i$')
ax0.text(x0-0.3, v0 + 0.2, r'$B_i$', color='black', fontsize=16)
ax0.scatter(x01, v01, color='black', label='$A_i$')
ax0.text(x01 - 0.3, v01 + 0.2, r'$A_i$', color='black', fontsize=16)
ax0.scatter(2, 0, color='black', label='$B_f$')
ax0.text(2 + 0.1, 0 + 0.2, r'$B_f$', color='black', fontsize=16)
ax0.scatter(x01, 1.3786930837621627, color='black', label='$A_f$')
ax0.text(x01 - 0.3, 1.3786930837621627 + 0.4, r'$A_f$', color='black', fontsize=16)
ax0.scatter(x01, -1.2553221390082507, color='black', label='$A_f2$')
ax0.text(x01 - 0.3, -1.2553221390082507 + 0.2, r"$A'_f$", color='black', fontsize=16)
ax0.plot([x01, x01], [v01, -1.2553221390082507], color='blue',ls = '--')
ax0.annotate('', xy=(x01, (v01 + (-1.2553221390082507))/2 + 0.05), xytext=(x01, v01 - 0.05), 
             arrowprops=dict(arrowstyle='-|>', color='blue',linestyle='--', lw=1.5))
ax0.plot([x01, x01], [v01, 1.3786930837621627], color='red', ls = '--')
ax0.annotate('', xy=(x01, (v01 + (1.3786930837621627))/2 + 0.05), xytext=(x01, v01 - 0.05), 
             arrowprops=dict(arrowstyle='-|>', color='red',linestyle='--', lw=1.5))
def F(t):
    return -c1**2*xEL(t)/(xEL(t)**2 - 1) + mu*c1*np.sqrt(xEL(t)**2 -1) + xEL(t)
ax1.plot(tt/mu, F(tt))
ax1.plot([0,tf/mu],[x01,x01],'--')
ax1.annotate('', xy=(0, F(0)-20), xytext=(0, F(0)), 
             arrowprops=dict(arrowstyle='-|>', color='blue', lw=1.5))
ax1.annotate('', xy=(tf/mu, F(tf)+20), xytext=(tf/mu, F(tf)), 
             arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5))
ax1.annotate('', xy=(tf/mu, x01 - 20), xytext=(tf/mu, x01), 
             arrowprops=dict(arrowstyle='-|>', color='blue', linestyle='--', lw=1.5))
ax1.annotate('', xy=(tf/mu, x01 + 20), xytext=(tf/mu, x01), 
             arrowprops=dict(arrowstyle='-|>', color='red', linestyle='--', lw=1.5))


ax0.plot([1,1],[-6,3],'k-.',linewidth=1)
ax0.plot([0,5.2],[0,0],'k--',linewidth=1)
tcycle = np.linspace(90,100,1000)
ax0.plot(xfree(tcycle),vfree(tcycle),'k-',linewidth = 2)
ax0.tick_params(axis='both', which='major', direction = 'in',top=True, right=True, labeltop=False, labelright=False,labelsize=16)
ax1.tick_params(axis='both', which='major', direction = 'in',top=True, right=True, labeltop=False, labelright=False,labelsize=16)
ax0.set_xlim(0,5.2)
ax0.set_ylim(-6,3)
ax1.set_xlim(-0.4,10.4)
ax1.set_ylim(-70,25)
fig1.tight_layout()
fig2.tight_layout()
plt.show()


