import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import scipy.integrate as integr

# Initial value and parameter
x0 = 5
v0 = 0
mu = 10**(-1)
# Limit Cycle Points for that mu. We keep only the points with vf<0, those are the ones minimizing Wt for this case
data = np.loadtxt('limitcycle1new.dat') # Generating by solving numerically van der Pol oscillator with the same mu and initial condition x=2, x'=0 and letting it evolve enough to reach equilibrium (LimitCycle.py).
xftt = data[:,0] 
vftt = data[:,1]
index = vftt <= 0 
xft = xftt[index]
vft = vftt[index]


def Wnc(xf):
    tf = sf * mu
    c0 = 1/2 * x0 * np.sqrt(x0**2 - 1) - 1/2 * np.log(np.abs(x0 + np.sqrt(x0**2 - 1)))
    c1 = 1/(tf) * (1/2 * xf * np.sqrt(xf**2 - 1) - 1/2 * np.log(np.abs(xf + np.sqrt(xf**2 - 1))) - c0)
    return mu * c1**2 * tf

def Wc(xf,vf):
    Wc = 1/2*(xf**2 + vf ** 2)
    return Wc

def Wtotal(xf,vf):
    return Wc(xf,vf) + Wnc(xf) - 0.5*(x0**2 + v0**2)

fig1, ax1 = plt.subplots()
Wtotalc = np.zeros(len(xft))
Wcc = np.zeros(len(xft))
Wncc = np.zeros(len(xft))
sftt = [170,174.7,180,190]
colors = ['tab:blue','tab:orange','tab:green','tab:red']
xfminf = np.zeros(len(sftt))
vfminf = np.zeros(len(sftt))
for j in range(len(sftt)):
    sf = sftt[j]
    for i in range(len(xft)):
        Wtotalc[i] = Wtotal(xft[i],vft[i])
        Wcc[i] = Wc(xft[i],vft[i])
        Wncc[i] = Wnc(xft[i])
    xfminf[j] = xft[np.argmin(Wtotalc)]
    vfminf[j] = vft[np.argmin(Wtotalc)]
    label = f'$\\mathrm{{s_f}} = {sftt[j]:.1f}$'
    if j == 1:
        ax1.plot(1.33,np.min(Wtotalc),'*',c=colors[j],ms = 10)
        label = f'$\\mathrm{{s_f^*}} = {sftt[j]:.1f}$'
    ax1.plot(xft,Wtotalc,'-',label=label,c=colors[j])
    ax1.plot(xfminf[j],np.min(Wtotalc),'*',c=colors[j],ms = 10)
    ax1.plot(xfminf[j],np.min(Wtotalc),'*',c=colors[j],ms = 10)
    ax1.plot([1,2],[np.min(Wtotalc),np.min(Wtotalc)],'--',c=colors[j],lw = 1)
ax1.set_xlim(0.95,2.05)
ax1.set_ylim(-10.04,-9.89)
ax1.legend(fontsize=16,loc = 'lower right')
ax1.set_xlabel(r'$x_{1f}$',fontsize=20)
ax1.set_ylabel(r'$W$',fontsize=20)
ax1.tick_params(axis='both', which='major', direction = 'in',top=True, right=True, labeltop=False, labelright=False,labelsize=16)
fig1.tight_layout()

sft = np.linspace(0.01,1000,100)
xfmin = np.zeros(len(sft))
vfmin = np.zeros(len(sft))
for j in range(len(sft)):
    sf = sft[j]
    for i in range(len(xft)):
        Wtotalc[i] = Wtotal(xft[i],vft[i])
        Wcc[i] = Wc(xft[i],vft[i])
        Wncc[i] = Wnc(xft[i])
    xfmin[j] = xft[np.argmin(Wtotalc)]
    vfmin[j] = vft[np.argmin(Wtotalc)]
fig2, ax2 = plt.subplots()
ax2.plot(sft,xfmin,'.',c = 'tab:blue', ms = 8)
for i in range(len(sftt)):
    if i != 1:
        ax2.plot(sftt[i],xfminf[i],'.',c = 'tab:blue', ms = 8)
        ax2.plot(sftt[i],xfminf[i],'s',c = colors[i],mfc = 'None', ms = 10)
ax2.plot([174.7,174.7],[0.9,2.1],'k:')
text = r'$s_f^*$'
ax2.text(120,1.1,text,fontsize = 20)
ax2.set_xlim(-10,1010)
ax2.set_ylim(0.9,2.1)
ax2.set_xlabel(r'$s_f$',fontsize=20)
ax2.set_ylabel(r'$x_{1f}^{\mathrm{opt}}$',fontsize=20)
ax2.tick_params(axis='both', which='major', direction = 'in',top=True, right=True, labeltop=False, labelright=False,labelsize=16)
fig2.tight_layout()

plt.show()