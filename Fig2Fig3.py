import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Initial and final conditions and parameters used
v0 = 0
xf = 2.000087518309731 
vf = 0
mu = 10**(-1)
x0t = np.linspace(2.1, 5.1, 1000)
sft = np.linspace(0.1, 10.1, 1000)
X, Y = np.meshgrid(x0t, sft)

# W_nc^min
def Wncanalytic(x0, sf):
    tf = sf * mu
    c0 = 1/2 * x0 * np.sqrt(x0**2 - 1) - 1/2 * np.log(np.abs(x0 + np.sqrt(x0**2 - 1)))
    c1 = 1/(tf) * (1/2 * xf * np.sqrt(xf**2 - 1) - 1/2 * np.log(np.abs(xf + np.sqrt(xf**2 - 1))) - c0)
    return mu * c1**2 * tf

Z = Wncanalytic(X, Y)


# Density plot of W_nc^min
plt.figure()
plt.imshow(np.log10(Z), aspect='auto', origin='lower', extent=[x0t.min(), x0t.max(), sft.min(), sft.max()], vmin = -2, vmax = 3, cmap='jet')
cb = plt.colorbar()
axcb = cb.ax
axcb.tick_params(axis='both', which='major', labelsize=16)
colorbar_ticks = cb.get_ticks()
colorbar_tick_labels = [f'$10^{{{int(tick)}}}$' for tick in colorbar_ticks]
cb.set_ticks(colorbar_ticks)
cb.set_ticklabels(colorbar_tick_labels)
axcb.set_ylabel(r'$\widetilde{W}_{\mathrm{nc}}^{\;\mathrm{min}}$', fontsize=20)
plt.xlabel(r'$|x_{10}|$',fontsize=20)
plt.ylabel(r'$s_f$',fontsize=20)
plt.tight_layout()
plt.tick_params(axis='both', which='major', direction = 'in',top=True, right=True, labeltop=False, labelright=False,labelsize=16)



# W_nc^min vs x_0 and W_nc^min vs s_f plots
fig1,ax = plt.subplots()
fig2,ax1 = plt.subplots()
x0s = [2,2.5,3,3.5]
sfs = [0.5,1,5,10]
lss=['-','--','-.',':']
for i in range(len(x0s)):
    sft = np.linspace(0.1, 10.1, 1000)
    Wncsf = Wncanalytic(x0s[i],sft)
    if x0s[i] <= 2:
        sft = np.append(0,sft)  
        Wncsf = np.append(0,Wncsf)
    ax.plot(sft,Wncsf,ls = lss[i])
ax.legend([r'$|x_{10}| = 2.0$', r'$|x_{10}| = 2.5$', r'$|x_{10}| = 3.0$', r'$|x_{10}| = 3.5$'], fontsize=16)

for i in range(len(sfs)):
    x0t = np.linspace(2.1, 5.1, 100)
    Wncx0 = Wncanalytic(x0t,sfs[i])
    x0t = np.append(0,x0t)
    Wncx0 = np.append(0,Wncx0)
    ax1.plot(x0t,Wncx0,ls=lss[i])
ax1.plot([xf,xf],[-0.5,50],'k-.',lw = 1)
text = r'$x_{\ell c}^{\mathrm{max}}$'
ax1.text(1.6,2,text,fontsize = 20)
ax.set_xlabel(r'$s_f$',fontsize=20)
ax.set_ylabel(r'$\widetilde{W}_{\mathrm{nc}}^{\;\mathrm{min}}$',fontsize=20)
ax.tick_params(axis='both', which='major', direction = 'in',top=True, right=True, labeltop=False, labelright=False,labelsize=16)
ax.set_xlim(0,10)
ax.set_ylim(-0.1,10)
fig1.tight_layout()
ax1.set_xlabel(r'$|x_{10}|$',fontsize=20)
ax1.set_ylabel(r'$\widetilde{W}_{\mathrm{nc}}^{\;\mathrm{min}}$',fontsize=20)
ax1.tick_params(axis='both', which='major', direction = 'in',top=True, right=True, labeltop=False, labelright=False,labelsize=16)
ax1.set_xlim(1,5)
ax1.set_ylim(-0.5,50)
ax1.legend([r'$s_f = 0.5$', r'$s_f = 1.0$', r'$s_f = 5.0$', r'$s_f = 10.0$'],fontsize=16)
fig2.tight_layout()

plt.show()