import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integr
from scipy.optimize import minimize, root_scalar
from scipy.integrate import quad


# Initial conditions and parameters
x0 = 5
v0 = 0
mu = 10**(-1)

# Parameters of the finite time trajectory
tf = 1
k = 15
xf = 1.9551381824044118 
vf = -0.4130041173463992

# Finite time trajectory
def fmin(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0] - k]

def fmax(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0] + k]

def f(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0]]

def c1(t1,t2):
    return 1/(t2-t1) * ((1/2*xmax(t2)*(xmax(t2)**(2)-1)**(1/2)-1/2*np.log(np.abs(xmax(t2)+(xmax(t2)**(2)-1)**(1/2))))-(1/2*xmin(t1)*(xmin(t1)**(2)-1)**(1/2)-1/2*np.log(np.abs(xmin(t1)+(xmin(t1)**(2)-1)**(1/2)))))
def c2(t1,t2):
    return 1/(t2-t1) * ((1/2*xmin(t1)*(xmin(t1)**(2)-1)**(1/2)-1/2*np.log(np.abs(xmin(t1)+(xmin(t1)**(2)-1)**(1/2))))*t2 -(1/2*xmax(t2)*(xmax(t2)**(2)-1)**(1/2)-1/2*np.log(np.abs(xmax(t2)+(xmax(t2)**(2)-1)**(1/2))))*t1)
def eq1(t1,t2):
    return c1(t1,t2)/(xmin(t1)**2-1)**(1/2)
def eq2(t1,t2):
    return c1(t1,t2)/(xmax(t2)**2-1)**(1/2)
def wmininter(t):
    t1 = t[0]
    t2 = t[1]
    return np.abs(eq1(t1,t2)-vmin(t1))**2 + np.abs(eq2(t1,t2)-vmax(t2))**2
cons = {'type': 'ineq', 'fun': lambda t : t[1]-t[0]}

solmin = integr.solve_ivp(fmin,[0,tf],[x0,v0],'RK45',dense_output=True,rtol=1e-6,atol=1e-6)
def xmin(t):
    return solmin.sol(t)[0]
def vmin(t):
    return solmin.sol(t)[1]
solmax = integr.solve_ivp(fmax,[tf,0],[xf,vf],'RK45',dense_output=True,rtol=1e-6,atol=1e-6)
def xmax(t):
    return solmax.sol(t)[0]
def vmax(t):
    return solmax.sol(t)[1]
result = minimize(wmininter, [0.1, 0.9], constraints=cons, bounds=[(0, tf), (0, tf)], method='SLSQP')
times = result.x
if (np.abs(result.fun) > 10**(-2) ):
    print('Warning!!! Fun is too high!!\n')
def xEL(t):
    return root_scalar(lambda x: 1/2*x*np.sqrt(x**2-1) - 1/2*np.log(np.abs(x+np.sqrt(x**2-1))) - c1(times[0],times[1])*t - c2(times[0],times[1]),x0=(xf+x0)/2).root
xEL = np.vectorize(xEL)

# Free trajectory
sol = integr.solve_ivp(f,[0,100],[x0,v0],'RK45',dense_output=True,rtol=1e-6,atol=1e-6)
def xfree(t):
    return sol.sol(t)[0]
def vfree(t):
    return sol.sol(t)[1]


# Plotting
fig, ax = plt.subplots()
ax.set_xlabel(r'$x_1$',fontsize=20)
ax.set_ylabel(r'$x_2$',fontsize=20)
ax.tick_params(axis='both', which='major', direction='in',top=True, right=True, labeltop=False, labelright=False, labelsize=16)
tt = np.linspace(0,100,1000)
ax.plot(xfree(tt),vfree(tt),'r',linestyle='--')
ax.plot(xfree(np.linspace(90,100,1000)),vfree(np.linspace(90,100,1000)),'k-',linewidth=3)
ax.plot(xmin(np.linspace(0,times[0],100)),vmin(np.linspace(0,times[0],100)),'b')
ax.plot(xmax(np.linspace(times[1],tf,100)),vmax(np.linspace(times[1],tf,100)),'b')
ax.plot(xEL(np.linspace(times[0],times[1])),c1(times[0],times[1])/np.sqrt(xEL(np.linspace(times[0],times[1]))**2-1),'b')
ax.plot([-2.001,-2.001],[0,3],'k:')
ax.plot([2.001,2.001],[0,3],'k:')
text = r'$-x_{\ell c}^{\mathrm{max}}$'
ax.text(-3.6,2,text,fontsize = 20)
text2 = r'$x_{\ell c}^{\mathrm{max}}$'
ax.text(2.2,2,text2,fontsize = 20)
ax.plot([-4,6],[0,0],'k--', lw = 1)
ax.plot([0,0],[-6,4],'k--', lw = 1)
ax.set_aspect('equal')
fig.tight_layout()
ax.set_xlim(-4,6)
ax.set_ylim(-5.5,3)
plt.show()