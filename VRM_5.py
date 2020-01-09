import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.linalg import solve
from timeit import default_timer as timer
from winsound import Beep
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.stats import moment
from numpy.random import rand,randint
import numba
from numba import jit


import pandas as pd


#JG=a*x+b*y+c*z,b*x+d*y+o*z,c*x+o*y+f*z
#(a,b,c,d,o,f,h)

@jit(nopython=True)
def RungeKutta(y0,t0,koraki,dt,argumenti):
    r=RungeKutta_one_step(y0,dt,argumenti)
    N=int(argumenti[-1])
    shramba=np.zeros((koraki,2*N+2),dtype=np.float64)
    
    for i in np.arange(koraki):
        r=RungeKutta_one_step(r,dt,argumenti)
        shramba[i]=r
    return shramba

@jit(nopython=True)
def RungeKutta_one_step(nabla,dt,args):
    
    F1=np.multiply(f1(nabla,args),dt)
    
    F2=np.multiply(f1(np.add(nabla,np.divide(F1,2)),args),dt)
    
    F3=np.multiply(f1(np.add(nabla,np.divide(F2,2)),args),dt)
    
    F4=np.multiply(f1(np.add(nabla,F3),args),dt)
    
    
    return np.add(nabla,np.divide(np.add(np.add(F1,np.multiply(2,np.add(F2,F3))),F4),6))

@jit(nopython=True)
def f1(nabla,arg):
    ''' arg=(m1,m2,m3,...,m_N,lamda,tau,T_L,T_R,N) 
    ,nabla[2n]=p_n
    nabla[2n+1]=q_n
    nabla[2N]=\ceta_L
    nabla[2N+1]=\ceta_R
    '''
    N,tau,TL,TR,lamda=int(arg[-1]),arg[-4],arg[-3],arg[-2],arg[-5]
    
    
    x=np.array([0 for i in range(2*N+2)],dtype=np.float64)

    
    x[-2]=(nabla[0]**2/arg[0]-TL)/tau
    x[-1]=(nabla[-4]**2/arg[-6]-TR)/tau
    #---------------------------------------------------------
    x[1]=nabla[0]/arg[0]
    x[0]=-(2*nabla[1]+4*lamda*nabla[1]**3-nabla[3])-nabla[-2]*nabla[0]
    #---------------------------------------------------------
    for j in np.arange(1,N-1):
        x[2*j+1]=nabla[2*j]/arg[j]
        x[2*j]=-(3*nabla[2*j+1]+4*lamda*nabla[2*j+1]**3-nabla[2*(j-1)+1]-nabla[2*(j+1)+1])
    #---------------------------------------------------------
    x[-3]=nabla[2*(N-1)]/arg[(N-1)]
    x[-4]=-(2*nabla[-3]+4*lamda*nabla[-3]**3-nabla[-5])-nabla[-1]*nabla[-4]

    return x

@jit(nopython=True)
def Obdelava(rešitev,N,koraki,dt):
    T=np.zeros(N,dtype=np.float64)
    for j in range(N):
        x=rešitev[:,2*j]
        T[j]=np.dot(x,x)*dt/(koraki*dt)
    return T

#nabla=[x1,y1,z1,x2,y2,z2]
def f(t,nabla,arg):
    ''' arg=(m1,m2,m3,...,m_N,lamda,tau,T_L,T_R,N) 
    ,nabla[2n]=p_n
    nabla[2n+1]=q_n
    nabla[2N]=\ceta_L
    nabla[2N+1]=\ceta_R
    '''
    N,tau,TL,TR,lamda=arg[-1],arg[-4],arg[-3],arg[-2],arg[-5]
    
    
    x=np.array([0 for i in range(2*N+2)])

    
    x[-2]=(nabla[0]**2/arg[0]-TL)/tau
    x[-1]=(nabla[-4]**2/arg[-6]-TR)/tau
    #---------------------------------------------------------
    x[1]=nabla[0]/arg[0]
    x[0]=-(2*nabla[1]+4*lamda*nabla[1]**3-nabla[3])-nabla[-2]*nabla[0]
    #---------------------------------------------------------
    for j in np.arange(1,N-1):
        x[2*j+1]=nabla[2*j]/arg[j]
        x[2*j]=-(3*nabla[2*j+1]+4*lamda*nabla[2*j+1]**3-nabla[2*(j-1)+1]-nabla[2*(j+1)+1])
    #---------------------------------------------------------
    x[-3]=nabla[2*(N-1)]/arg[(N-1)]
    x[-4]=-(2*nabla[-3]+4*lamda*nabla[-3]**3-nabla[-5])-nabla[-1]*nabla[-4]

    return x

#rešimo sistem enačb
def Hoover(y0,t0,t1,dt,argumenti):
    rezultat=[]
    r = ode(f).set_integrator('dopri5')
    r.set_initial_value(y0, t0).set_f_params(argumenti)
    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
        rezultat.append(r.y)
    g=np.array(rezultat)
    return g


N=10
lamda=0
tau=1
T_L=1
T_R=2
#argumenti=()
#for i in range(N):
#    argumenti=argumenti+(1,)
#    
#argumenti=argumenti+(lamda,tau,T_L,T_R,N,)
argumenti=[]
for i in range(N):
    argumenti.append(1)
    

argumenti.append(lamda)
argumenti.append(tau)
argumenti.append(T_L)
argumenti.append(T_R)
argumenti.append(N)



začetni=np.array([rand() for i in range(2*N+2)],dtype=np.float64)


#t1=5000
dt=0.1
koraki=500000
#sol=Hoover(začetni,0,t1,dt,argumenti)
sol=RungeKutta(začetni,0,koraki,dt,np.array(argumenti,dtype=np.float64))
print(argumenti)



T=Obdelava(sol,N,koraki,dt)
plt.plot(T)












