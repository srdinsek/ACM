import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_triangular as tri
from scipy.linalg import solve_banded
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import axes3d
pi=np.pi


#%%
###########################################################################################################
#
#       1.1. Metoda končnih diferenc
#
###########################################################################################################

# h je dolžina krajevnega koraka, L velikost območja, T končni čas, a zamik potenciala, b zamik začetnega pogoja
# ta funkcija je za hermitov polinom N=0, spremeni če hočeš višjega

def KončnaDiferencaNormirana(h,razmerje,L,T,lamda,a,b):
    # definiramo potrebne količine
    tau=(h**2)*1/razmerje
    N,M=int(T/tau),int(L/h)
#    print(' h={}\n tau={}\n L={}\n T={}\n lamda={}\n a={}\n b={}'.format(h,tau,L,T,lamda,a,b))
    
    # tu izberemo potencial
    V=[(L/2-m*h-a)**2/2+lamda*(L/2-m*h-a)**4 for m in np.arange(M+1)]

    # postavimo začetno vrednost funkcije
    # vsaka vrstica je ob različnem času in vsak stolpec na različnem kraju
    PSIbrez=[]
    
    c=np.exp(-(L/2-b)**2/2)/pi**(1/4)
#    PSI.append([np.exp(-(L/2-m*h-b)**2/2)/pi**(1/4)-c for m in np.arange(M+1)])
    PSIbrez.append([np.exp(-(L/2-m*h-b)**2/2)/pi**(1/4)-c for m in np.arange(M+1)])
    for n in np.arange(N):
#        psi=[PSI[-1][m]+tau*1j*((PSI[-1][m+1]-2*PSI[-1][m]+PSI[-1][m-1])/(2*h**2)-V[m]*PSI[-1][m]) if m!=0 and m!=M else PSI[-1][m] for m in np.arange(M+1)]
        psibrez=[PSIbrez[-1][m]+tau*1j*((PSIbrez[-1][m+1]-2*PSIbrez[-1][m]+PSIbrez[-1][m-1])/(2*h**2)-V[m]*PSIbrez[-1][m]) if m!=0 and m!=M else PSIbrez[-1][m] for m in np.arange(M+1)]
#        norm=sum(h*psi[i]**2 for i in range(len(psi)))
#        PSI.append(np.divide(psi,norm))
        PSIbrez.append(psibrez)
    return PSIbrez
###########################################################################################################
#
#       1.2. Metoda: Skok s končnim propagatorjem
#
###########################################################################################################

# K je red do katerega razvijemo, h2 pa h**2
def ExpH(M,V,K,h2,razmerja):
    d=[(-1/h2-V[m]) if m!=0 and m!=M else 0 for m in range(M+1)]
    a=[1/(2*h2) if m!=0 else 0 for m in range(M)]
    b=[1/(2*h2) if m!=M-1 else 0 for m in range(M)]
    A=np.diag(d,0)+np.diag(a,1)+np.diag(b,-1)
    # tu uporabim maksimalno normo! Ne vem kako se obnesejo ostale,
    # ker sem jo uvedel zato ker ni delalao. A na koncu sem ugotovil, da
    # ni delalo ker sem prištel (namesto odštel) potencial
    
    tau=2*pi/(np.linalg.norm(A,np.inf)*razmerja)
    
    #tau=(h**2)*1/razmerje
    return np.sum((-1j*tau)**k*np.linalg.matrix_power(A,k)/math.factorial(k) for k in range(K+1)),tau

# h je dolžina krajevnega koraka, L velikost območja, T končni čas, a zamik potenciala, b zamik začetnega pogoja         
# ta funkcija je za hermitov polinom N=0, spremeni če hočeš višjega
def SkokSPropagatorjem(h,razmerje,L,T,lamda,a,b,K):
    # definiramo potrebne količine
    M=int(L/h)
    
    # tu izberemo potencial
    V=[(L/2-m*h-a)**2/2+lamda*(L/2-m*h-a)**4 for m in np.arange(M+1)]
    
    # postavimo začetno vrednost funkcije
    # vsaka vrstica je ob različnem času in vsak stolpec na različnem kraju
    PSI=[]
    #(4*(L/2-b-m*h)**2-2)/np.sqrt(8)*
    c=np.exp(-(L/2-b)**2/2)/pi**(1/4)
    PSI.append([np.exp(-(L/2-m*h-b)**2/2)/pi**(1/4)-c for m in np.arange(M+1)])
    H,tau=ExpH(M,V,K,h**2,razmerje)
    
    N=int(T/tau)
#    print(' h={}\n tau={}\n L={}\n T={}\n lamda={}\n a={}\n b={}'.format(h,tau,L,T,lamda,a,b))
    for n in np.arange(N):
        psi=np.dot(H,PSI[-1])
        # zelo dobro deluje, ta metoda ne potrebuje normiranja
#        norm=sum(h*psi[i]**2 for i in range(len(psi)))
#        PSI.append(np.divide(psi,norm))
        PSI.append(psi)
    return PSI,tau

###########################################################################################################
#
#       1.3. Implicitna metoda
#
###########################################################################################################
    
def UI(M,V,h2,tau):
    d=[1+1j*tau*(1/h2-V[m])/2 if m!=0 and m!=M else 1 for m in range(M+1)]
    a=[1j*tau*(1/(2*h2))/2 if m!=0 else 0 for m in range(M)]
    b=[1j*tau*(1/(2*h2))/2 if m!=M-1 else 0 for m in range(M)]
    A=np.diag(d,0)+np.diag(a,1)+np.diag(b,-1)
    B=np.conj(A)
    return A,B


# h je dolžina krajevnega koraka, L velikost območja, T končni čas, a zamik potenciala, b zamik začetnega pogoja         
# ta funkcija je za hermitov polinom N=0, spremeni če hočeš višjega
def Implicitna(h,tau,L,T,lamda,a,b):
    # definiramo potrebne količine
    M,N=int(L/h),int(T/tau)
#    print(' h={}\n tau={}\n L={}\n T={}\n lamda={}\n a={}\n b={}'.format(h,tau,L,T,lamda,a,b))
    # tu izberemo potencial
    V=[(L/2-m*h-a)**2/2+lamda*(L/2-m*h-a)**4 for m in np.arange(M+1)]
    
    # postavimo začetno vrednost funkcije
    # vsaka vrstica je ob različnem času in vsak stolpec na različnem kraju
    PSI=[]
    c=np.exp(-(L/2-b)**2/2)/pi**(1/4)
    PSI.append([np.exp(-(L/2-m*h-b)**2/2)/pi**(1/4)-c for m in np.arange(M+1)])
    A,B=UI(M,V,h**2,tau)

#    A,B=UIs(M,V,h**2,tau)

    AC=np.transpose(np.array([[A[i][(j+i-1)] for j in range(3)] if i != M else [A[M][-2],A[M][-1],A[M][0]] for i in range(M+1)]))
    for n in np.arange(N):
        C=np.dot(B,PSI[-1])
        psi=solve_banded((1,1),AC,C)
#        psi=np.linalg.solve(A,C)
        PSI.append(psi)
    return PSI

def ImplicitnaVklop(h,tau,L,T,lamda,a,b,boost):
    # definiramo potrebne količine
    M,N=int(L/h),int(T/tau)
#    print(' h={}\n tau={}\n L={}\n T={}\n lamda={}\n a={}\n b={}'.format(h,tau,L,T,lamda,a,b))
    # tu izberemo potencial
    
    # postavimo začetno vrednost funkcije
    # vsaka vrstica je ob različnem času in vsak stolpec na različnem kraju
    PSI=[]
    Lamb=[]
    c=2*(L/2-b)/np.sqrt(2)*np.exp(-(L/2-b)**2/2)/pi**(1/4)
    PSI.append([2*(L/2-b-m*h)/np.sqrt(2)*np.exp(-(L/2-m*h-b)**2/2)/pi**(1/4)-c for m in np.arange(M+1)])
    for n in np.arange(int(N/2)):
        Lamb.append(lamda*n/(N-N/2))
        V=[(L/2-m*h-a)**2/2+(lamda*n/(N-N/2))*(L/2-m*h-a)**4 for m in np.arange(M+1)]
        A,B=UI(M,V,h**2,tau)
        C=np.dot(B,PSI[-1])
        AC=np.transpose(np.array([[A[i][(j+i-1)] for j in range(3)] if i != M else [A[M][-2],A[M][-1],A[M][0]] for i in range(M+1)]))
        psi=solve_banded((1,1),AC,C)
        PSI.append(psi)
        
    V=[(L/2-m*h-a)**2/2+(lamda*n/(N-N/2))*(L/2-m*h-a)**4 for m in np.arange(M+1)]
    A,B=UI(M,V,h**2,tau)
    AC=np.transpose(np.array([[A[i][(j+i-1)] for j in range(3)] if i != M else [A[M][-2],A[M][-1],A[M][0]] for i in range(M+1)]))
    for n in np.arange(int(N/2),N*boost):
        Lamb.append(lamda)
        C=np.dot(B,PSI[-1])
        psi=solve_banded((1,1),AC,C)
        PSI.append(psi)
    return PSI,Lamb

#%%
#%%###########################################################################################################
#
#       Tu računam podatke
#
###########################################################################################################

#                       DIF
h,razmerje,L,T,lamda,a,b=0.1,20,10,1,0,0,0
tau=(h**2)*1/razmerje
N,M=int(T/tau),int(L/h)
PSI=KončnaDiferencaNormirana(h,razmerje,L,T,lamda,a,b)
#%%                    Skok
h,razmerje,L,T,lamda,a,b=0.01,5,30,10,0.5,0,10
PSI1,tau=SkokSPropagatorjem(h,razmerje,L,T,lamda,a,b,10)
Lamb=[lamda for i in range(int(T/tau))]
N,M=int(T/tau),int(L/h)
#                      Implic
tau=(h**2)*1/razmerje
#PSI1,Lamb=ImplicitnaVklop(h,tau,L,T,lamda,a,b,2)
N,M=int(T/tau),int(L/h)

 #%%###########################################################################################################
#
#       Tu rišem rezultate in delam videe
#
###########################################################################################################

plt.title('Časovna evolucija $|\psi(x)|^2$ (Skok10), $\lambda={}, a={}, N=0$'.format(lamda,a))
X,Y=np.meshgrid([L/2-i*h for i in range(M+1)],[i*tau*1000 for i in range(int(len(PSI1)/1000))])
plt.contourf(X,Y,[np.abs(PSI1[i*1000]) for i in range(int(len(PSI1)/1000))],50,colormap='hot')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
#%%
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X,Y=np.meshgrid([L/2-i*h for i in range(M+1)],[i*tau for i in range(len(PSI))])
ax.plot_wireframe(X,Y,[np.abs(PSI[i]) for i in range(len(PSI))],cmap=cm.coolwarm)
#%%
plt.plot(np.abs(PSI[0]))
#%%
from matplotlib import animation


fig = plt.figure()
ax = plt.axes(xlim=(-L/2, L/2), ylim=(-0.025, 3))
#line, = ax.plot([], [],'-', color='black', lw=2,label='Implic')
line1, = ax.plot([], [],'-', color='black', lw=2,label='Implic')
lineV, = ax.plot([], [],color='r', lw=0.5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
lamb_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
plt.title('Časovna evolucija za h=0.03, razmerje=10 $\lambda={},a={}, N=0$'.format(lamda,a))
plt.xlabel('x')
plt.ylabel('$|\psi(x)|^{2}$')
xV=[L/2-100*m*0.001 for m in range(1000)]
plt.legend(loc=1)

def init():
    lineV.set_data([],[])
#    line.set_data([], [])
    line1.set_data([], [])
    time_text.set_text('')
    lamb_text.set_text('')
    return line1, time_text, lineV

hitrost=100
def animate(ru):
    turt=ru*hitrost
    x=[L/2-m*h for m in range(M+1)]
#    y=np.multiply(PSI1[turt],np.conj(PSI1[turt]))
    y1=np.multiply(PSI1[turt],np.conj(PSI1[turt]))
    yV=[(L/2-100*m*0.001-a)**2/2+Lamb[turt]*(L/2-100*m*0.001-a)**4 for m in np.arange(1000)]
    print(turt*tau)
#    line.set_data(x, y)
    line1.set_data(x, y1)

    lineV.set_data(xV,yV)
    time_text.set_text('čas = {0:.5f}'.format(turt*tau))
    lamb_text.set_text('$\lambda = {0:.5f}$'.format(Lamb[turt]))
    return lineV, line1, time_text, lamb_text,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(len(PSI1)/hitrost), interval=1, blit=True)



#anim.save('b10_2.mp4',fps=100)
#plt.show()

#%%###########################################################################################################
#
#       TU DELAM GLOBALCE
#
###########################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#-----------------------------------------------------------------------------
# določimo dolžino korakov
S2,Lamda,Tau=[],[],[]
h,razmerje,L,T,a,b,K=0.2,50,10,10,1,0,10
for lamda in np.arange(0,2,0.02):
    PSI,tau=SkokSPropagatorjem(h,razmerje,L,T,lamda,a,b,K)
    S2.append(PSI)
    Lamda.append(lamda)
    Tau.append(tau)
#-----------------------------------------------------------------------------
#%%
from matplotlib import animation

fig,ax = plt.subplots()
hitrost=1
def animate(o):
       i=o*hitrost
       ax.clear()
       X,Y=np.meshgrid([L/2-i*h for i in range(int(L/h)+1)],[100*j*Tau[i] for j in range(int(len(S2[i])/100))])
       ax.contourf(X,Y,[np.abs(S2[i][100*j]) for j in range(int(len(S2[i])/100))],100)
       
       plt.title('Rešitev v odvsnosti od $\lambda$ (Skok) za $a={}$'.format(a))
       ax.text(0.02, 0.95, '$\lambda={}$'.format(Lamda[i]), transform=ax.transAxes)
       plt.xlabel('x')
       plt.ylabel('t')


ani = animation.FuncAnimation(fig,animate,int(len(Lamda)/hitrost),interval=1,blit=False)
ani.save('resitev_a1_2.mp4',fps=5)
plt.show()



#%%












