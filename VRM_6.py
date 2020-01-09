import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand,randint
import numba
from numba import jit
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
from winsound import Beep

# to je samo skalarni produkt v polarnih koordinatah
@jit(nopython=True)
def skalar(phi1,phi2,th1,th2):
    return -np.sin(phi1)*np.sin(phi2)*np.cos(th1-th2)+np.cos(phi1)*np.cos(phi2)

#tole je metropolis, ki vrne celotno zgodovino (dve matriki, eno s phi, eno s theta koti)
@jit(nopython=True)
def heisenberg_full(N,koraki,J,h,beta):
    a=rand(2*N)
    THETA,PHI=np.zeros((int((koraki+1)/3*N),N)),np.zeros((int((koraki+1)/3*N),N))
    theta=np.multiply(2*np.pi,a[:N])
    phi=np.arccos(np.multiply(2,a[N:])-1)
    PHI[0,:],THETA[0,:]=phi,theta
    """v zgornjih dveh vrsticah smo že skonstruirali začetno verigo.
    Spin na n-tem mestu je enostavno theta[n],phi[n]
    shranimo samo to na vakem koraku, ne rabimo poračunait vseh elementov.
    """
    
    for i in np.arange(koraki):
        mesto=randint(0,N)
        x,y=np.arccos(2*rand()-1),2*np.pi*rand()
        p,th,p1,th1,pm1,thm1=phi[mesto],theta[mesto],phi[(mesto+1)%N],theta[(mesto+1)%N],phi[(mesto-1)%N],theta[(mesto-1)%N]
        novo=J*(skalar(x,p1,y,th1)+skalar(x,pm1,y,thm1))+h*np.cos(x)
        staro=J*(skalar(p,p1,th,th1)+skalar(p,pm1,th,thm1))+h*np.cos(p)
        
        dE=(novo-staro)
        if dE<0: phi[mesto],theta[mesto]=x,y
        elif rand()<np.exp(-beta*dE): phi[mesto],theta[mesto]=x,y
        
        if i%(N*3)==0:
            PHI[i+1,:],THETA[i+1,:]=phi,theta
    return PHI,THETA

#tole je metropolis ki vrne samo končno stanje
# tu sem tudi popravil napako zaradi zamenjave phi in theta
# tu sem tudi popravil napako ker sem pozabil žrebati po sferi v zanki
@jit(nopython=True)
def heisenberg(N,koraki,J,h,beta,spp):
    a=rand(2*N)
    theta=np.multiply(2*np.pi,a[:N])
    phi=np.arccos(np.multiply(2,a[N:])-1)
    """v zgornjih dveh vrsticah smo že skonstruirali začetno verigo.
    Spin na n-tem mestu je enostavno theta[n],phi[n]
    shranimo samo to na vakem koraku, ne rabimo poračunait vseh elementov.
    """
    E=np.zeros(koraki)
    skuzi=np.add(np.multiply(J,skalar(phi,np.roll(phi,1),theta,np.roll(theta,1))),np.multiply(h,np.cos(phi)))
    E0=np.sum(skuzi)
    
    spin=np.zeros(N)
    for i in np.arange(koraki):
        mesto=randint(0,N)
        x,y=np.arccos(2*rand()-1),2*np.pi*rand()
        p,th,p1,th1,pm1,thm1=phi[mesto],theta[mesto],phi[(mesto+1)%N],theta[(mesto+1)%N],phi[(mesto-1)%N],theta[(mesto-1)%N]
        novo=J*(skalar(x,p1,y,th1)+skalar(x,pm1,y,thm1))+h*np.cos(x)
        staro=J*(skalar(p,p1,th,th1)+skalar(p,pm1,th,thm1))+h*np.cos(p)
        
        dE=(novo-staro)
        if dE<0: phi[mesto],theta[mesto],E[i]=x,y,dE
        elif rand()<np.exp(-beta*dE): phi[mesto],theta[mesto],E[i]=x,y,dE
        
        if i>spp:
            spin=np.add(spin,np.multiply(np.cos(phi[0]),np.cos(phi)))
        
    return phi,theta,E,E0,np.divide(spin,koraki-spp)

@jit(nopython=True)
def heisenberg_brez_E(N,koraki,Jx,Jy,Jz,h,beta,spp):
    
    a=rand(2*N)
    theta=np.multiply(2*np.pi,a[:N])
    phi=np.arccos(np.multiply(2,a[N:])-1)
    
    X=np.sin(phi)*np.cos(theta)
    Y=np.sin(phi)*np.sin(theta)
    Z=np.cos(phi)
    """v zgornjih dveh vrsticah smo že skonstruirali začetno verigo.
    Spin na n-tem mestu je enostavno theta[n],phi[n]
    shranimo samo to na vakem koraku, ne rabimo poračunait vseh elementov.
    """
    
    spin=np.zeros(N)
    gg=0
    for i in np.arange(koraki):
        mesto=randint(0,N)
        p,t=np.arccos(2*rand()-1),2*np.pi*rand()
        
        ss=np.sin(p)
        x,y,z=ss*np.cos(t),ss*np.sin(t),np.cos(p)
        xx,yy,zz=x-X[mesto],y-Y[mesto],z-Z[mesto]
        
        dE=-(Jx*xx*(X[(mesto+1)%N]+X[(mesto-1)%N])+Jy*yy*(Y[(mesto+1)%N]+Y[(mesto-1)%N])+Jz*zz*(Z[(mesto+1)%N]+Z[(mesto-1)%N]))+h*zz

        if dE<0: X[mesto],Y[mesto],Z[mesto]=x,y,z
        elif rand()<np.exp(-beta*dE): X[mesto],Y[mesto],Z[mesto]=x,y,z
        
        
        if i%(N*2)==0:
            gg=gg+1
            for j in np.arange(N):
                spin[j]=spin[j]+np.dot(Z,np.roll(Z,j))/N
        
    return X,Y,Z,np.divide(spin,gg)

@jit(nopython=True)
def heisenbergMag_brez_E(N,koraki,J,h,beta,spp):
    '''to je narejeno samo za prave začetne pogoje! 4*N'''
    
    R=np.zeros((N,3))
    for hugo in np.arange(N):
        if hugo%4==0:
            R[hugo][2]=1
        elif hugo%4==1:
            R[hugo][0]=1
        elif hugo%4==2:
            R[hugo][2]=1
        else: R[hugo][0]=-1
        
    """v zgornjih vrsticah smo že skonstruirali začetno verigo."""
    
    spin=np.zeros(N)
    gg=0
    
    for i in np.arange(koraki):
        mesto=randint(0,N)
        ''' vsakič bom pogledal desnega, in najprej poračunam M-vstoto teh dveh vektorjev'''
        R1,R2=R[mesto-1],R[mesto]
        k=np.add(R1,R2)
        norma=np.sqrt(np.dot(k,k))
        kn=np.divide(k,norma)
        
        kot=2*np.pi*rand()
        a=np.cos(kot/2)
        hod=np.sin(kot/2)
        
        b,c,d=kn[0]*hod,kn[1]*hod,kn[2]*hod
        a2,b2,c2,d2=a*a,b*b,c*c,d*d
        bc,ad,bd,ac,cd,ab=2*b*c,2*a*d,2*b*d,2*a*c,2*c*d,2*a*b
        
        x1=(a2+b2-c2-d2)*R1[0]+(bc-ad)*R1[1]+(bd+ac)*R1[2]
        y1=(bc+ad)*R1[0]+(a2+c2-b2-d2)*R1[1]+(cd-ab)*R1[2]
        z1=(bd-ac)*R1[0]+(cd+ab)*R1[1]+(a2+d2-b2-c2)*R1[2]
        X1=np.array([x1,y1,z1])
        X2=np.subtract(k,X1)
        
        XX1,XX2=np.subtract(X1,R1),np.subtract(X2,R2)
        
        dE=-J*(np.dot(R[mesto-2],XX1)+np.dot(X1,X2)-np.dot(R1,R2)+np.dot(XX2,R[(mesto+1)%N]))+h*(XX1[2]+XX2[2])

        if dE<0: R[mesto-1],R[mesto]=X1,X2
        elif rand()<np.exp(-beta*dE): R[mesto-1],R[mesto]=X1,X2
        
        
        if i%(N*10)==0 and i>1000000:
            gg=gg+1
            Z=R[:,2]
            for j in np.arange(N):
                spin[j]=spin[j]+np.dot(Z,np.roll(Z,j))/N
        
    return R[:,0],R[:,1],R[:,2],np.divide(spin,gg)

def faznik(N,koraki,J,h,betaray):
    
    for beta in betaray:
        x,y,E,E0=heisenberg(N,koraki,J,h,beta)
        toko=np.zeros(koraki)
        oko=0
        for i in np.arange(koraki):
            oko=oko+E[i]
            toko[i]=oko
    return N

#%%
import seaborn as sns
sns.set_palette(sns.color_palette("brg", 8))
#%%
start=timer()
korak=10**8
beta=10
for beta in [0,0.1,1,4,6,10,20]:
    x,y,z,s=heisenberg_brez_E(500,korak,-1,-1,-1,0.01,beta,0)
#    #Beep(500,500)
#    oko=E0
#    toko=[]
#    for i in E:
#        oko=oko+i
#        toko.append(oko)
#    
    a=timer()-start
    print('numba:',a)
#    
#    
#    
#    plt.figure('energija')
#    plt.plot([i*100 for i in range(int(len(toko)/100))],[toko[i*100] for i in range(int(len(toko)/100))],'-',label='$\\beta$={}'.format(beta))
#    
#    
#    ax = plt.subplot(111)
#    ax.get_xaxis().tick_bottom()  
#    ax.get_yaxis().tick_left()
#    plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
#    from matplotlib.font_manager import FontProperties
#    font0 = FontProperties()
#    font = font0.copy()
#    font.set_style('italic')
#    font.set_size('x-large')
#    font.set_family('serif')
#    plt.title('Energija N=50 J=1 h=0 korak=$10^{7}$',fontsize='14')
#    plt.xlabel('korak N',fontsize='14')
#    plt.ylabel('energija',fontsize='14')
#    plt.legend()

#start=timer()
#xx,yy=heisenberg_full(500,1000000,1,10,1)
#b=timer()-start3
#print('numba_vsi:',b)
#print('razmerje:',b/a)

    plt.figure('spini')
    plt.plot(np.arange(-250,250,1),np.roll(s,250),'-',label='$\\beta$={}'.format(beta))
    
    ax = plt.subplot(111)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
    from matplotlib.font_manager import FontProperties
    font0 = FontProperties()
    font = font0.copy()
    font.set_style('italic')
    font.set_size('x-large')
    font.set_family('serif')
    plt.title('$\\langle \\sigma_{0}^{z}\\sigma_{r}^{z}\\rangle$ N=500 J=(-1,-1,-1) h=0.01 korak=$10^{8}$',fontsize='14')
    plt.xlabel('r',fontsize='14')
    plt.ylabel('C(r)',fontsize='14')
    plt.legend()
#%%
#%%

korak=10**7
from matplotlib.font_manager import FontProperties
ssss=[]
for h in [0,0.0001,0.01,0.1,1,10]:
    ss=[]
    print('izbrani h je:',h)
    start=timer()
    for beta in [0,0.1,1,4,6,8,10,15,20]:
        p,t,ff,s=heisenberg_brez_E(50,korak,-1,h,beta,0)
        ss.append(s)
        a=timer()-start
        print('numba:',a)
    ssss.append(ss)
#%%
from scipy.optimize import curve_fit

def func(x, a, c, d):
    return a*np.exp(-c*x)+d


h=[0,0.0001,0.01,0.1,1,10]
sns.set_palette(sns.color_palette("brg", 10))
plt.figure('spini')
for j in range(len(h)):
    gr=[]
    for i in range(9):
        popt, pcov = curve_fit(func, np.arange(25), np.abs(ssss[j][i][:25]),p0=(0.3, 2, 0.1))
        gr.append(popt[1])
    plt.plot([0,0.1,1,4,6,8,10,15,20],np.divide(1,gr),marker='.',label='h={}'.format(h[j]))
#    plt.plot([0,0.1,1,4,6,10,20],[ssss[j][i][0] for i in range(len(ssss[j]))],'-')

ax = plt.subplot(111)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
font0 = FontProperties()
font = font0.copy()
font.set_style('italic')
font.set_size('x-large')
font.set_family('serif')
plt.title('tipična korelacijska dolžina od h N=50 J=-1 korak=$10^{7}$',fontsize='14')
plt.xlabel('$\\beta$',fontsize='14')
plt.ylabel('korelacijska dolžina',fontsize='14')
plt.legend()

#%%
korak=10**8
beta=4
#x,y,z,s=heisenbergMag_brez_E(200,korak,1,0,beta,0)
x,y,z,s=heisenberg_brez_E(400,korak,0,1,1,0,beta,0)

for i in [-1]:
    vectors=np.array( [ [i,0,0,x[i],y[i],z[i]] for i in range(len(y))])
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')
    for vector in vectors:
        v = np.array([vector[3],vector[4],vector[5]])
        vlength=np.linalg.norm(v)
        ax.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
                pivot='tail', normalize=True,color='black')
    ax.set_xlim([-1,401])
    ax.set_ylim([-2,2])
    ax.set_zlim([-1,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Rešitev pri N=400 J=(0,1,1) h=0 $\\beta$=4 korak=$10^{8}$ $\\vec{M}=0$',fontsize='14')
    plt.show()


#%%
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

num_frames = len(xx)
N=len(xx[0])
X, Y, Z= [i for i in range(N)],[0 for i in range(N)],[0 for i in range(N)]


def compute_segs(j):
    N=len(xx[0])
    i=j*1000
    X, Y, Z= [ii for ii in range(N)],[0 for ii in range(N)],[0 for ii in range(N)]

    U = np.add(np.multiply(np.sin(xx[i]),np.cos(yy[i])),X)
    V = np.multiply(np.sin(xx[i]),np.sin(yy[i]))
    W = np.cos(xx[i])

    return X,Y,Z,U,V,W


segs = compute_segs(0)
quivers = ax.quiver(*segs, length=1, pivot='tail', color='black',arrow_length_ratio=0.3, normalize=True)


ax.set_xlim([-1,501])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])
def animate(i):

    segs = np.array(compute_segs(i)).reshape(6,-1)
    new_segs = [[[x,y,z],[u,v,w]] for x,y,z,u,v,w in zip(*segs.tolist())]
    quivers.set_segments(new_segs)
    return quivers


ani = FuncAnimation(fig, animate, frames = int(len(xx)/1000), interval = 1, blit=False)
#ani.save('update_3d_quiver.gif', writer='imagemagick')

plt.show()
