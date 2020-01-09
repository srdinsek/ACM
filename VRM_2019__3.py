import math
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.linalg import expm
from scipy.integrate import odeint
from scipy.integrate import ode
from winsound import Beep
import matplotlib.pylab as pl

pi=np.pi
#%%###########################################################################################################
#
#       3.1. Definicije integratorjev
#
###########################################################################################################

def T(x,tau):
    return [x[0]+x[2]*tau,x[1]+x[3]*tau,x[2],x[3]]


#x=[x,y,px,py]
def V(x,tau,lamda):
    return [x[0],x[1],x[2]-(x[0]+2*lamda*x[0]*x[1]*np.conj(x[1]))*tau,x[3]-(x[1]+2*lamda*x[0]*np.conj(x[0])*x[1])*tau]


# Ta algoritem zabeleži tudi vmesne korake!
# To je enostaven Trotterjev alg.
def Trotter(x,t,dt,lamda):
    N=int(t/dt)
    X=np.zeros((N+1,4))
    X[0]=x
    for i in range(N):
        x=V(x,dt,lamda)
        x=T(x,dt)
        X[i+1]=x
    return X


# ta NE zabeleži vmesnih
# To je simetriziran Trotterjev
def TrotterSym(x,t,dt,lamda):
    N=int(t/dt)
    X=np.zeros((N+1,4))
    X[0]=x
    for i in range(N):
        x=S2(x,lamda,dt)
        X[i+1]=x
    return X


# ta NE zabeleži vmesnih
def S2(x,lamda,dt):
    x=T(x,dt/2)
    x=V(x,dt,lamda)
    x=T(x,dt/2)
    return x


def S3(x,lamda,p):
    x=T(x,p[4])
    x=V(x,p[3],lamda)
    x=T(x,p[2])
    x=V(x,p[1],lamda)
    x=T(x,p[0])
    return x


# Tole je četrtega reda
def SS4(x,t,dt,lamda):
    N=int(t/dt)
    X=np.zeros((N+1,4))
    X[0]=x
    dt0=-2**(1/3)/(2-2**(1/3))*dt
    dt1=1/(2-2**(1/3))*dt
    for i in range(N):
        x=S2(x,lamda,dt1)
        x=S2(x,lamda,dt0)
        x=S2(x,lamda,dt1)
        X[i+1]=x
    return X


# tole je tretjega reda brez slaloma
def SS3(x,t,dt,lamda):
    N=int(t/dt)
    X=np.zeros((N+1,4))
    X[0]=x
    p1=0.25*(1+1j/np.sqrt(3))*dt
    p5=np.conj(p1)
    p2=2*p1
    p4=np.conj(p2)
    p3=0.5*dt
    p=np.array([p1,p2,p3,p4,p5])
    for i in range(N):
        x=S3(x,lamda,p)
        X[i+1]=x
    return X


# tole je tretjega reda slalomski
def SS33(x,t,dt,lamda):
    N=int(t/dt)
    dt=dt/2
    X=np.zeros((N+1,4))
    X[0]=x
    p1=0.25*(1+1j/np.sqrt(3))*dt
    p5=np.conj(p1)
    p2=2*p1
    p4=2*np.conj(p1)
    p3=0.5*dt
    p=np.array([p1,p2,p3,p4,p5])
    for i in range(N):
        x=S3(x,lamda,np.conj(p))
        x=S3(x,lamda,p)
        X[i+1]=x
    return X



#(A,x,tri,dt,lamda,a,b,c,k)
#nabla=[x,y,p_x,p_y]
def f(t,nabla,arg):
    #-------------------------------------------------------------------------------------------------------------------------------------
    x_t=nabla[0]+2*arg*nabla[0]*nabla[1]*nabla[1]
    y_t=nabla[1]+2*arg*nabla[1]*nabla[0]*nabla[0]
    #-------------------------------------------------------------------------------------------------------------------------------------
    return [nabla[2],nabla[3],-x_t,-y_t]


#rešimo sistem enačb
def RungeKutta(y0,t1,dt,lamda):
    rezultat=[]
    r = ode(f).set_integrator('dopri5')
    r.set_initial_value(y0, 0).set_f_params(lamda)
    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
        rezultat.append(np.real(r.y))
    g=np.array(rezultat)
    return g[:,0],g[:,1],g[:,2],g[:,3]

#%%###########################################################################################################
#
#       3.2. Funkcija za izračun poincaréjeve preslikave (ki vmes tudi izriše preslikavo!)
#
###########################################################################################################


def SS4P(x,t,dt,lamda,Nmax):
    N=int(t/dt)
    X=[x]
    dt0=-2**(1/3)/(2-2**(1/3))*dt
    dt1=1/(2-2**(1/3))*dt
    for i in range(N):
        x1=x
        x2=S2(x1,lamda,dt1)
        x3=S2(x2,lamda,dt0)
        x=S2(x3,lamda,dt1)
        if (x[1]>0 and x1[1]<0) or x[1]==0:
            delta=-x1[1]*dt/(x[1]-x1[1])
            iks=delta*(x[0]-x1[0])/dt+x1[0]
            iks_od=delta*(x[2]-x1[2])/dt+x1[2]
            y_od=delta*(x[3]-x1[3])/dt+x1[3]
            X.append(np.array([iks,0,iks_od,y_od]))
        if len(X)==Nmax:
            X=np.array(X)
            return X[:,0],X[:,1],X[:,2],X[:,3]
    X=np.array(X)
    return X[:,0],X[:,1],X[:,2],X[:,3]



# to je funkcija, ki vrne točke na poincaréjevi preslikavi
def Poincaré(E,lamda,Tmax,dp,dx,dt,Nmax):
    print('\nPognali ste funkcijo, ki bo vrnila točke za Poincaréjevo presliakvo.')
    print('Parametri so:\n E={},\n lamda={}, \n Tmax={},\n dp={},\n dx={},\n dt={},\n Nmax={}'.format(E,lamda,Tmax,dp,dx,dt,Nmax))
    start=timer()
    globalcx,globalcx_od=[],[]
    
    # tu začnemo zanko
    pp=np.sqrt(2*E)
    GRU=len(np.arange(-pp,pp,dp))
    colore=pl.cm.rainbow(np.linspace(0,1,GRU))
    i=0
    for px in np.arange(-pp,pp,dp):
        xx=np.sqrt(pp**2-px**2)
        grafox_od,grafox=[],[]
        print('Korak po px osi je',i, 'od', GRU, 'korakov.')
        for x in np.arange(-xx,xx,dx):
            
            py=np.sqrt(xx**2-x**2)
            #------------------------------------------------------------
            začasnix,g,začasnix_od,gu=sezam=SS4P([x,0,px,py],Tmax,dt,lamda,Nmax) # glavni del, poglej zgornjo funkcijo
            #------------------------------------------------------------
            plt.plot(začasnix,začasnix_od,'.',color=colore[i%GRU],markersize=0.5)
            grafox.append(začasnix)
            grafox_od.append(začasnix_od)
        i=i+1
        globalcx.append(grafox)
        globalcx_od.append(grafox_od)
    Beep(780, 1000)
    print('\nFunkcija je za izračun porabila {} min'.format((timer()-start)/60))
    plt.xlabel('x')
    plt.ylabel('$p_{x}$')
    plt.title('$\lambda={}, E={}$'.format(lamda,E))
    plt.axis('equal')
    return globalcx,globalcx_od


#%%###########################################################################################################
#
#       3.3. Funkcija za izračun razlike <p>-<p>
#
###########################################################################################################

def ekvipart(t,dt,lamda,x,px,tcut):
    Int=[]
    Ncut=int(tcut/dt)
    for l in lamda:
        print('$\lambda$:',l)
        start=timer()
        x,px=0.3,0.2
        py=np.sqrt(2*0.625-px**2-x**2)
        X=np.array(SS4([x,0,px,py],t,dt,l))
        
        integral=[]
        vsota1=0
        vsota2=0
        for i in range(len(X)):
            vsota1=vsota1+X[i][2]**2*dt
            vsota2=vsota2+X[i][3]**2*dt
            if i > Ncut:
                integral.append(vsota1/(i*dt)-vsota2/(i*dt))
        Int.append(integral)
        end=timer()
        print('Čas tega koraka znaša:',end-start,'s.')
    return Int
#%%###########################################################################################################
#
#       3.4. Izriše <p>-<p> za različne \lambda
#
###########################################################################################################


t=100000
dt=0.1
Int1,Int2=[],[]
for lamda in np.arange(6,12,0.3):
    print(lamda)
    start=timer()
#    x,px=0,0.5
#    py=np.sqrt(2*0.625-px**2-x**2)
#    X=np.array(SS4([x,0,px,py],t,dt,lamda))
    
    X=np.array(SS4([0.25,0,0.73,np.sqrt(2*0.625-0.25**2-0.73**2)],t,dt,lamda))
    
    integral1,integral2=[],[]
    vsota1=0
    vsota2=0
    for i in range(len(X)):
        vsota1=vsota1+X[i][2]**2*dt
        vsota2=vsota2+X[i][3]**2*dt
        integral1.append(vsota1/(i*dt))
        integral2.append(vsota2/(i*dt))
    Int1.append(integral1)
    Int2.append(integral2)
    end=timer()
    print(end-start)

plt.title('$x_{0}=(0.25,0,0.73,0.8091)$')
TT=[i*dt*1000 for i in range(int(len(Int1[0])/1000))]
LA=[lamda for lamda in np.arange(6,12,0.3)]
colore=pl.cm.rainbow(np.linspace(0,1,len(LA)))
for lamda in range(len(LA)):
    plt.plot(TT,np.abs(np.subtract([Int1[lamda][g*1000] for g in range(int(len(Int1[0])/1000))],[Int2[lamda][g*1000] for g in range(int(len(Int1[0])/1000))])),color=colore[lamda],label='$\lambda={}$'.format(LA[lamda]))
plt.xlabel('t')
plt.ylabel('|$(1/t)\int_{0}^{t}  p_{x}(t´)^{2}dt´$-$(1/t)\int_{0}^{t}  p_{y}(t´)^{2}dt´$|')
plt.legend()

#%%###########################################################################################################
#
#       3.5. Riše fazni prostor glede na to katri lambda je kritičen.
#
###########################################################################################################



t=10000
dt=0.1

fa=np.sqrt(2*0.625)
gogo=[]
r=0
for px in np.linspace(-fa,fa,40):
    iksovi=[]
    print('Nahajam se pri',r,'-tem koraku.')
    start=timer()
    for x in np.linspace(0,fa,40):
        if 2*0.625-px**2-x**2 >= 0:
            py=np.sqrt(2*0.625-px**2-x**2)
            Int=[]
            for lamda in np.arange(1,5,0.5):
                X=np.array(Trotter([x,0,px,px],t,dt,lamda))
                
                integral=[]
                vsota1=0
                vsota2=0
                for i in range(len(X)):
                    vsota1=vsota1+X[i][2]**2*dt
                    vsota2=vsota2+X[i][3]**2*dt
                    integral.append(vsota1/(i*dt)-vsota2/(i*dt))
                Int.append(integral)
        else:
            Int=[]
            for lamda in np.arange(1,5,0.5):
                integral=[]
                for i in range(int(t/dt)):
                    integral.append(1)
                Int.append(integral)
        iksovi.append(Int)
    print('Traja že:',timer()-start,'s')
    r=r+1
    gogo.append(iksovi)

#%%
G=np.array(gogo)
VA=[]
for z in range(len(np.linspace(-fa,fa,20))):
    a=[]
    for r in range(len(np.linspace(0,fa,20))):
        A=np.array([np.abs(G[z][r][i]) for i in range(len(G[z][r]))])
        A=np.array([np.amax(A[i,9000:]) for i in range(len(A))])
        a.append(A)
    VA.append(a)
#%%
Buta=[]
L=[lamda for lamda in np.arange(1,5,0.5)]
for i in range(len(np.linspace(-fa,fa,20))):
    buta=[]
    for j in range(len(np.linspace(0,fa,20))):
        for gu in range(len(VA[i][j])):
            if VA[i][j][gu] < 0.05 and VA[i][j][gu]!=0:
                buta.append(L[gu])
                break
            elif VA[i][j][gu]==1:
                buta.append(np.log(-1))
                break
        else:
            print('lol')
            buta.append(L[-1])
    Buta.append(buta)
#%%
from matplotlib import cm
plt.imshow(Buta,extenT=(0,2*fa,fa,-fa),cmap=cm.jet)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('$p_{x}$')
plt.title('$\lambda$ ergodičnosti, y=0, H=0.625 meja 0.05 9000:')





#%%###########################################################################################################
#
#       3.6. Tu rišem videe, nazadnje Poincaré
#
###########################################################################################################



from matplotlib import animation
A,B=-1.2,1.2

fig = plt.figure()
ax = plt.axes(xlim=(A, B), ylim=(-1.2,1.2))
line, = ax.plot([], [], '.',color='black', markersize=0.5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
plt.title('Poincaré pri E=0.625 in $y=0$')
plt.xlabel('x')
plt.ylabel('$p_{x}$')

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

#DU=np.array([SS4([0.25,0.4,0,np.sqrt(2*0.625-0.4**2-0.25**2)],1000,0.1,i) for i in np.arange(0,4,0.01)])
def animate(ru):
    x,y=Ax[ru],Ay[ru]
    line.set_data(x, y)
    if ru<len(np.arange(0,4,0.0266)):
        time_text.set_text('$\lambda$ = {0:.2f}'.format(np.arange(0,4,0.0266)[ru]))
    elif ru>=len(np.arange(0,4,0.0266)) and ru< len(np.arange(4,5,0.0266))+len(np.arange(0,4,0.0266)):
        time_text.set_text('$\lambda$ = {0:.2f}'.format(np.arange(4,5,0.0266)[ru-len(np.arange(0,4,0.0266))]))
    else:
        time_text.set_text('$\lambda$ = {0:.2f}'.format(np.arange(5,6,0.0266)[ru-len(np.arange(0,4,0.0266))-len(np.arange(4,5,0.0266))]))
    return line, time_text

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(Ax), interval=100, blit=True)


#anim.save('poincare2.mp4')
#plt.show()  


def PoincaréVIDEO(E,lamda,Tmax,dp,dx,dt,Nmax):
    start=timer()
    grafox_od,grafox=[],[]
    
    # tu začnemo zanko
    pp=np.sqrt(2*E)
    GRU=len(np.linspace(-pp,pp,dp))
    colore=pl.cm.rainbow(np.linspace(0,1,GRU))
    i=0
    for px in np.linspace(-pp,pp,dp):
        xx=np.sqrt(pp**2-px**2)
#        print('Korak po px osi je',i, 'od', GRU, 'korakov.')
        for x in np.linspace(-xx,xx,dx):
            py=np.sqrt(xx**2-x**2)
            #------------------------------------------------------------
            začasnix,g,začasnix_od,gu=sezam=SS4P([x,0,px,py],Tmax,dt,lamda,Nmax) # glavni del, poglej zgornjo funkcijo
            #------------------------------------------------------------
            grafox.append(začasnix)
            grafox_od.append(začasnix_od)
        i=i+1
    print('\nFunkcija je za izračun porabila {} min'.format((timer()-start)/60))
    print('\n$\lambda$= {}'.format(lamda))
    return grafox, grafox_od
#%%
#  Tu računam poincaré ob različnih $\lambda$


for lamda in np.arange(5,6,0.0266):
    print('Korak po $$\lambda$$ je',int(lamda/0.0266), 'od 37 korakov.')
    A=PoincaréVIDEO(0.625,lamda,500,20,20,0.1,1000)
    Ax.append([A[0][i][j] for i in range(len(A[0])) for j in range(len(A[0][i]))])
    Ay.append([A[1][i][j] for i in range(len(A[1])) for j in range(len(A[1][i]))])
Beep(780, 1000)
    
#%%   
plt.plot(Ax[-50],Ay[-50],'.',markersize=0.5)

#%%











