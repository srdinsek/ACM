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
from scipy.special import eval_hermite as herm
from decimal import Decimal
pi4=Decimal(pi**(1/4))

from numpy.polynomial.hermite import hermval

#%%###########################################################################################################
#
#       2.1. Definicija lastne funkcije
#
###########################################################################################################
# tu je bil zaplet
# moral sem koreniti ulomek, ker koren imenovalca ne deluje.
# zdaj dela povsem lepo, vendar se ne da vektorizirati...
# tale s eni obnesla, eksponent je treba dati ven iz korena!!!! poglej spodaj
# def LastnaFunkcija(N,x):
#    return herm(N,x)*np.sqrt(np.multiply(np.exp(-x**2),np.reciprocal(np.multiply(pow(2,N),math.factorial(N)))))/(pi4)

# no ja, dela povsem lepo in pravilno točno do N=156, od tam dalje vrne konstanto
def LastnaFunkcija(N,x):
    return float(np.multiply(np.multiply(Decimal(hermval(x,[0 if d!=N else 1 for d in range(N+1)])),np.multiply(Decimal(np.exp(-Decimal(x)**2/Decimal(2))),np.sqrt(np.reciprocal(np.multiply(Decimal(math.pow(2,N)),Decimal(math.factorial(N))))))),np.reciprocal(pi4)))

# to milim da mi bo prišlo še prav
# na podlagi članka https://hackernoon.com/speeding-up-your-code-2-vectorizing-the-loops-with-numpy-e380e939bed3
# sem se že pred časom prepričal, da bo verjetno to pohitrilo operacijo
# sem sedaj to preveril in ugootivl, da to ni res. To je peecej vseeno
# kar se vidi na grafu vektorizacija na območju L=-100,100
LastnoStanje=np.vectorize(LastnaFunkcija)
#%%###########################################################################################################
#
#       2.2. Diagonalizacija hamiltonjana
#
###########################################################################################################

def H0(n):
    return n+0.5

# Odločil sem se, da bom za izračun skalarnega produkta uporabil metodo np.trapz
# Za to me je prepričal članek https://stackoverflow.com/a/33230017
# ampak bom metodo še malo preveril

def H(N,L,dx,lamda):
    # Najprej bomo vedno pri računanju naredili x-os
    # L priporočam okoli 15
    x=np.arange(-L,L,dx)
    x4=np.power(x,4)
    
    # potem bom že od začetka skonstruiral N lastnih funkcij na tej mreži, da jih ne bom klical vsakič posebi
    # tako bom zahtevnost tega koraka zmanjšal iz N^2 na N.
    # število stanj je N
    PHI=np.array([LastnoStanje(n,x) for n in range(N)])
    A=[]
    for i in range(N):
        a=[]
        for j in range(N):
            if i==j:
                a.append(H0(i)+lamda*np.trapz(np.multiply(PHI[i],np.multiply(x4,PHI[j])),x,dx))
            elif j>i:
                a.append(lamda*np.trapz(np.multiply(PHI[i],np.multiply(x4,PHI[j])),x,dx))
            else:
                a.append(A[j][i])
        A.append(a)
    # ker so PHI realne jih ne konjugiram, sicer bi moral poskrbeti še za to.
    # in zato je tudi matrika simetrična
    return np.array(A)
#%%
Dd=[]
for l in [0,0.01,0.05,0.1,0.2,0.5,1,10]:
    d=[]
    for n in [179,180]:
        Hami=H(n,25,0.14,l)
        d.append(np.linalg.eigh(Hami))
    Dd.append(d)

#%%
PP=[LastnoStanje(i,np.arange(-20,20,0.01)) for i in range(180)]
#%%
n=20
l=3
colors = pl.cm.rainbow(np.linspace(0,1,6))
Lum=[0,0.01,0.05,0.1,0.2,0.5,1,10]
for l in [0,3,-2,-1]:
    gu=np.sum(np.multiply(Dd[l][1][1][i][n],PP[i]) for i in range(len(PP)))
    plt.plot(np.multiply(gu,np.conj(gu)),color=colors[l],label='$\lambda={}$'.format(Lum[l]))
    
plt.legend()
plt.xlabel('x')
plt.ylabel('$|\psi(x)|^{2}$')
plt.title('Lastne funkcije za 3 lastno stanje')
#%%###########################################################################################################
#
#       2.3. Določanje r
#
###########################################################################################################

# iz slikic je očitno da je energija že določena z  E_N^0
# potem je ves problem le še v računanju integrala. Možnost bi bil monte-carlo
# jaz bom uporabil kar trapz
    
def r(N,lamda,epsilon):
    h0=H0(N)
    x=np.linspace(-np.sqrt(2*h0),np.sqrt(2*h0),10000)
    povE=2*np.trapz([np.sqrt(2*(h0-epsilon-0.5*f**2-lamda*f**4)) for f in x if h0-0.5*f**2-lamda*f**4 >=0],[f for f in x if h0-0.5*f**2-lamda*f**4 >=0])
    pov0=2*pi*h0
    return povE/pov0, povE/(2*pi)
r=np.vectorize(r)

#%%###########################################################################################################
#
#       2.4. Časovni razvoj
#
###########################################################################################################

# pri tej metodi privzamem idealen L=25, dx=0.14, N=180
# upošteval bom lasnte energije za r(lamda)[1]-5 ali pa r(lamda)[1]-3
# začetni je array like in je seznam koeficientov v razvoju po osnovnih stanjih
# tegale malo opustimo ane
def Časovni(začetni,lamda,t,dt):
    L,dx,N=25,0.14,180
    x=np.arange(-L,L,dx)
    PP=[LastnoStanje(g,x) for g in range(N)]
    PSI0=np.sum(PP[g]*začetni[g] for g in range(len(začetni)))
    E1,psi1=np.linalg.eigh(H(N,L,dx,lamda))
    PSI1=np.array([np.sum(PP[g]*psi1[h][g] for g in range(N)) for h in range(N)])
    Ne=int(r(N,lamda,0.01)[1])-5
    Q=[]
    if Ne!=0:
        koef=[np.dot(psi1[i],začetni) for i in range(N)]
        for tau in np.arange(0,t,dt):
            a=sum(np.multiply(PSI1[i],koef[i]*np.exp(-1j*E1[i]*tau)) for i in range(Ne))
            Q.append(a)
        return Q

# začetni je tu zapisan v |x>
def ČasovniX(začetni,L,dx,lamda,t,dt):
    N=180
    x=np.arange(-L,L,dx)
    if lamda!=0:
        print(0)
        Ne=int(r(N,lamda,0.01)[1])-5
    else:
        print(1)
        Ne=N
    PP=np.array([LastnoStanje(g,x) for g in range(Ne)])
    
    E1,psi1=np.linalg.eigh(H(N,25,0.1,lamda))

    PSI1=np.array([np.sum(np.multiply(PP[g],psi1[g][h]) for g in range(Ne)) for h in range(Ne)])
    Q=[]
    koef=[np.trapz(np.multiply(PSI1[i],začetni),x,dx) for i in range(Ne)]
    for tau in np.arange(0,t,dt):
        a=sum(np.multiply(PSI1[i],koef[i]*np.exp(-1j*E1[i]*tau)) for i in range(Ne))
        Q.append(a)
    return Q


#%%
#A=Časovni([1 if i==0 else 0 for i in range(180)],0.1,10,0.1)
L=20
dx=0.01
l=0.1
t=500
dt=0.05
A=ČasovniX(LastnoStanje(10,[x for x in np.arange(-L,L,dx)]),L,dx,l,t,dt)
#%%
h=0.1
razmerje=10
L=10
T=10
lamda=0.1
a=-1
b=0
n=5
K=10
B,tau=SkokSPropagatorjem(h,razmerje,L,T,lamda,a,b,n,K)
#%%
plt.figure(1)
#X,Y=np.meshgrid(np.linspace(-L/2,L/2,int(L/h)+1),np.arange(0,T,tau))
#B1=np.abs([B[i*100] for i in np.arange(int(len(B)/100))])
#A1=np.abs([A[i*100] for i in np.arange(int(len(A)/100))])

X,Y=np.meshgrid(np.arange(-L,L,dx),np.arange(0,t,dt))
plt.contourf(X,Y,np.abs(np.add(np.abs(B),np.multiply(-1,np.abs(A)))),50)
plt.colorbar()
plt.title('Rešitev pri a=1, razlika med rešitvama, n=5, $\lambda=0.1$')
plt.xlabel('x')
plt.ylabel('t')
#%%
plt.figure(0)
X,Y=np.meshgrid(np.arange(-L,L,dx),np.arange(0,t,dt))
#B=[[A[i][j]*np.conj(A[i][j]) for j in range(len(A[0]))] for i in range(len(A))]
plt.contourf(X,Y,np.abs(A),50)
plt.colorbar()
plt.title('Rešitev pri a=10, Razvoj po bazi, n=0, $\lambda=0.01$ s dx=0.01')
plt.xlabel('x')
plt.ylabel('t')
#%%
M=int(L/h)
    V=[(L/2-m*h-a)**2/2+lamda*(L/2-m*h-a)**4 for m in np.arange(M+1)]
    PSI=[]
    c=LastnaFunkcija(n,(L/2-b))
    PSI.append([LastnaFunkcija(n,(L/2-m*h-b))-c for m in np.arange(M+1)])

#%%
# potem bom pogledal še kako metoda ohranja unitarnost.
x=np.arange(-L,L,dx)
U=[np.abs(np.trapz(np.multiply(A[i],np.conj(A[i])),x,dx)-1) for i in range(len(A))]

plt.plot([i*0.05 for i in range(len(A))],U)
plt.xlabel('t[s]')
plt.ylabel('$\log(||\psi(x)|^2-1|)$')
plt.title('Ohranjanje norme dx=0.01 n=10, $\lambda=0.1$, a=0')

#%%###########################################################################################################
#
#       2.5. Metoda Lanczoseva
#
###########################################################################################################

def HH(L,dx,lamda):
    # tole sem samo vse prekopiral iz prejšnje naloge in priredil
    h2=dx**2
    luk=len(np.arange(-L,L,dx))
    d=[(1/h2+0.5*x**2+lamda*x**4) for x in np.arange(-L,L,dx)]
    a=[(-1/(2*h2)) for x in range(luk-1)]
    b=[(-1/(2*h2)) for x in range(luk-1)]
    A=np.diag(d,0)+np.diag(a,1)+np.diag(b,-1)
    return A

from scipy.sparse import diags
def LL(L,dx,lamda,N):
    # pokličemo H in krajevne koordinate
    Ham=HH(L,dx,lamda)
    x=np.arange(-L,L,dx)
    diag=[]
    izdag=[]
    
    # začnemo algoritem, prvo stanje
    PSI=[]
    psi=sum(LastnoStanje(n,x) for n in range(1))
    C=np.sqrt(np.trapz(np.multiply(psi,psi),x,dx))
    print(C)
    PSI.append(np.multiply(psi,1/C)) 
    
    # izračunamo drugo stanje
    psi=np.dot(Ham,PSI[0])
    alfa=np.trapz(np.multiply(PSI[0],psi),x,dx)
    diag.append(alfa)
    psi=psi-np.multiply(alfa,PSI[0])
    C=np.sqrt(np.trapz(np.multiply(psi,psi),x,dx))
    PSI.append(np.multiply(psi,1/C))
    
    # poženemo zanko za izračun preostalih stanj
    for j in range(1,N):
        a=np.dot(Ham,PSI[j])
        d=np.trapz(np.multiply(PSI[j],a),x,dx)
        diag.append(d)
        iz=np.trapz(np.multiply(PSI[j-1],a),x,dx)
        izdag.append(iz)
        psi=a-np.multiply(d,PSI[j])-np.multiply(iz,PSI[j-1])
        C=np.sqrt(np.trapz(np.multiply(psi,psi),x,dx))
        PSI.append(np.multiply(psi,1/C))
    return PSI, diags([diag,izdag,izdag],[0,-1,1])

#np.diag(diag,0)+np.diag(izdag,1)+np.diag(izdag,-1)


#%%#%%###########################################################################################################
#
#       2.6. Metoda Lanczoseva če vzamem H od 2.3.
#
###########################################################################################################

def kreacijskiH(N,lamb):
    a=np.diag([np.sqrt(i+1) for i in range(N-1)],1)+np.diag([np.sqrt(i+1) for i in range(N-1)],-1)
    x=np.dot(a,np.dot(a,np.dot(a,a)))
    return np.multiply(1/4*lamb,x)+np.diag([i+0.5 for i in range(N)],0)



def Lancz(začetni,lamda,N):
    Ham=kreacijskiH(N,lamda)
    #Ham=H(N,L,dx,lamda)
    diag=[]
    izdag=[]
    
    # začnemo algoritem, prvo stanje
    PSI=[]
    psi=začetni
    #[0 if n!=0 else 1 for n in range(N)]
    #[1 for i in range(N)]
    C=np.sqrt(np.dot(psi,psi))
    psi=np.multiply(psi,1/C)
    PSI.append(psi)
    
    
    # izračunamo drugo stanje
    psi=np.dot(Ham,PSI[0])
    alfa=np.dot(PSI[0],psi)
    diag.append(alfa)
    psi=psi-np.multiply(alfa,PSI[0])
    C=np.sqrt(np.dot(psi,psi))
    PSI.append(np.multiply(psi,1/C))
    # poženemo zanko za izračun preostalih stanj
    for j in range(1,N):
        a=np.dot(Ham,PSI[j])
        d=np.dot(PSI[j],a)
        diag.append(d)
        iz=np.dot(PSI[j-1],a)
        izdag.append(iz)
        psi=a-np.multiply(d,PSI[j])-np.multiply(iz,PSI[j-1])
        C=np.sqrt(np.dot(psi,psi))
        PSI.append(np.multiply(psi,1/C))
    return PSI, np.diag(diag,0)+np.diag(izdag,1)+np.diag(izdag,-1)


#diags([diag,izdag,izdag],[0,-1,1])

#%%
#HUGO=HamiltonianLL(20,10,0.01,0)
PSI,HUGO=Lancz(0.001,10)
#PSI,HUGO=LL(25,0.1,0.1,100)
#%%
HUGOO=H(180,25,0.1,0.001)
#%%
HUGOO=kreacijskiH(2000,0.001)

#%%
x=np.arange(-10,10,0.01)
baza=np.array([LastnoStanje(n,x) for n in range(100)])
#%%
# riši
plt.plot(np.sum(np.multiply(baza[i],PSI[-1][i]) for i in range(20)))
#%%
for t in range(3):
    if t%1==0:
        plt.plot(PSI[t],label='N={}'.format(t))
plt.legend()
#%%
from scipy.sparse.linalg import eigs
#from scipy.linalg import eigh_tridiagonal
#A=np.linalg.eigh(HUGO)
A=eigs(HUGO,k=6,which='SM')
#A=eigh_tridiagonal(HUGO,select='a')
B=np.linalg.eigh(HUGOO)
plt.plot(np.sort(A[0]),'.',label='Lancz_x, $\lambda$=0.1')
#plt.plot(D0)
#plt.plot(D11)
plt.plot(B[0],'.',label='H, $\lambda$=0.1')
plt.legend()
#%%
lamb=100
HUGOO=kreacijskiH(5000,lamb)
B=np.linalg.eigh(HUGOO)
BUU=[B[0][i] for i in range(int(r(5000,lamb,0.001)[1])-10)]
N=[i for i in range(int(r(5000,lamb,0.001)[1])-10)]

np.savetxt('Lastne_vrednosti_za_N5000_lambda_100.txt', BUU)

#%%
L=25
dx=0.1
x=np.arange(-L,L,dx)

AB=[[np.trapz(np.multiply(PSI[i],PSI[j]),x,dx) for j in range(len(PSI))] for i in range(len(PSI))]
#%%
L=10
dx=0.001
x=np.arange(-L,L,dx)

AB=[[np.dot(PSI[i],PSI[j]) for j in range(len(PSI))] for i in range(len(PSI))]

#%%
plt.imshow(np.abs(AB),cmap='hot_r')
plt.colorbar()

#%%%
lamda=0.1
spekter=[]
vektori=[]
N=5
NN=100
for n in range(N,NN):
    PSI,HUGO=Lancz([1 for i in range(n)],lamda,n)
    A=np.linalg.eigh(HUGO)
    spekter.append(A[0])
    vektori.append(PSI)
#%%
import matplotlib.pylab as pl


colors = pl.cm.rainbow(np.linspace(0,1,E-e+5))
e=20
E=50
N=5
numb=E-N+1
for i in range(e,E):
    plt.plot([n for n in range(N+numb,NN)],[np.abs(spekter[n][i]-B[0][i]) for n in range(numb,NN-N)],color=colors[i-e],label='$E_{}$'.format(i))

plt.legend()
plt.xlabel('N')
plt.ylabel('$\log(E_{N}-E_{moj})$')
plt.title('Razlika energij med Lancz in prejšnjo za $\lambda=0.001$')

#%% malo večji sistem
N=5
NN=100
BUU=[]
for i in range(NN-N):
    burka=[np.abs(spekter[n][i]-B[0][i]) if n>=i+N else 0 for n in range(0,NN-N) ]
    BUU.append(burka)
#%%
plt.imshow(np.log(BUU))
plt.colorbar()
plt.xlabel('Velikost matrike N')
plt.ylabel('i-ta Lastna energija')
plt.title('Odstopanje od pričakovane vrednosti $\lambda=0.1$ homogen')




#%%
# ta algoritem spodaj je bil spremenjen po nasvetu iz članka
# http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf, na 6. strani.
# mislim da je potrebno spremeniti vrstni red normiranja, da stvar steče kot je treba
# Q=PSI,    q=psi,    r=a,   
def LL2(L,dx,lamda,N):
    # pokličemo H in krajevne koordinate
    Ham=HH(L,dx,lamda)
    x=np.arange(-L,L,dx)
    diag=[]
    izdag=[]
    
    # začnemo algoritem, prvo stanje
    Q=[]
    q=sum(LastnoStanje(n,x) for n in range(1))
    C=np.sqrt(np.trapz(np.multiply(q,q),x,dx))
    q=np.multiply(q,1/C)
    Q.append(q) 
    
    #-------------------------------------------------------- do tu je isto kot na linku
    # izračunamo drugo stanje
    r=np.dot(Ham,q)
    alfa=np.trapz(np.multiply(q,r),x,dx)
    diag.append(alfa)
    r=r-np.multiply(alfa,q)
    beta=np.sqrt(np.trapz(np.multiply(r,r),x,dx))
    izdag.append(beta)
    
    #PSI.append(np.multiply(psi,1/C))
    
    # poženemo zanko za izračun preostalih stanj
    for j in range(1,N):
        v=q
        q=np.multiply(r,1/izdag[j-1])
        Q.append(q)
        r=np.dot(Ham,Q[-1])-np.multiply(izdag[j-1],v)
        alfa=np.trapz(np.multiply(Q[-1],r),x,dx)
        diag.append(alfa)
        r=r-np.multiply(alfa,Q[-1])
        beta=np.sqrt(np.trapz(np.multiply(r,r),x,dx))
        izdag.append(beta)
        if beta==0:
            print('šit')
            break
    return Q, np.diag(diag,0)+np.diag([izdag[i] for i in range(N-1)],1)+np.diag([izdag[i] for i in range(N-1)],-1)
#%%###########################################################################################################
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
    tau=2*pi/(np.linalg.norm(A,np.inf)*razmerja)
    return np.sum((-1j*tau)**k*np.linalg.matrix_power(A,k)/math.factorial(k) for k in range(K+1)),tau

def SkokSPropagatorjem(h,razmerje,L,T,lamda,a,b,n,K):
    M=np.arange(-L,L,h)
    V=[(x-a)**2/2+lamda*(x-a)**4 for x in M]
    PSI=[]
    c=LastnaFunkcija(n,(L-b))
    PSI.append([LastnaFunkcija(n,(x-b))-c for x in M])
    H,tau=ExpH(len(M)-1,V,K,h**2,razmerje)    
    N=int(T/tau)
    for n in np.arange(N):
        psi=np.dot(H,PSI[-1])
        PSI.append(psi)
    return PSI,tau

