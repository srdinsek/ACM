import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand,randint,normal,uniform
import numba
from numba import jit
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
from winsound import Beep


@jit(nopython=True)
def Locljivi_delci(N,M,lamda,DOLZINA,epsilon,beta):
    ''' N je število delcev, med tem ko je M število propagatorjev
        to je nek splošen primer, ki ga je dal za zgled na začetku.
        V resnici je naloga lažja, saj imamo samo en delec. Lahko pa morda
        posplošim kasneje na več delcev.'''
        
    baza=rand(M,N)
    E=np.zeros(DOLZINA)

    energija=0
    for i in np.arange(M):
        razlika=np.subtract(baza[(i+1) % M],baza[i])
        energija=energija+np.dot(razlika,razlika)+lamda*np.dot(baza[i],baza[i])/2
    for i in np.arange(DOLZINA):
        j=randint(0,M)
        novi=np.multiply(rand(N),epsilon)
        
        razlika=np.subtract(baza[(i+1) % M],novi)
        razlika2=np.subtract(novi,baza[j-1])
        E_novi=np.dot(razlika,razlika)+lamda*np.dot(novi,novi)/2+np.dot(razlika2,razlika2)
        
        razlika=np.subtract(baza[(i+1) % M],baza[j])
        razlika2=np.subtract(baza[j],baza[j-1])
        E_stari=np.dot(razlika,razlika)+lamda*np.dot(baza[j],baza[j])/2+np.dot(razlika2,razlika2)
        dE=E_novi-E_stari
        
        if dE<0:
            E[i]=dE
            baza[j]=novi
        elif rand()<np.exp(-beta*dE):
            E[i]=dE
            baza[j]=novi
        
    return baza,E,energija

@jit(nopython=True)
def Nerazlocljiv_delec(DOLZINA,beta,razmerje,delez,meja,lamda,mu):
    epsilon=0.3
    M=int(beta*razmerje)
    if M<=2:
        print('Napačni parametri!')
    else:
        MM=M
        if beta<=1:
            epsilon=np.sqrt(beta)
            MM=M*100
        
        
        
        
        baza=normal(0,1,M)
        E=np.zeros(DOLZINA)
        energija=0
        
        
        arg=np.subtract(baza,np.roll(baza,1))
        gugu=np.multiply(baza,baza)
        E0=razmerje*np.dot(arg,arg)/2+mu*np.dot(baza,baza)/(2*razmerje)+lamda*np.dot(gugu,gugu)/razmerje
        
        
        K,V,t,stevec,x,tt=0,0,0,0,0,0
#        baza_x=np.zeros((int((DOLZINA-meja)/(10*M)),M))
        X,S=np.zeros(int((DOLZINA-meja)/(10*M))),np.zeros(int((DOLZINA-meja)/(10*M)))
        velika=np.zeros(M)
        baza_x=0
        print(int((DOLZINA-meja)/(10*M)))
        for i in np.arange(DOLZINA):
            j=randint(0,M)
            q=baza[j]+uniform(-epsilon,epsilon)#normal(0,epsilon)
            
            D=q-baza[j]
            d=q+baza[j]
            D4=q**4-baza[j]**4
            dE=razmerje*(D)*(d-baza[(j+1) % M] - baza[j-1])+mu*D*d/(2*razmerje)+lamda*D4/razmerje
            
            if dE<0: baza[j],E[i],stevec=q,dE,stevec+1
            elif rand()<np.exp(-dE): baza[j],E[i],stevec=q,dE,stevec+1
            
            
            epsilon=epsilon+(stevec/(i+1)-delez)/MM
            
            if i>meja and i%(M*10)==0:
                arg=np.subtract(baza,np.roll(baza,1))
                K=K-razmerje*np.dot(arg,arg)/(2*beta)
                gugu=np.multiply(baza,baza)
                V=V+mu*np.dot(baza,baza)/(2*M)+lamda*np.dot(gugu,gugu)/M
                x=x+np.sum(baza)/M
#                baza_x[t,:]=baza
#                velika=np.add(velika,np.multiply(baza[0],baza))
                t=t+1
                X[t]=x/t
                S[t]=i
        
        weird=np.divide(velika,t)
        return baza,E,E0,V/t,razmerje/2+K/t,razmerje/2+(K+V)/t,x/t,baza_x,epsilon,stevec/DOLZINA,np.divide(velika,t),-np.multiply(razmerje,np.log(np.divide(weird,np.roll(weird,1)))),X,S


def TESTER(mreza,DOLZINA,beta,razmerje,meja,lamda,mu):
    d=len(mreza)
    S=np.zeros(d)
    for i in np.arange(d):
        a,b,c,d,e,f,g,h,rg,hz=Nerazlocljiv_delec(DOLZINA,beta,razmerje,mreza[i],meja,lamda,mu)
        S[i]=f
        print(i)
        print(f)
    return mreza,S


def BETA(mreza,DOLZINA,razmerje,delez,meja,lamda,mu):
    D=len(mreza)
    S,T,V=np.zeros(D),np.zeros(D),np.zeros(D)
    for i in np.arange(D):
        a,b,c,d,e,f,g,h,rg,hz=Nerazlocljiv_delec(DOLZINA,mreza[i],razmerje,delez,meja,lamda,mu)
        S[i]=f
        T[i],V[i]=d,e
        print(i)
        print(f,rg,hz)
    return mreza,S,T,V


#%%
plt.figure('Valovna funkcija')
import seaborn as sns
sns.set_palette(sns.color_palette("hot", 8))

#vse=Nerazlocljiv_delec(3*10**7,100,10,0.7,10**6)

hogo=[vse[-3][i][j] for i in range(len(vse[-3])-1) for j in range(len(vse[-3][i]))]
n=np.histogram(hogo, 100)
norma=sum([n[0][i]*(n[1][i+1]-n[1][i]) for i in range(len(n[0]))])
weights = [1/norma for i in range(len(hogo))]
gu=plt.hist(hogo, 100,weights=weights,histtype='step',label='$\\lambda=0.1$ $\\mu=-0.5$')
print(sum([gu[0][i]*(gu[1][i+1]-gu[1][i]) for i in range(len(gu[0]))]))

hogo=[vse1[-3][i][j] for i in range(len(vse1[-3])-1) for j in range(len(vse1[-3][i]))]
n=np.histogram(hogo, 100)
norma=sum([n[0][i]*(n[1][i+1]-n[1][i]) for i in range(len(n[0]))])
weights = [1/norma for i in range(len(hogo))]
gu=plt.hist(hogo, 100,weights=weights,histtype='step',label='$\\lambda=0.1$ $\\mu=-1$')
print(sum([gu[0][i]*(gu[1][i+1]-gu[1][i]) for i in range(len(gu[0]))]))

hogo=[vse2[-3][i][j] for i in range(len(vse2[-3])-1) for j in range(len(vse2[-3][i]))]
n=np.histogram(hogo, 100)
norma=sum([n[0][i]*(n[1][i+1]-n[1][i]) for i in range(len(n[0]))])
weights = [1/norma for i in range(len(hogo))]
gu=plt.hist(hogo, 100,weights=weights,histtype='step',label='$\\lambda=0.1$ $\\mu=-2$')
print(sum([gu[0][i]*(gu[1][i+1]-gu[1][i]) for i in range(len(gu[0]))]))

hogo=[vse3[-3][i][j] for i in range(len(vse3[-3])-1) for j in range(len(vse3[-3][i]))]
n=np.histogram(hogo, 100)
norma=sum([n[0][i]*(n[1][i+1]-n[1][i]) for i in range(len(n[0]))])
weights = [1/norma for i in range(len(hogo))]
gu=plt.hist(hogo, 100,weights=weights,histtype='step',label='$\\lambda=0.1$ $\\mu=-3$')
print(sum([gu[0][i]*(gu[1][i+1]-gu[1][i]) for i in range(len(gu[0]))]))

hogo=[vse4[-3][i][j] for i in range(len(vse4[-3])-1) for j in range(len(vse4[-3][i]))]
n=np.histogram(hogo, 100)
norma=sum([n[0][i]*(n[1][i+1]-n[1][i]) for i in range(len(n[0]))])
weights = [1/norma for i in range(len(hogo))]
gu=plt.hist(hogo, 100,weights=weights,histtype='step',label='$\\lambda=0.1$ $\\mu=-4$')
print(sum([gu[0][i]*(gu[1][i+1]-gu[1][i]) for i in range(len(gu[0]))]))



plt.plot(a[-1],a[-2],color='black')
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
plt.title('$\\beta=100$ M/$\\beta$=10 koraki=$3 10^{8}$ razmerje=0.6 $\\mu=-4$ $\\lambda=0.1$',fontsize='14')
plt.xlabel('N',fontsize='14')
plt.ylabel('$\\langle x\\rangle$',fontsize='14')
plt.draw()
plt.pause(1)
plt.legend()
plt.xscale('log')
#
#vse 100
#vse2 1000
# vse3 10
#vse4 1
#vse5 0.3

#%%
#fig, ax = plt.subplots() # create a new figure with a default 111 subplot
#  
#
#ax.plot(a[0],np.abs((a[1]-0.5)/0.5),color='black')
#plt.xlabel('Delež sprejetih potez',fontsize='14')
#plt.ylabel('Napaka |E-0.5|/0.5',fontsize='14')
#plt.title('Koraki=$10^{8}$ $\\beta=10000$ M/$\\beta$=10',fontsize='14')
#plt.yscale('log')



#import seaborn as sns
#sns.set_palette(sns.color_palette("autumn", 4000))
#t=np.arange(-10,10,0.01)
#
#for a in np.arange(-1,1,0.1):
#    for b in np.arange(0,1,0.1):
#        for c in np.arange(-1,1,0.1):
#            y=[c*np.exp(2*b*i)/(2*b)+0.1 for i in t]
#            x=[a*np.exp(b*i) for i in t]
#        
#    
#            plt.plot(x,y)

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

def LastnaFunkcija(N,x):
    return float(np.multiply(np.multiply(Decimal(hermval(x,[0 if d!=N else 1 for d in range(N+1)])),np.multiply(Decimal(np.exp(-Decimal(x)**2/Decimal(2))),np.sqrt(np.reciprocal(np.multiply(Decimal(math.pow(2,N)),Decimal(math.factorial(N))))))),np.reciprocal(pi4)))


LastnoStanje=np.vectorize(LastnaFunkcija)




#%%
import seaborn as sns
sns.set_palette(sns.color_palette("hot", 18))

buz=["0p001","0p01","0p1","1","10","100"]
for j in range(len(buz)):
    buba=np.loadtxt("Lastne_vrednosti_za_N5000_lambda_"+buz[j]+".txt")
    def povpE(beta,energije):
        E=np.exp(np.multiply(-beta,energije))
        Z=np.dot(energije,E)/np.sum(E)
        return Z
    np.vectorize(povpE)
    y=np.arange(0.3,1000,0.1)
    plt.plot(y,[povpE(i,buba) for i in y],label='$\lambda$='+buz[j])
    if j<=3:
        plt.plot(x[j][0],x[j][1],'.')
    else:
        plt.plot(xx[j-4][0],xx[j-4][1],'.')
plt.legend()


plt.plot(lam,X,'.')
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
plt.title('$\\beta=10$,$M/\\beta$=10 $\\mu=1$ r=0.6 korak=$3 10^{7}$',fontsize='14')
plt.xlabel('$\\lambda$',fontsize='14')
plt.ylabel('Energija',fontsize='14')
plt.draw()
plt.pause(1)
plt.legend(loc=7)
plt.xscale('symlog')
plt.yscale('symlog')

