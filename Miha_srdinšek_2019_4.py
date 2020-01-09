import math
import numpy as np
import matplotlib.pyplot as plt
from random import random
from numpy.random import normal

###########################################################################################################
#
#       Tu definiram vse funkcije
#
###########################################################################################################

from numba import jit
@jit(nopython=True)
def sodi(seznam):
    N,z=2,1
    L=2**N
    začetno=np.zeros(L,dtype=np.complex)
    el=np.exp(z,dtype=np.complex)
    elm=np.exp(-z,dtype=np.complex)
    sh=elm*np.sinh(2*z,dtype=np.complex)
    ch=elm*np.cosh(2*z,dtype=np.complex)
     
    for i in np.arange(L):
        naša=bin(i)[2:].zfill(N)
        #-------------------------------------------------
        # tu samo naredimo korak za j=0
        a=naša[-2::1]
        if a=='00' or a=='11': začetno[i]=el*seznam[i]
        elif a=='01': začetno[i]=ch*seznam[i]+sh*seznam[i^3]
        elif a=='10': začetno[i]=ch*seznam[i]+sh*seznam[i^3]
        #-------------------------------------------------
    
    #-------------------------------------------------
    # tu pa naredimo korake za preostale j
    for j in np.arange(2,N,2):
        seznam=np.array([i for i in začetno])
        for i in np.arange(L):
            naša=bin(i)[2:].zfill(N)
            a=naša[-(j+2):-(j)]
            if a=='00' or a=='11': začetno[i]=el*seznam[i]
            elif a=='01': 
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]=ch*seznam[i]+sh*seznam[n]
            elif a=='10':
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]=ch*seznam[i]+sh*seznam[n]
    #-------------------------------------------------
    return začetno

def lihi(N,z,seznam):
    
    L=2**N
    začetno=np.zeros(L,dtype=np.complex)
    el=np.exp(z)
    elm=np.exp(-z)
    sh=elm*np.sinh(2*z)
    ch=elm*np.cosh(2*z)
    huuk=2**0+2**(N-1)
    for i in range(L):
        naša=bin(i)[2:].zfill(N)
        #-------------------------------------------------
        # tu samo naredimo korak za j=N
        a=naša[-1]+naša[0]
        if a=='00' or a=='11': začetno[i]=el*seznam[i]
        elif a=='01': začetno[i]=ch*seznam[i]+sh*seznam[i^huuk]
        elif a=='10': začetno[i]=ch*seznam[i]+sh*seznam[i^huuk]
        #-------------------------------------------------
        
    #-------------------------------------------------
    # tu pa naredimo korake za preostale j
    for j in np.arange(1,N-1,2):
        seznam=[i for i in začetno]
        for i in range(L):
            naša=bin(i)[2:].zfill(N)
            a=naša[-(j+2):-(j)]
            if a=='00' or a=='11': začetno[i]=el*seznam[i]
            elif a=='01': 
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]=ch*seznam[i]+sh*seznam[n]
            elif a=='10':
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]=ch*seznam[i]+sh*seznam[n]
        #-------------------------------------------------
    return začetno

def H(N,seznam):
    L=2**N
    začetno=np.zeros(L,dtype=np.complex)
    huuk=2**0+2**(N-1)
    for i in range(L):
        naša=bin(i)[2:].zfill(N)
        #-------------------------------------------------
        # tu samo naredimo korak za j=N
        a=naša[-1]+naša[0]
        if a=='00' or a=='11': začetno[i]+=seznam[i]
        elif a=='01': začetno[i]+=-seznam[i]+2*seznam[i^huuk]
        elif a=='10': začetno[i]+=-seznam[i]+2*seznam[i^huuk]
        #-------------------------------------------------
    
    for i in range(L):
        naša=bin(i)[2:].zfill(N)
        #-------------------------------------------------
        # tu samo naredimo korak za j=0
        a=naša[-2::1]
        if a=='00' or a=='11': začetno[i]+=seznam[i]
        elif a=='01': začetno[i]+=-seznam[i]+2*seznam[i^3]
        elif a=='10': začetno[i]+=-seznam[i]+2*seznam[i^3]
        #-------------------------------------------------
    
    #-------------------------------------------------
    # tu pa naredimo korake za preostale j
    for j in np.arange(1,N-1,1):
        for i in range(L):
            naša=bin(i)[2:].zfill(N)
            a=naša[-(j+2):-(j)]
            if a=='00' or a=='11': začetno[i]+=seznam[i]
            elif a=='01': 
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]+=-seznam[i]+2*seznam[n]
            elif a=='10':
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]+=-seznam[i]+2*seznam[n]
    #-------------------------------------------------
    return začetno
    
def sigmaZ(N,seznam,j):
    L=2**N
    začetno=np.zeros(L,dtype=np.complex)
    
    for i in range(L):
        naša=bin(i)[2:].zfill(N)
        a=naša[-j-1]
        začetno[i]=(-1)**int(a)*seznam[i]
    
    return začetno

def spinski_tok(N,seznam):
    L=2**N
    začetno=np.zeros(L,dtype=np.complex)
    huuk=2**0+2**(N-1)

    for i in range(L):
        naša=bin(i)[2:].zfill(N)
        #-------------------------------------------------
        # tu samo naredimo korak za j=N
        a=naša[-1]+naša[0]
        if a=='01': začetno[i]+=-2j*seznam[i^huuk]
        elif a=='10': začetno[i]+= 2j*seznam[i^huuk]
        #-------------------------------------------------
        
    for i in range(L):
        naša=bin(i)[2:].zfill(N)
        #-------------------------------------------------
        # tu samo naredimo korak za j=0
        a=naša[-2::1]
        if a=='01': začetno[i]+=-2j*seznam[i^3]
        elif a=='10': začetno[i]+= 2j*seznam[i^3]
        #-------------------------------------------------
    
    #-------------------------------------------------
    # tu pa naredimo korake za preostale j
    for j in np.arange(1,N-1,1):
        for i in range(L):
            naša=bin(i)[2:].zfill(N)
            a=naša[-(j+2):-(j)]
            if a=='01': 
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]+=-2j*seznam[n]
            elif a=='10':
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]+= 2j*seznam[n]
    #-------------------------------------------------
    return začetno

def S2(x,db,N):
    x=lihi(N,-db/2,x)
    x=sodi(N,-db,x)
    x=lihi(N,-db/2,x)
    return x

def S3(x,p,N):
    x=lihi(N,-p[4],x)
    x=sodi(N,-p[3],x)
    x=lihi(N,-p[2],x)
    x=sodi(N,-p[1],x)
    x=lihi(N,-p[0],x)
    return x

def Trotter(x,b,db,N):
    n=int(abs(b/db))
    X=[x]
    for i in range(n):
        x=lihi(N,-db,x)
        x=sodi(N,-db,x)
        X.append(x)
    return X

def TrotterSym(x,b,db,N):
    n=int(abs(b/db))
    X=[x]
    for i in range(n):
        print('znotraj:',i)
        x=S2(x,db,N)
        X.append(x)
    return X

# Tole je četrtega reda
def SS4(x,b,db,N):
    n=int(abs(b/db))
    db0=-2**(1/3)/(2-2**(1/3))*db
    db1=1/(2-2**(1/3))*db
    X=[x]
    for i in range(n):
        x=S2(x,db1,N)
        x=S2(x,db0,N)
        x=S2(x,db1,N)
        X.append(x)
    return X

# tole je tretjega reda slalomski
def SS33(x,b,db,N):
    n=int(abs(b/(db*2)))
    p1=0.25*(1+1j/np.sqrt(3))*db
    p5=np.conj(p1)
    p2=2*p1
    p4=2*np.conj(p1)
    p3=0.5*db
    p=np.array([p1,p2,p3,p4,p5])
    X=[x]
    for i in range(n):
        x=S3(x,np.conj(p),N)
        x=S3(x,p,N)
        X.append(x)
    return X

def Integrator(x,b,db,N,ime):
    if ime=='trot':
        return Trotter(x,b,db,N)
    if ime=='trotsym':
        return TrotterSym(x,b,db,N)
    if ime=='SS4':
        return SS4(x,b,db,N)
    if ime=='SS33':
        return SS33(x,b,db/2,N)

def Z(št,b,db,N,ime):
    print('Poklicali ste funkcijo Z'+ime+'.')
    L=2**N
    X,Y=[],[]
    for zumba in range(št):
        FX=[normal(0,1)+1j*normal(0,1) for i in range(L)]
        FX=np.divide(FX,np.sqrt(np.vdot(FX,FX)))
        novi=Integrator(FX,b/2,db/2,N,ime)
        X.append([np.vdot(novi[i],novi[i]) for i in range(len(novi))])
        kovi=[H(N,novi[i]) for i in range(len(novi))]
        Y.append([np.vdot(kovi[i],novi[i]) for i in range(len(novi))])
    Y=np.array(Y,dtype=np.complex)
    X=np.array(X,dtype=np.complex)
    return np.divide([sum(X[:,i]) for i in range(len(X[0]))],št),np.divide([sum(Y[:,i]) for i in range(len(Y[0]))],št)

def Z_Osnovni(b,db,N,ime):
    print('Poklicali ste funkcijo Z_Osnovni_'+ime+'.')
    L=2**N
    X,Y=[],[]
    for zumba in range(L):
        FX=[1 if i==zumba else 0 for i in range(L)]
        novi=Integrator(FX,b/2,db/2,N,ime)
        X.append([np.vdot(novi[i],novi[i]) for i in range(len(novi))])
        kovi=[H(N,novi[i]) for i in range(len(novi))]
        Y.append([np.vdot(novi[i],kovi[i]) for i in range(len(novi))])
    Y=np.array(Y,dtype=np.complex)
    X=np.array(X,dtype=np.complex)
    return np.divide([sum(X[:,i]) for i in range(len(X[0]))],L),np.divide([sum(Y[:,i]) for i in range(len(Y[0]))],L)

def korelacijaZ(št,t,dt,N,mesto,ime):
    print('Poklicali ste funkcijo za izracun korelacijske funkcije sigmaZ{}'.format(mesto)+' z metodo '+ime+'.')
    L=2**N
    Y=[]
    for zumba in range(št):
        print(zumba)
        FX=[normal(0,1)+1j*normal(0,1) for i in range(L)]
        FX=np.divide(FX,np.sqrt(np.vdot(FX,FX)))
        x=sigmaZ(N,FX,mesto)
        noviS=Integrator(x,-1j*t,-1j*dt,N,ime)
        novi=Integrator(FX,-1j*t,-1j*dt,N,ime)
        kovi=[sigmaZ(N,noviS[i],mesto) for i in range(len(noviS))]
        Y.append([np.vdot(novi[i],kovi[i]) for i in range(len(novi))])
    Y=np.array(Y,dtype=np.complex)
    return np.divide([sum(Y[:,i]) for i in range(len(Y[0]))],št)

def korelacijaJ(št,t,dt,N,ime):
    print('Poklicali ste funkcijo za izracun korelacijske funkcije spinski_tok z metodo '+ime+'.')
    L=2**N
    Y=[]
    for zumba in range(št):
        print(zumba)
        FX=[normal(0,1)+1j*normal(0,1) for i in range(L)]
        FX=np.divide(FX,np.sqrt(np.vdot(FX,FX)))
        x=spinski_tok(N,FX)
        print(zumba)
        noviS=Integrator(x,-1j*t,-1j*dt,N,ime)
        novi=Integrator(FX,-1j*t,-1j*dt,N,ime)
        kovi=[spinski_tok(N,noviS[i]) for i in range(len(noviS))]
        Y.append([np.vdot(novi[i],kovi[i]) for i in range(len(novi))])
    Y=np.array(Y,dtype=np.complex)
    return np.divide([sum(Y[:,i]) for i in range(len(Y[0]))],št)


###########################################################################################################
#
#       Tu potem kodo uporabljamo
#
###########################################################################################################


'''
import seaborn as sns
sns.set_palette(sns.color_palette("rainbow_r", 14))
ss=[]
db=0.01
osnova=Z_Osnovni(10,db,2,'trotsym')
B=np.arange(0,10+db,db)
for j in [1,2,3,4,5,10]:
    ss=Z(j,10,db,2,'trotsym')
    plt.plot(B,[-np.log(ss[0][i])/B[i] for i in range(len(B))],label='F št.={}'.format(j))
#    plt.plot(B,[ss[1][i]/ss[0][i] for i in range(len(B))],label='E št.={}'.format(j))

plt.plot(B,[-np.log(osnova[0][i])/B[i] for i in range(len(B))],color='black',label='F Eksaktno št.={}'.format(j))
#plt.plot(B,[osnova[1][i]/osnova[0][i] for i in range(len(B))],color='brown',label='E Eksaktno št.={}'.format(j))




#B=np.arange(0,10,0.01)
#plt.plot(B,[-np.log(3*np.exp(-2*i)+np.exp(6*i))/i for i in B],color='black',label='Analitično F')
#plt.plot(B,[6*(np.exp(-2*i)-np.exp(6*i))/(3*np.exp(-2*i)+np.exp(6*i)) for i in B],color='brown',label='Analitično E')


#import seaborn as sns
#sns.set_palette(sns.color_palette("rainbow_r", 4))
#
#T=10
#dt=0.1
#start=timer()
#kor=korelacijaZ(5,T,dt,14,0,'trotsym')
#end=timer()
#B=np.arange(0,T+dt,dt)
#plt.plot(B,kor,color='black',lw=1)
#plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
#from matplotlib.font_manager import FontProperties
#font0 = FontProperties()
#font = font0.copy()
#font.set_style('italic')
#font.set_size('x-large')
#font.set_family('serif')
#plt.title('$\langle \\sigma_{0}(t)\\sigma_{0}\\rangle$ N=14 TrotSym $dt=0.1$ št=5',fontsize='14')
#plt.xlabel('Čas t',fontsize='14')
#plt.ylabel('Korelacijska funkcija $C_{\\sigma\\sigma}(t)$',fontsize='14')
#plt.legend()


#plt.plot([2,4,6,8],cas,'.',markersize=1,color='red')
#plt.plot([2,4,6,8],cas,'-',lw=0.9,color='black')
#
#plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
#from matplotlib.font_manager import FontProperties
#font0 = FontProperties()
#font = font0.copy()
#font.set_style('italic')
#font.set_size('x-large')
#font.set_family('serif')
#plt.title('Časovna zahtevnost $C_{\sigma\\sigma}(t)$',fontsize='14')
#plt.xlabel('Velikost verige N',fontsize='14')
#plt.ylabel('Čas [s]',fontsize='14')
#plt.legend()




##%%
#N=6
#zug=[1 if i==0 else 0 for i in range(2**N)]
#zug=np.divide(zug,np.sqrt(sum(np.multiply(zug,zug))))
#gulag=lihi(N,0.1,zug)
##%%
#N=10
#zug=[random() for i in range(2**N)]
#zug=np.divide(zug,np.sqrt(sum(np.multiply(zug,zug))))
#gulag=Trotter(zug,1,0.001,N)

#%%

št=5
N=14
n=10
b=2

#odvisnost2=[]
#for N in [8,10,14]:
#    odvisnost2.append(ZSS4(št,b,b/n,N))
#odvisnost7,H7=Z(št,b,b/n,N,'trot')
odvisnost8,H8=Z(št,b,b/n,N,'trotsym')
#odvisnost9,H9=Z(št,b,b/n,N,'SS4')
#odvisnost11,H11=Z(št,b,b/n,N,'SS33')

#odvisnost1=Z_OsnovniTrot(b,b/n,N)
#odvisnost2=Z_OsnovniTrotsym(b,b/n,N)
#odvisnost3=Z_OsnovniSS4(b,b/n,N)
#odvisnost4=Z_OsnovniSS33(b,b/n,N)

#odvisnostss3=[]
#for n in [20,50,100,200,1000,2000]:
#    odvisnostss3.append(ZSS33(št,b,b/n,N))
#%%
import seaborn as sns
sns.set_palette(sns.color_palette("rainbow_r", 6))
n=[20,50,100,200,1000,2000]
for j in range(len(n)):
    B=[i for i in np.arange(0,2+2/n[j],2/n[j])]
    F=[-np.log(odvisnostss3[j][i])/B[i] for i in range(len(odvisnostss3[j]))]
    plt.plot(B,F,label='$\\beta$={}'.format(2/n[j]))
    
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()
plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_style('italic')
font.set_size('x-large')
font.set_family('serif')
plt.title('Odvisnost metode SS33 od $d\\beta$ N=6',fontsize='14')
plt.xlabel('Inverz temperature $\\beta$',fontsize='14')
plt.ylabel('Prosta energija F($\\beta$)',fontsize='14')
plt.legend()
#%%
B=[i for i in np.arange(0,2+0.2,0.2)]
import seaborn as sns
sns.set_palette(sns.color_palette("hot_r", 1))
#F7=[-np.log(odvisnost7[i])/B[i] for i in range(len(odvisnost7))]
F8=[-np.log(odvisnost8[i])/B[i] for i in range(len(odvisnost8))]
#F9=[-np.log(odvisnost9[i])/B[i] for i in range(len(odvisnost9))]
#F11=[-np.log(odvisnost11[i])/B[i] for i in range(len(odvisnost11))]

#E7=[H7[i]/odvisnost7[i] for i in range(len(H7))]
E8=[H8[i]/odvisnost8[i] for i in range(len(H8))]
#E9=[H9[i]/odvisnost9[i] for i in range(len(H9))]
#E11=[H11[i]/odvisnost11[i] for i in range(len(H11))]

#H7=[-np.diff(np.log(odvisnost7))[i]/((B[i]+np.diff(B)[i]/2)*0.1) for i in range(len(odvisnost7)-1)]
#H8=[-np.diff(np.log(odvisnost8))[i]/((B[i]+np.diff(B)[i]/2)*0.1) for i in range(len(odvisnost8)-1)]
#H9=[-np.diff(np.log(odvisnost9))[i]/((B[i]+np.diff(B)[i]/2)*0.1) for i in range(len(odvisnost9)-1)]
#H11=[-np.diff(np.log(odvisnost11))[i]/((B[i]+np.diff(B)[i]/2)*0.1) for i in range(len(odvisnost11)-1)]

#F1=[-np.log(odvisnost1[i])/B[i] for i in range(len(odvisnost1))]
#F2=[-np.log(odvisnost2[i])/B[i] for i in range(len(odvisnost2))]
#F3=[-np.log(odvisnost3[i])/B[i] for i in range(len(odvisnost3))]
#F4=[-np.log(odvisnost4[i])/B[i] for i in range(len(odvisnost4))]
#plt.plot(B,F7,label='Trot')
#plt.plot(B,F1,label='Ekza-Trot')
plt.plot(B,F8,label='TrotSym')
#plt.plot(B,F2,label='Ekza-TrotSym')
#plt.plot(B,F9,label='SS4')
#plt.plot(B,F3,label='Ekza-SS4')
#plt.plot(B,F11,label='SS33')
#plt.plot(B,F4,label='Ekza-SS33')


#plt.plot(B,E7,label='Trot')
plt.plot(B,E8,label='TrotSym')
#plt.plot(B,E9,label='SS4')
#plt.plot(B,E11,label='SS33')


#plt.plot([(B[i]+np.diff(B)[i]/2) for i in range(len(odvisnost7)-1)],H7,label='Trot')
#plt.plot([(B[i]+np.diff(B)[i]/2) for i in range(len(odvisnost8)-1)],H8,label='TrotSym')
#plt.plot([(B[i]+np.diff(B)[i]/2) for i in range(len(odvisnost9)-1)],H9,label='SS4')
#plt.plot([(B[i]+np.diff(B)[i]/2) for i in range(len(odvisnost11)-1)],H11,label='SS33')



#ax.get_xaxis().tick_bottom()  
#ax.get_yaxis().tick_left()
plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_style('italic')
font.set_size('x-large')
font.set_family('serif')
plt.title(' N=14',fontsize='14')
plt.xlabel('Inverz temperature $\\beta$',fontsize='14')
plt.ylabel('Prosta energija F($\\beta$) in energija E($\\beta$)',fontsize='14')
plt.legend()
#plt.xscale('log')
#odvisnost_osnovna6=[Z_Osnovni(b,b/n,N) for b in B]
#plt.plot(np.arange(1,0,-0.1),odvisnost,label='Približek')
#plt.plot(np.arange(1,0,-0.1),odvisnost_osnovna,label='Eksaktno')
#plt.plot(np.arange(1,0,-0.1),odvisnost2,label='Približek2')
#plt.plot(np.arange(1,0,-0.1),odvisnost_osnovna2,label='Eksaktno2') N=6


#%%


#%%
#########################################################################################
#                                                                                       #
#                                   DIFUZIJSKA KONSTANTA                                #
#                                                                                       #
#########################################################################################

# Zdaj bom za vsak N izrisal ta integral, pri čemer bom shranil ne nekaj t-jev, zanimajo me namreč veliki t
# potem bom na koncu na podlagi tega dobil graf v int(T) v odvisnosti od N za različne T.
# na koncu bom seveda uporabil le tisti rezultat pri dolgih časih.

t=10
dt=0.333333333
ime='trotsym'
št=5


for N in [20]:
    print(N)
    start=timer()
    dvajset=korelacijaJ(št,t,dt,N,ime)
    print('Korak je trajal',timer()-start,'minut.')


#%%

shramba=np.array(shramba)
Integral19=[np.trapz(shramba[0][:-1],np.arange(0,19,0.1),0.1),np.trapz(shramba[1][:-1],np.arange(0,19,0.1),0.1),np.trapz(shramba[2][:-1],np.arange(0,19,0.1),0.1),np.trapz(shramba[3][:-1],np.arange(0,19,0.1),0.1),np.trapz(shramba[4][:-1],np.arange(0,19,0.1),0.1)]

Integral19.append(np.trapz(shramba[-1][:-1],np.arange(0,19,0.2),0.2))

Integral19.append(np.trapz(šestnajs,np.arange(0,19,0.33333333333333),0.33333333333333))
#%%

Integral10=[np.trapz(shramba[0][:100],np.arange(0,10,0.1),0.1),np.trapz(shramba[1][:100],np.arange(0,10,0.1),0.1),np.trapz(shramba[2][:100],np.arange(0,10,0.1),0.1),np.trapz(shramba[3][:100],np.arange(0,10,0.1),0.1),np.trapz(shramba[4][:100],np.arange(0,10,0.1),0.1)]

Integral10.append(np.trapz(shramba[-1][:50],np.arange(0,10,0.2),0.2))

Integral10.append(np.trapz(dvajset,np.arange(0,10,0.333333333),0.333333333))
#%%
Integral19.append(np.trapz([šestnajst[i] for i in range(len(šestnajst)-1)],np.arange(0,19,0.33333333333333),0.33333333333333))



#%%
šestnajs=[32.04083901+2.77555756e-17j, 23.62378757-1.86578467e-02j,
       12.91239785-2.20705475e-02j,  7.36215916-2.81691931e-02j,
        5.50579162-2.70448678e-02j,  5.45107537-2.64384888e-03j,
        6.16722295+2.44966021e-02j,  6.82855385-4.02399788e-02j,
        6.94086482-7.50168804e-02j,  6.5409097 -7.20398164e-02j,
        5.81515496-3.91574444e-02j,  5.20983463+1.19399257e-02j,
        4.97786585-2.57324072e-02j,  5.01217014-1.08150745e-02j,
        5.08645564-1.68723656e-02j,  5.28934838-2.96975254e-02j,
        5.58488864-3.88797610e-02j,  5.8954174 -2.29469590e-02j,
        6.16741209+2.52345054e-03j,  6.36509563+1.79980938e-02j,
        6.48812159+1.61108269e-02j,  6.56470711+4.53312062e-02j,
        6.67366261+8.30716612e-02j,  6.80593635+9.21099677e-02j,
        6.88082112+9.11667836e-03j,  6.95406386-2.23734450e-03j,
        6.95025892+4.24142850e-02j,  6.99377416-1.74999256e-03j,
        7.02060036-2.60664582e-02j,  7.11395119+1.53311414e-02j,
        7.1866697 +1.91454030e-02j,  7.25216235+2.85499346e-02j,
        7.35316879+1.21123649e-02j,  7.48160884+1.71141306e-02j,
        7.48671637-2.58287475e-02j,  7.39991495-3.80900441e-02j,
        7.39275592-4.12110323e-02j,  7.38588056-5.72956385e-02j,
        7.39436233-1.13975779e-02j,  7.36061324-1.35027223e-02j,
        7.34011064+1.24522808e-02j,  7.3724662 +4.74050003e-02j,
        7.34081287+3.12570821e-02j,  7.35200709-2.41697799e-02j,
        7.36046371-6.45473934e-02j,  7.33215364-4.39862424e-02j,
        7.35323292-6.81245491e-03j,  7.40525019-3.83088059e-02j,
        7.3406302 -5.19565053e-02j,  7.25855016-4.96949689e-02j,
        7.26173846-9.78181139e-03j,  7.3384234 +1.03563057e-02j,
        7.52997706+1.98222379e-02j,  7.76372371-1.64505284e-02j,
        7.88461408-5.87143069e-02j,  7.84663108-5.47674225e-02j,
        7.83553116-4.89583715e-02j,  7.80153588-2.90207357e-02j]

dvajset2=[40.00942622+2.87964097e-17j, 29.53239678+1.06152920e-03j,
       16.16578745+6.92409952e-03j,  9.23766969-1.14367679e-02j,
        6.87930721-2.08850480e-02j,  6.74574127-1.00344994e-02j,
        7.11343063-1.12862064e-03j,  7.12707253+6.04139346e-03j,
        7.01350036+1.18745386e-02j,  6.77074031+5.83793084e-03j,
        6.53103623+2.70021205e-03j,  6.3292563 -3.80469644e-03j,
        6.20375606-3.73181685e-03j,  6.13188243-1.54219018e-02j,
        6.05501888-2.90771475e-02j,  5.96993773-1.61236231e-02j,
        5.90134484-2.30799862e-02j,  5.91139182-2.52791010e-02j,
        5.99948566-1.49506367e-02j,  6.12884418-9.03088771e-03j,
        6.26459015-1.63907937e-02j,  6.37303542-1.03860203e-02j,
        6.43963357-3.65754300e-03j,  6.48818415+3.44726141e-04j,
        6.54468003-2.68227024e-04j,  6.5961589 -8.17028498e-03j,
        6.66626606-3.30987450e-04j,  6.74173683+1.14388190e-03j,
        6.81444406+1.80799634e-04j,  6.90261405+2.96705727e-03j,
        6.98474883+1.37371893e-02j]
#%%

import seaborn as sns
sns.set_palette(sns.color_palette("rainbow_r", 8))

for T in [2,5,7,10,12,14,16,18]:
    Integralko=[]
    Integralko=[np.trapz(shramba[0][:int(T/0.1)],np.arange(0,T,0.1),0.1),np.trapz(shramba[1][:int(T/0.1)],np.arange(0,T,0.1),0.1),np.trapz(shramba[2][:int(T/0.1)],np.arange(0,T,0.1),0.1),np.trapz(shramba[3][:int(T/0.1)],np.arange(0,T,0.1),0.1),np.trapz(shramba[4][:int(T/0.1)],np.arange(0,T,0.1),0.1)]
    Integralko.append(np.trapz(shramba[-1][:int(T/0.2)],np.arange(0,T,0.2),0.2))
    Integralko.append(np.trapz(šestnajs[:int(T/0.33333333333333)+1],np.arange(0,T,0.33333333333333),0.33333333333333))
    
    
    FF=[2,4,8,10,12,14,16]
    plt.plot(FF,[Integralko[i]/FF[i] for i in range(len(FF))],'-',lw=1,label='T={}'.format(T))
#    plt.plot(FF,[Integralko[i]/FF[i] for i in range(len(FF))],'.',markersize=5,label='T={}'.format(T))
    plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
    from matplotlib.font_manager import FontProperties
    font0 = FontProperties()
    font = font0.copy()
    font.set_style('italic')
    font.set_size('x-large')
    font.set_family('serif')
    plt.title(' Kolikšna je difuzijska konstanta?',fontsize='14')
    plt.xlabel('Velikost verige N',fontsize='14')
    plt.ylabel('$\\frac{1}{N} \\int_{0}^{T}\\langle J(t)J(0)\\rangle$',fontsize='14')
    plt.xscale('log')
plt.legend()
#%%
n=[2,4,8,10,12]
import seaborn as sns
sns.set_palette(sns.color_palette("rainbow_r", 8))
gušt=[]
for N in range(len(n)):
    a=[np.trapz([shramba[N][:int(T/0.1)]],np.arange(0,T,0.1),0.1) for T in np.arange(0,18,0.5)]
    plt.plot(np.arange(0,18,0.5),a,lw=1,label='N={}'.format(n[N]))
    gušt.append(np.trapz([shramba[N][:int(15/0.1)]],np.arange(0,15,0.1),0.1))

a=[np.trapz(shramba[-1][:(int(T/0.2))],np.arange(0,T,0.2),0.2) for T in np.arange(0,18,1)]
plt.plot(np.arange(0,18,1),a,lw=1,label='N=14')
gušt.append(np.trapz(shramba[-1][:(int(15/0.2))],np.arange(0,15,0.2),0.2))

a=[np.trapz(šestnajs[:(int(T/0.33333333333333)+1)],np.arange(0,T,0.33333333333333),0.33333333333333) for T in np.arange(0,18,1)]
plt.plot(np.arange(0,18,1),a,lw=1,label='N=16')
gušt.append(np.trapz(šestnajs[:(int(15/0.33333333333333)+1)],np.arange(0,15,0.33333333333333),0.33333333333333))

a=[np.trapz(dvajset[:(int(T/0.33333333333333)+1)],np.arange(0,T,0.33333333333333),0.33333333333333) for T in np.arange(0,10,1)]
plt.plot(np.arange(0,10,1),a,lw=1,label='N=20')
    
plt.legend()
plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_style('italic')
font.set_size('x-large')
font.set_family('serif')
plt.title('Vrednost integrala v odvisnosti od N',fontsize='14')
plt.xlabel('Meja integrala T',fontsize='14')
plt.ylabel('$\\int_{0}^{T}\\langle J(t)J(0)\\rangle$',fontsize='14')

#%%


# 12, 20 fit

plt.plot([2,4,8,10,12,14,16],np.abs(gušt))


#%%

plt.plot([2,4,8,10,12,14,16],[Integral10[i] for i in range(len(Integral10)-1)])
plt.plot(np.arange(0.1,30,0.1),[p(i) for i in np.arange(0.1,30,0.1)],label='fit')

plt.grid(linestyle=':',linewidth=0.95,color='seagreen')
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font = font0.copy()
font.set_style('italic')
font.set_size('x-large')
font.set_family('serif')
plt.title('Vrednost integrala v odvisnosti od N za T=10',fontsize='14')
plt.xlabel('Velikost verige N',fontsize='14')
plt.ylabel('$\\int_{0}^{10}\\langle J(t)J(0)\\rangle$',fontsize='14')
#%%
gušt=[0.+0.j,
 40.89982475+0.18288198j,
 84.88574389-1.57781543j,
 92.39970499-0.44506173j,
 97.24413199+0.38889694j,
 (102.6822863663154-0.3922362331377751j),
 (111.77617113499886-0.08986477794666567j)]
x=np.array([np.log(np.abs(gušt[i])) for i in range(1,len(gušt))])
y=np.array([4,8,10,12,14,16])

z=np.polyfit(y,x,1)
#%%

z=np.polyfit(np.array([2,4,8,10,12,14,16]),np.abs(gušt),20)
p = np.poly1d(z)

'''

