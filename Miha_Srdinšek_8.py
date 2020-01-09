import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import svd as SVD
from numpy.random import rand,randint,normal,uniform
import numba
from numba import jit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh





#t=15
#stanje=rand(2**t)
#stanje=stanje/np.dot(stanje,stanje)
#a,s=RAZCEP(stanje,2)
#
#
#import seaborn as sns
#sns.set_palette(sns.color_palette("hot", 10))
#seznam=[2,4,6,8,10,12,14]
#hranilnica2=[]
#for i in np.arange(len(seznam)):
#    stanje=rand(4**seznam[i])
#    stanje=stanje/np.dot(stanje,stanje)
#    a,s=RAZCEP(stanje,4)
#    plt.plot(np.linspace(0,1,len(s)),s,'-',label='N:'+str(seznam[i]))
#    naslov('Entropija prepletenostoi od N random s=1','Razmerje particije','$\\sum \;\;\\lambda_{k}^{2}\\log(\lambda_{k}^{2})$')
#    print(i)
#    print(seznam[i])
#    print(len(s))
#    hranilnica2.append(s)








#
#

#slika=[hranilnica[i][int(len(hranilnica[i])/3-1)] for i in range(len(hranilnica))]
#slika2=[hranilnica1[i][int(len(hranilnica1[i])/3-1)] for i in range(len(hranilnica1))]
#slika3=[hranilnica2[i][int(len(hranilnica2[i])/3-1)] for i in range(len(hranilnica2))]
#plt.plot(seznam,slika,color='red',label='d=2')
#plt.plot(seznam,slika2,color='black',label='d=3')
#plt.plot([seznam[i] for i in range(len(seznam)-1)],slika3,color='orange',label='d=4')
#naslov('Simetrična biparticija random od N','število delcev N','Entropija preplettenosti na sredi')
#
#plt.yscale('log')





##########################################################################################
#
#                           OSNOVNO STANJE HAMILTONJANA
#
##########################################################################################




#w1,v1=np.linalg.eigh(a)




# zdaj bom tu pridelal lastna stanja
#bazen_vektorjev_peri=[]
#lastne_energije_peri=[]
#for n in [2,4,6,8,10,12,14]:
#    a=H_M(n,peri=True)
#    A=csr_matrix(a,dtype=np.float)
#    w,v=eigsh(A,k=1,which='SA',return_eigenvectors=True)
#    
#    print(w)
#    np.save('bazni_vektorji_heisenberg_pT_'+str(n),v)
#    
#    lastne_energije_peri.append(w)
#    bazen_vektorjev_peri.append(v)



##########################################################################################
#
#                           NEKOMPAKTNA PARTICIJA
#
##########################################################################################




#def SD_particija(stanje,N):
#    L=len(stanje)
#    G=int(2**(N/2))
#    GRUPA=np.zeros((G,G))
#    for i in np.arange(L):
#        #dekompozicija binarnega števila na dve števili
#        a=[(i>>j)&1 for j in np.arange(0,N,2)]
#        s=np.sum(a[i]*2**i for i in range(len(a)))
#        a=[(i>>j)&1 for j in np.arange(1,N+1,2)]
#        l=np.sum(a[i]*2**i for i in range(len(a)))
#        
#        GRUPA[s,l]=stanje[i]
#    return GRUPA

# tole je drugačna SD particija
#def SD_particija(stanje,N):
#    L=len(stanje)
#    G1=2**len(np.arange(0,N,5))
#    G2=2**len([j for j in np.arange(1,N,1) if j not in np.arange(0,N,5)])
#    GRUPA=np.zeros((G1,G2))
#    for i in np.arange(L):
#        #dekompozicija binarnega števila na dve števili
#        a=[(i>>j)&1 for j in np.arange(0,N,5)]
#        s=np.sum(a[i]*2**i for i in range(len(a)))
#        a=[(i>>j)&1 for j in np.arange(1,N,1) if j not in np.arange(0,N,5)]
#        l=np.sum(a[i]*2**i for i in range(len(a)))
#        GRUPA[s,l]=stanje[i]
#    return GRUPA
#
#
#
#
#
#
#seznam=[2,4,6,8,10,12,14]
#e=[]
#for n in range(len(seznam)):
#    print('n:',n)
#    a=np.load('bazni_vektorji_heisenberg_pT_'+str(seznam[n])+'.npy')
#    U,S,V=SVD(SD_particija(a,seznam[n]), full_matrices=False, compute_uv=True)
#    
#    e.append(sum(-S[i]**2*np.log(S[i]**2) for i in range(len(S)) if S[i]!=0))
#
#plt.plot(seznam,e,':',color='red',label='periodični Hei')
#
#
#e=[]
#for n in range(len(seznam)):
#    a=np.load('bazni_vektorji_heisenberg_pF_'+str(seznam[n])+'.npy')
#    U,S,V=SVD(SD_particija(a,seznam[n]), full_matrices=False, compute_uv=True)
#    
#    e.append(sum(-S[i]**2*np.log(S[i]**2) for i in range(len(S)) if S[i]!=0))
#    
#plt.plot(seznam,e,':',color='black',label='neperiodični Hei')
#
#e=[]
#for n in range(len(seznam)):
#    stanje=rand(2**seznam[n])
#    a=stanje/np.dot(stanje,stanje)
#    U,S,V=SVD(SD_particija(a,seznam[n]), full_matrices=False, compute_uv=True)
#    
#    e.append(sum(-S[i]**2*np.log(S[i]**2) for i in range(len(S)) if S[i]!=0))
#
#plt.plot(seznam,e,':',color='orange',label='random')
#
#e=[]
#for n in range(len(seznam)):
#    stanje=np.array([normal(0,1)+1j*normal(0,1) for i in range(2**seznam[n])])
#    a=np.divide(stanje,np.vdot(stanje,stanje))
#    U,S,V=SVD(SD_particija(a,seznam[n]), full_matrices=False, compute_uv=True)
#    
#    e.append(sum(-S[i]**2*np.log(S[i]**2) for i in range(len(S)) if S[i]!=0))
#
#plt.plot(seznam,e,':',color='brown',label='normal random')
#
#naslov('Nekompaktna biparticija (štirice)','Število delcev N','Entropija prepletenosti od N')





























#########################################################################################################################################
    
#########################################################################################################################################
    
#########################################################################################################################################
    
#########################################################################################################################################
    
#########################################################################################################################################

#########################################################################################################################################
    
#########################################################################################################################################
    
#########################################################################################################################################
    
#########################################################################################################################################
    
#########################################################################################################################################


def RAZCEP(stanje,d):
    N=len(stanje)
    n=int(math.log(N,d))
    print(n)
    
    A,entropija=[],[]
    SV=stanje
    vrstice,stolpci=1,N
    for power in np.arange(1,n):
        
        U,S,V=SVD(SV.reshape((vrstice*d,int(stolpci/d))), full_matrices=False, compute_uv=True)
        zbirka=[U[i::d] for i in np.arange(d)]
        A.append(zbirka)
        
        
        SV=np.dot(np.diag(S),V)
        vrstice,stolpci=len(SV),len(SV[0])
        
        entropija.append(sum(-S[i]**2*np.log(S[i]**2) for i in range(len(S))))
    
    
    zbirka=[SV[:,i::d] for i in np.arange(d)]
    A.append(zbirka)
    
    
    
    return A,entropija

@jit(nopython=True)
def H(N,seznam,peri):
    L=2**N
    začetno=np.zeros(L,dtype=np.int16)
    huuk=2**0+2**(N-1)
    if peri==True:
        for i in np.arange(L):
            #-------------------------------------------------
            # tu samo naredimo korak za j=N
            c=2**(N-1)
            prvi=i&1
            drugi=i&c
            if (prvi==0 and drugi==0) or (prvi==1 and drugi==c): začetno[i]+=seznam[i]
            elif prvi==1 and drugi==0: začetno[i]+=-seznam[i]+2*seznam[i^huuk]
            elif prvi==0 and drugi==c: začetno[i]+=-seznam[i]+2*seznam[i^huuk]
            #-------------------------------------------------
    
    for i in np.arange(L):
        #-------------------------------------------------
        # tu samo naredimo korak za j=0
        
        prvi=i&1
        drugi=(i>>1)&1
        
        
        if (prvi==0 and drugi==0) or (prvi==1 and drugi==1): začetno[i]+=seznam[i]
        elif prvi==1 and drugi==0: začetno[i]+=-seznam[i]+2*seznam[i^3]
        elif prvi==0 and drugi==1: začetno[i]+=-seznam[i]+2*seznam[i^3]
        #-------------------------------------------------
    
    #-------------------------------------------------
    # tu pa naredimo korake za preostale j
    for j in np.arange(1,N-1,1):
        for i in range(L):
            
            prvi=(i>>j)&1
            drugi=(i>>(j+1))&1
            
            if (prvi==0 and drugi==0) or (prvi==1 and drugi==1): začetno[i]+=seznam[i]
            elif prvi==1 and drugi==0: 
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]+=-seznam[i]+2*seznam[n]
            elif prvi==0 and drugi==1:
                cor=2**(j)+2**(j+1)
                n=i^cor
                začetno[i]+=-seznam[i]+2*seznam[n]
    #-------------------------------------------------
    return začetno


@jit(nopython=True)
def H_M(N,peri):
    L=2**N
    h=np.zeros((L,L),dtype=np.int16)
    for i in np.arange(L):
        seznam=np.zeros(L,dtype=np.int16)
        seznam[i]=1
        h[i,:]=H(N,seznam,peri)
    return h

def H_M_stari(N,peri):
    L=2**N
    h=np.zeros((L,L),dtype=np.int16)
    for i in range(L):
        
        seznam=np.zeros(L,dtype=np.int16)
        seznam[i]=1
        h[i,:]=H_stara(N,seznam,peri)
    return h

def H_stara(N,seznam,peri):
    L=2**N
    začetno=np.zeros(L,dtype=np.int16)
    huuk=2**0+2**(N-1)
    if peri==True:
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


def naslov(ttitle,x_label,y_label):
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
    plt.title(ttitle,fontsize='14')
    plt.xlabel(x_label,fontsize='14')
    plt.ylabel(y_label,fontsize='14')
    plt.legend()
