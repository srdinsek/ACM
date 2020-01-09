import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import svd as SVD
from numpy.random import rand,randint,normal,uniform
import numba
from numba import jit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from timeit import default_timer as timer




def RAZCEP(stanje,d):
    ''' Starinski in počasen razcep na A.'''
    
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

    return A




def MnoženjeA(N,cutoff,A):
    ''' Ta funkcija iz A izračunanega z RAZCEP in RAZCEP_z_rezanjem_A_lambda
    izračuna začetno stanje pred razcepom.
    '''
    cutoff=len(A[0][0])
    n=int(np.log(N)/np.log(2))
    
    psi=np.zeros(N)
    for i in np.arange(N):
        sezi=np.zeros((n,cutoff,cutoff),dtype=np.complex128)
        for korak in np.arange(n):
            sezi[korak]=A[korak][i>>(n-korak-1)&1]
        
        psi[i]=np.linalg.multi_dot(sezi)[0,0]
    return psi
    


def MnoženjeB(B_lambda):
    ''' Ta funkcija iz B izračunanega z 
    RAZCEP_B izračuna začetno stanje
    pred razcepom.
    '''
    B=B_lambda[0]
    S=B_lambda[1]
    
    cutoff=len(B[0,0])
    n=len(B)
    d=len(B[0])
    N=d**n
    
    psi=np.zeros(N,dtype=np.complex128)
    for i in np.arange(N):
        sezi=vmesna_pohitritev(n,B,i,S,cutoff)
        psi[i]=np.linalg.multi_dot(sezi)[0,0]
    return psi

def MnoženjeB2(B,S):
    ''' Ta funkcija iz B izračunanega z 
    RAZCEP_B izračuna začetno stanje
    pred razcepom.
    '''

    
    cutoff=len(B[0,0])
    n=len(B)
    d=len(B[0])
    N=d**n
    
    psi=np.zeros(N,dtype=np.complex128)
    for i in np.arange(N):
        sezi=vmesna_pohitritev(n,B,i,S,cutoff)
        psi[i]=np.linalg.multi_dot(sezi)[0,0]
    return psi

@jit(nopython=True)
def vmesna_pohitritev(n,B,i,S,cutoff):
    sezi=np.zeros((n,cutoff,cutoff),dtype=np.complex128)
    for korak in np.arange(n):
        if korak==0:
            sezi[korak]=B[korak][i>>(n-korak-1)&1]
        else:
            sezi[korak]=np.dot(np.diag(S[korak-1]),B[korak][i>>(n-korak-1)&1])
    return sezi




def RAZCEP_z_rezanjem_A_lambda(stanje,d,cutoff):
    ''' Funkcija naredi razcep A pri izbranem cutoffu in vrne
        A=seznam matrik A oblike (N,d,cutoff,cutoff)
        Lamda=seznam lambd oblike (N-1,cutoff).
        '''
    N=len(stanje)
    n=int(np.log(N)/np.log(d))
#    print(n)
    
    A=np.zeros((n,d,cutoff,cutoff),dtype=np.complex128)
    Lamda=np.zeros((n-1,cutoff),dtype=np.complex128)
#    entropija=np.zeros(cutoff,dtype=np.complex128)
    
    

    vrstice,stolpci=1,N
    U,S,V=SVD(stanje.reshape((vrstice*d,int(stolpci/d))), full_matrices=False)
    
    for power in np.arange(0,n-1):
        
        
        dolžina=len(S)
        if dolžina<cutoff:
            Lamda[power,:dolžina]=S
        else:
            Lamda[power]=S[:cutoff]
        

        for i in np.arange(d):
            Vmesni=U[i::d]
            višina=len(Vmesni)
            dolžina=len(Vmesni[0])
            if višina<cutoff and dolžina<cutoff:
                A[power,i,:višina,:dolžina]=Vmesni
            elif višina>=cutoff and dolžina<cutoff:
                #truncate
                A[power,i,:,:dolžina]=Vmesni[:cutoff, :]
            elif višina<cutoff and dolžina>=cutoff:
                #truncate
                A[power,i,:višina,:]=Vmesni[:, :cutoff]
            else:
                #truncate
                A[power,i]=Vmesni[:cutoff, :cutoff]
                
        
        
        SV=np.dot(np.diag(S.astype(np.complex128)),V)
        #SV=np.dot(np.diag(S),V)
        vrstice,stolpci=len(SV),len(SV[0])
        
        U,S,V=SVD(SV.reshape((vrstice*d,int(stolpci/d))), full_matrices=False)
        
        vsota=0
        for t in np.arange(len(S)):
            if S[t]>0:
                vsota=vsota-S[t]**2*np.log(S[t]**2)
#        entropija[power]=vsota
    
    
    for i in np.arange(d):
        Vmesni=SV[:,i::d]
        višina=len(Vmesni)
        dolžina=len(Vmesni[0])
        if višina<cutoff and dolžina<cutoff:
            A[-1][i][:višina,:dolžina]=Vmesni
    
    return A,Lamda



def A_v_B(A_lambda,meja):
    ''' Prehod iz zapisa A v zapis B'''
    
    A=A_lambda[0]
    S=A_lambda[1]
    
    
    n=len(A)
    d=len(A[0])
    cutoff=len(A[0,0])
    
    B=np.zeros((n,d,cutoff,cutoff),dtype=np.complex128)
    B[0]=A[0]
    
    for j in np.arange(n-1):
        S_inverz=np.zeros(cutoff,dtype=np.complex128)
        
        k=0
        while k<cutoff and np.abs(S[j][k])>meja:
            S_inverz[k]=1/S[j][k]
            k=k+1
        
        for spin in np.arange(d):
            B[j+1,spin]=np.dot(np.diag(S_inverz),A[j+1,spin])
            
    return B,S
    

def RAZCEP_B(stanje,cutoff,meja):
    ''' To je funkcija, ki dejansko naredi razcep začetnega stanja na
    matrike B in lambde.
    B[0]=seznam B matrik oblike (N,2,cutoff,cutoff)
    B[1]=seznam lambd oblike (N-1,cutoff).
    
    Parameter meja je meja, kako majhne lambde kar odrežemo. Minimum je 0.
    
    '''
    A=RAZCEP_z_rezanjem_A_lambda(stanje,2,cutoff)
    B=A_v_B(A,meja)
    return B

@jit(nopython=True)
def RAZCEP_B2(stanje,cutoff,meja):
    ''' To je funkcija, ki dejansko naredi razcep začetnega stanja na
    matrike B in lambde.
    B[0]=seznam B matrik oblike (N,2,cutoff,cutoff)
    B[1]=seznam lambd oblike (N-1,cutoff).
    
    Parameter meja je meja, kako majhne lambde kar odrežemo. Minimum je 0.
    
    '''
    A=RAZCEP_z_rezanjem_A_lambda(stanje,2,cutoff)
    B=A_v_B(A,meja)
    return B[0],B[1]

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
    ''' To je funkcija, ki se jo potem dejansko kliče.
    Ta funkcija ti vrne Hamiltonjan za Heisenbergov model za N delcev
    s periodičnimi (peri=True) ali prostimi (peri=False) robnimi pogoji.
    '''
    L=2**N
    h=np.zeros((L,L),dtype=np.int16)
    for i in np.arange(L):
        seznam=np.zeros(L,dtype=np.int16)
        seznam[i]=1
        h[i,:]=H(N,seznam,peri)
    return h


def naslov(ttitle,x_label,y_label):
    ''' To je funkcija za risanje graofv.'''
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



@jit(nopython=True)
def delovanje_U(X,lamda,z,j):
    ''' Preskušeno na numbi. Dela,.'''

    B=X
    S=lamda
    
    cutoff=len(S[0])
        
    e=np.exp(z)
    ch=np.exp(-z)*np.cosh(2*z)
    sh=np.exp(-z)*np.sinh(2*z)
    
    B_skupni=np.zeros((2,2,cutoff,cutoff),dtype=np.complex128)
    B_skupni[0,0]=np.multiply(e,np.dot(B[j,0],np.dot(np.diag(S[j]),B[j+1,0])))
    B_skupni[0,1]=np.add(np.multiply(ch,np.dot(B[j,0],np.dot(np.diag(S[j]),B[j+1,1]))),np.multiply(sh,np.dot(B[j,1],np.dot(np.diag(S[j]),B[j+1,0]))))
    B_skupni[1,0]=np.add(np.multiply(sh,np.dot(B[j,0],np.dot(np.diag(S[j]),B[j+1,1]))),np.multiply(ch,np.dot(B[j,1],np.dot(np.diag(S[j]),B[j+1,0]))))
    B_skupni[1,1]=np.multiply(e,np.dot(B[j,1],np.dot(np.diag(S[j]),B[j+1,1])))
    
    return B_skupni

@jit(nopython=True)
def Q(B_skupni,B,lamda,j):
    ''' Preskušeno na numbi. Dela,.'''
    S=lamda
    cutoff=len(S[0])
    D=cutoff*2
    Q1=np.zeros((D,D),dtype=np.complex128)
    n=len(S)
    
    if j==0:
        for k_minus in np.arange(D):
            for k_plus in np.arange(D):
                km=k_minus>>1
                kp=k_plus>>1
                Q1[k_minus,k_plus]=B_skupni[k_minus&1,k_plus&1,km,kp]*S[j+1,kp]
    elif j==n-1:
        for k_minus in np.arange(D):
            for k_plus in np.arange(D):
                km=k_minus>>1
                kp=k_plus>>1
                Q1[k_minus,k_plus]=S[j-1,km]*B_skupni[k_minus&1,k_plus&1,km,kp]
    else:
         for k_minus in np.arange(D):
            for k_plus in np.arange(D):
                km=k_minus>>1
                kp=k_plus>>1
                Q1[k_minus,k_plus]=S[j-1,km]*B_skupni[k_minus&1,k_plus&1,km,kp]*S[j+1,kp]
    return Q1


@jit(nopython=True)
def B_nova(cutoff,U,V,lamda,j,meja=0):
    S=lamda
    B_j=np.zeros((2,cutoff,cutoff),dtype=np.complex128)
    par=0
    km=par>>1
    if j!=0:
        while km<cutoff and np.abs(S[j-1,km])>meja:
            for sam in np.arange(cutoff):
                B_j[par&1,km,sam]=U[par,sam]/S[j-1,km]
            par=par+1
            km=par>>1
    else:
        while km<cutoff and 1>meja:
            for sam in np.arange(cutoff):
                B_j[par&1,km,sam]=U[par,sam]
            par=par+1
            km=par>>1
    
    B_jp=np.zeros((2,cutoff,cutoff),dtype=np.complex128)
    par=0
    kp=par>>1
    if (j+1)<len(S):
        while kp<cutoff and np.abs(S[j+1,kp])>meja:
            for sam in np.arange(cutoff):
                B_jp[par&1,sam,kp]=V[sam,par]/S[j+1,kp]
            par=par+1
            kp=par>>1
    else:
        while kp<cutoff and 1>meja:
            for sam in np.arange(cutoff):
                B_jp[par&1,sam,kp]=V[sam,par]
            par=par+1
            kp=par>>1

    return B_j,B_jp


def Korak(B,lamda, cutoff,meja,z,j):
    
    B_skupni=delovanje_U(B,lamda,z,j)
    
    q=Q(B_skupni,B,lamda,j)
    U,D,V=SVD(q, full_matrices=False)
    prvi,drugi=B_nova(cutoff,U,V,lamda,j)
    
    
    B[j]=prvi
    B[j+1]=drugi
    lamda[j]=D[:cutoff]
    return B,lamda


def sodi(n,B,lamda,cutoff, meja, z):
    for j in np.arange(int(n/2)):
        B,lamda=Korak(B,lamda,cutoff,meja,z,2*j)
    return B,lamda


def lihi(n,B,lamda,cutoff,meja,z):
    if n%2==0:
        for j in np.arange(1,int(n/2)):
            B,lamda=Korak(B,lamda,cutoff,meja,z,2*j-1)
    else:
        for j in np.arange(1,int(n/2)+1):
            B,lamda=Korak(B,lamda,cutoff,meja,z,2*j-1)
    return B,lamda



def S2(n,x,y,cutoff,meja,db):
    x,y=lihi(n,x,y,cutoff,meja,-db/2)
    x,y=sodi(n,x,y,cutoff,meja,-db)
    x,y=lihi(n,x,y,cutoff,meja,-db/2)
    return x,y

@jit(nopython=True)
def S3(n,x,y,cutoff,meja,p):
    x,y=lihi(n,x,y,cutoff,meja,-p[4])
    x,y=sodi(n,x,y,cutoff,meja,-p[3])
    x,y=lihi(n,x,y,cutoff,meja,-p[2])
    x,y=sodi(n,x,y,cutoff,meja,-p[1])
    x,y=lihi(n,x,y,cutoff,meja,-p[0])
    return x,y

@jit(nopython=True)
def TrotterSym(n,x,y,cutoff,meja,db,b):
    N=int(abs(b/db))
    X=[x]
    Y=[y]
    for i in range(N):
        x,y=S2(n,x,y,cutoff,meja,db)
        X.append(x)
        Y.append(y)
    return X,Y

# Tole je četrtega reda
@jit(nopython=True)
def SS4(n,x,y,cutoff,meja,db,b):
    N=int(abs(b/db))
    db0=-2**(1/3)/(2-2**(1/3))*db
    db1=1/(2-2**(1/3))*db
    
    d,f=np.copy(x),np.copy(y)
    X=[d]
    Y=[f]
    for i in range(N):
        print(i*db)
        x,y=S2(n,x,y,cutoff,meja,db1)
        x,y=S2(n,x,y,cutoff,meja,db0)
        x,y=S2(n,x,y,cutoff,meja,db1)
        
        d,f=np.copy(x),np.copy(y)
        X.append(d)
        Y.append(f)
    return X,Y


# Tole je četrtega reda
def SS4_norm(n,x,y,cutoff,meja,db,b):
    N=int(abs(b/db))
    db0=-2**(1/3)/(2-2**(1/3))*db
    db1=1/(2-2**(1/3))*db
    
    d,f=np.copy(x),np.copy(y)
    X=[d]
    Y=[f]
    for i in range(N):
        
        x,y=S2(n,x,y,cutoff,meja,db1)
#        if i==0:
#            for j in np.arange(n-1):
#                norma=np.sqrt(np.abs(np.vdot(y[j],y[j])))
#                y[j]=np.copy(np.divide(y[j],norma))
        x,y=S2(n,x,y,cutoff,meja,db0)
#        if i==0:
#            for j in np.arange(n-1):
#                norma=np.sqrt(np.abs(np.vdot(y[j],y[j])))
#                y[j]=np.copy(np.divide(y[j],norma))
        x,y=S2(n,x,y,cutoff,meja,db1)
        
        
        
        if i%100==0:
            print(i)
            psi=MnoženjeB([x,y])
#            norma=np.sqrt(np.abs(np.vdot(psi,psi)))
#            psi=np.divide(psi,norma)
            x,y=RAZCEP_B(psi,cutoff,meja)
            
        d,f=np.copy(x),np.copy(y)
        X.append(d)
        Y.append(f)
    return X,Y

# tole je tretjega reda slalomski
@jit(nopython=True)
def SS33(n,x,y,cutoff,meja,db,b):
    N=int(abs(b/(db*2)))
    p1=0.25*(1+1j/np.sqrt(3))*db
    p5=np.conj(p1)
    p2=2*p1
    p4=2*np.conj(p1)
    p3=0.5*db
    p=np.array([p1,p2,p3,p4,p5])
    
    d,f=np.copy(x),np.copy(y)
    X=[d]
    Y=[f]
    for i in range(N):
        x,y=S3(n,x,y,cutoff,meja,np.conj(p))
        x,y=S3(n,x,y,cutoff,meja,p)
        
        d,f=np.copy(x),np.copy(y)
        X.append(d)
        Y.append(f)
    return X,Y



def SS4_energija(n,x,y,cutoff,meja,db,b,stanje,D,začx,začy,delilec):
    N=int(abs(b/db))
    db0=-2**(1/3)/(2-2**(1/3))*db
    db1=1/(2-2**(1/3))*db
    EXP,beta=[],[]
    E=[]
    for i in range(N):
        print(i*db)
        x,y=S2(n,x,y,cutoff,meja,db1)
        x,y=S2(n,x,y,cutoff,meja,db0)
        x,y=S2(n,x,y,cutoff,meja,db1)
        
        
        if i> D and i%delilec==0:
            EXP.append(np.log(np.abs(norma(začx,začy,x,y,n,cutoff))))
            beta.append((i+1)*db)
            
    return EXP,beta,x,y,E

def Energija_osnovnega_stanja_SS4(N,cutoff,meja,z,b,delilec):
    stanje=np.array([normal(0,1) + normal(0,1)*1j for i in range(2**N)])
    stanje=stanje/np.sqrt(np.vdot(stanje,stanje))
    
#    stanje=np.add(normal(0,1,2**N),np.multiply(normal(0,1,2**N),1j))
    
#    stanje=np.zeros(2**N)
#    for i in range(2**N):
#        if i&1==0:
#            stanje[i]=normal(0,1)
#    stanje=stanje/np.sqrt(np.dot(stanje,stanje))
    
    print('razcep\n')
    B,S=RAZCEP_B(stanje,cutoff,meja)
    
    print('štart:')
    d=int(10/z)
    EXP,beta,B,S,E=SS4_energija(len(B),B,S,cutoff,meja,z,b,stanje,d,np.copy(B),np.copy(S),delilec)
    
    
    k=0
    korkor=[]
    for q in np.arange(1,len(B)):
        korkor.append(korelacija(k,q,B,S,N,cutoff))
    
    k=int(len(B)/2)
    korkor2=[]
    for q in np.arange(len(B)):
        if q!=k:
            korkor2.append(korelacija(k,q,B,S,N,cutoff))
    
    plt.plot(beta,EXP)
    EXP,beta=np.array(EXP),np.array(beta)
    xp=np.polyfit(beta,EXP,1)
    p = np.poly1d(xp)
    print(p[1])
    return korkor,korkor2,E
    

a=H_M(10,peri=False)
h=np.sort(np.linalg.eigh(a)[0])[0]
print(-h)
    

def norma(B1,S1,B2,S2,n,cutoff):
    T=pohitritev_norma(B1,S1,B2,S2,n,cutoff)
    return np.abs(np.linalg.multi_dot(T)[0,0])


@jit(nopython=True)
def pohitritev_norma(B1,S1,B2,S2,n,cutoff):
    size=cutoff**2
    T=np.zeros((n,size,size),dtype=np.complex128)
    j=0
    T[j]=np.add(np.kron(B1[j][0],B2[j][0]),np.kron(B1[j][1],B2[j][1]))
    for j in np.arange(1,n):
        s1,s2=np.diag(S1[j-1]),np.diag(S2[j-1])
        T[j]=np.add(np.kron(np.dot(s1,B1[j][0]),np.dot(s2,B2[j][0])),np.kron(np.dot(s1,B1[j][1]),np.dot(s2,B2[j][1])))
    
    return T

def korelacija(k,q,B,S,n,cutoff):
    T=pohitritev_korelacija(k,q,B,S,n,cutoff)
    return np.linalg.multi_dot(T)[0,0]

@jit(nopython=True)
def pohitritev_korelacija(k,q,B,S,n,cutoff):
    size=cutoff**2
    T=np.zeros((n,size,size),dtype=np.complex128)
    j=0
    if k==j or q==j:
        T[j]=np.subtract(np.conj(np.kron(B[j][0],B[j][0])),np.kron(B[j][1],B[j][1]))
    else:
        T[j]=np.add(np.conj(np.kron(B[j][0],B[j][0])),np.kron(B[j][1],B[j][1]))
    for j in np.arange(1,n):
        s=np.diag(S[j-1])
        if k==j or q==j:
            T[j]=np.subtract(np.conj(np.kron(np.dot(s,B[j][0]),np.dot(s,B[j][0]))),np.kron(np.dot(s,B[j][1]),np.dot(s,B[j][1])))
        else:
            T[j]=np.add(np.conj(np.kron(np.dot(s,B[j][0]),np.dot(s,B[j][0]))),np.kron(np.dot(s,B[j][1]),np.dot(s,B[j][1])))

    return T

    
    
    
