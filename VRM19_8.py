import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import svd as SVD
from numpy.random import rand,randint,normal,uniform


#stanje=np.array([0,0,0,1,0,0,0,0]) # to je začetno stanje
#N=len(stanje)
#d=2
#
#A=[]
#
#U,S,V=SVD(stanje.reshape((d,int(N/d))), full_matrices=False, compute_uv=True)
#
#A.append(U)
#
#
#SV=np.dot(np.diag(S),V)
#U,S,V=SVD(SV.reshape((d**2,int(N/d**2))), full_matrices=False, compute_uv=True)
#A.append(U)
#
#SV=np.dot(np.diag(S),V)
#
#A.append(SV.reshape((4,1)))


def RAZCEP(stanje,d):
    N=len(stanje)
    n=int(math.log(N,d))
    print(n)
    
    A=[]
    SV=stanje
    vrstice,stolpci=1,N
    for power in np.arange(1,n):
        
        U,S,V=SVD(SV.reshape((vrstice*d,int(stolpci/d))), full_matrices=False, compute_uv=True)
        zbirka=[U[i::d] for i in np.arange(d)]
        A.append(zbirka)
        
        
        SV=np.dot(np.diag(S),V)
        vrstice,stolpci=len(SV),len(SV[0])
    
    
    zbirka=[SV[:,i::d] for i in np.arange(d)]
    A.append(zbirka)
    
    
#    psi=[]
#    for i in range(N):
#        j=i
#        sezi=[]
#        for korak in range(n):
#            sezi.append(A[korak][j>>(n-korak-1)])
#            if j>>(n-korak-1)==1:
#                j=j-d**(n-korak-1)
#        
#        psi.append(np.linalg.multi_dot(sezi))
#    psi=np.array(psi)
    return A#,psi.reshape(N)
    

t=20
stanje=rand(2**t)
stanje=stanje/np.dot(stanje,stanje)
#stanje=np.array([0., 0., 0., 0., 0., 1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 1, 0., 0., 0., 0., 0.])
start=timer()
a=RAZCEP(stanje,2)
čas1.append(timer()-start)
#razlika=np.subtract(stanje,psi)
#print(razlika)
#for i in razlika:
#    if i >10**-15:
#        pritn('pazi')
        

    



