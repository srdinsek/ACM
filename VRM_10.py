import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from timeit import default_timer as timer
from scipy.optimize import root, bisect
from scipy.signal import argrelextrema
from scipy.integrate import trapz
from numba import jit
import seaborn as sns
sns.set_palette(sns.color_palette("hot", 20))




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
def R1(ixs,z):
    ''' ixs nam pove dolžino našega arraya torej je povezan z natančnostjo kot dx=1/ixs
     z nam pove z iz enačb
     Ta algoritem nam da začetni pogoj'''
    d=len(ixs)
    rezultat=np.zeros(d)
    for i in np.arange(d):
        x=ixs[i]
        rezultat[i]=2*np.sqrt(z-5/16)*x*(z-5/16)*np.exp(-(z-5/16)*x)
        
    return rezultat
# poklicali bomo recimo R=R1(ixs,z) in potem pognali algoritem

@jit(nopython=True)
def phi(R,x,dt,dol):
    ''' prej je bila definicija
    dol=int(x/dt),
    a zdaj več ni.
    '''
    R2=np.square(R)
#    Y=np.arange(x,len(R)*dt,dt)
    d=len(R2[dol:])
    os=np.arange(0,len(R2)*dt,dt)
    Y=np.zeros(d)
    for y in np.arange(d):
        Y[y]=y*dt+x
    if x!=0:
        return -trapz(R2[:(dol+1)],os[:(dol+1)],dx=dt)/x-trapz(np.divide(R2[dol:],Y),os[dol:],dx=dt)
    if x==0:a
        return -trapz(np.divide(R2,Y),os,dx=dt)


# sedaj bom definiral NUMEROVA
@jit(nopython=True)
def k(x,z,e,PHI,dt,dol,R):
    if x!=0:
        #np.pi**2 ali ne??
        return 2*e-2*(-z/x+2*PHI[dol]/x-(3*R[dol]**2/(2*np.pi**2*x**2))**(1/3))#   2*z/x-2*PHI[dol]/x+(3*R[dol]**2/(2*np.pi**2*x**2))**(1/3)
    if x==0:
        return 2*e-2*(-z/(10**-10)+2*PHI[dol]/(10**-10)-(3*R[dol]**2/(2*np.pi**2*(10**-10)**2))**(1/3)) #2*z/(10**-10)-2*PHI[dol]/(10**-10)+2*(3*R[dol]**2/(2*np.pi**2*(10**-10)**2))**(1/3)


@jit(nopython=True)
def solver(z,e,PHI,dt,R):
    
#    fi=np.array([PHI[1],PHI[2],PHI[3]])
    fi=np.array([0.001,0.01,0.1])
    a1=1
    a2=np.array([-(z+fi[i]) for i in np.arange(3)])
    a3=np.array([(-e+2*fi[i]*(fi[i]+2*z))/6 for i in np.arange(3)])
    a4=np.array([(2*e*(fi[i]+z)-fi[i]*fi[i]*(fi[i]-3*z))/18 for i in np.arange(3)])
#    a5=np.array([(e*(3*e-10*fi[i]*fi[i]-20*z*fi[i])+2*fi[i]*fi[i]*fi[i]*(fi[i]+4*z))/360 for i in np.arange(3)])
    
    x=np.arange(0,len(PHI)*dt,dt)
    
    dolžina=len(x)

    Y=np.zeros(dolžina)
    Y[0]=a1*x[0]+a2[0]*x[0]**2+a3[0]*x[0]**3+a4[0]*x[0]**4
    Y[1]=a1*x[1]+a2[1]*x[1]**2+a3[1]*x[1]**3+a4[1]*x[1]**4
    Y[2]=a1*x[2]+a2[2]*x[2]**2+a3[2]*x[2]**3+a4[2]*x[2]**4
    
    K=np.array([k(j*dt,z,e,PHI,dt,j,R) for j in np.arange(len(x))])
    
    Y=HitriLoop(K,Y,dt,dolžina)
    return Y

@jit(nopython=True)
def HitriLoop(K,Y,dt,dolžina):
    for i in np.arange(dolžina-3):
        Y[i+3]=(2*(1-5*(dt**2)*K[i+2]/12)*Y[i+2]-(1+(dt**2)*K[i+1]/12)*Y[i+1])/(1+(dt**2)*K[i+3]/12)
    return Y


@jit(nopython=True) 
def strelec(x,*args):
    '''s tem straljamo in args=(z,PHI3,dt,R)
    '''
    return solver(args[0],x,args[1],args[2],args[3])[-1]

@jit(nopython=True) 
def energija(R,z,PHI,dt,epsilon):
    '''izračunajmo še končno energijo.
    '''
    os=np.arange(dt,len(PHI)*dt,dt)
    R2=np.square(R[1:])
    R8_3=np.zeros(len(os))
    for i in range(len(os)):
        vmesni_rezultat=R[i+1]**(8/3)
        if math.isnan(vmesni_rezultat):
            R8_3[i]=0
        else:
            R8_3[i]=vmesni_rezultat
    return 2*epsilon-2*trapz(np.multiply(np.divide(PHI[1:],os),R2),os,dt)+1/2*(3/(2*np.pi**2))**(1/3)*trapz(np.divide(R8_3,np.power(os,2/3)),os,dt)

def risalec_funkcije(dt,l,z,x0,xmax,dx):
    seznam,graf=hitri_risalec(dt,l,z,x0,xmax,dx)
#    graf=np.array([strelec(x,z,R,dt) for x in seznam])
    plt.plot(seznam,np.abs(graf))
    naslov('Vrednost funkcije, ki ji iščemo ničlo za $Li$','$\\epsilon$','|f|')
    plt.yscale('log')
  
@jit(nopython=True)
def hitri_risalec(dt,l,z,x0,xmax,dx):
    os=np.arange(0,l,dt)
    R=R1(os,z)
    volumenR=np.sqrt(trapz(np.square(R),os,dx=dt))
    R=np.divide(R,volumenR)
    PHI,R=Potencial(R,dt)
    seznam=np.arange(x0,xmax,dx)
    graf=np.zeros(len(seznam))
    for x in np.arange(len(seznam)):
        graf[x]=strelec(seznam[x],z,PHI,dt,R)
    return seznam,graf

@jit(nopython=True)
def Potencial(R,dt):
    R,kk,os = Obrezovanje(R,dt)
    i=len(R)
    A=np.zeros(i)
    for j in np.arange(i):
        A[j]=-(phi(R,j*dt,dt,j))*j*dt+kk*j*dt
        
    return A,R #tole vrne \phi in ne \Phi. \Phi bi dobil, če A[j]=-(phi(R,j*dt,dt,j))+kk

def HartreeFock(dt,l,z,meja_minus,meja_plus,koraki=20):
    os=np.arange(0,l,dt)
    R=R1(os,z)
    volumenR=np.sqrt(trapz(np.square(R),os,dx=dt))
    R=np.divide(R,volumenR)
    mali=[]
    bali=[]
    
    PHI,R=Potencial(R,dt)
    # tu se začne zanka
    for bruno in np.arange(koraki):
        rešitev=bisect(strelec, meja_minus, meja_plus, args=(z,PHI,dt,R,), xtol=1e-12, rtol=4.4408920985006262e-15, maxiter=100, full_output=False, disp=True)
        mali.append(rešitev)
#        print('e=',rešitev)
        
        
        R=solver(z,rešitev,PHI,dt,R)
        PHI,R=Potencial(R,dt)
        plt.plot(os,R,label='{}. korak'.format(bruno))
        
        bali.append(energija(R,z,PHI,dt,rešitev))
        print('e=',rešitev,'\n','energija=',bali[-1])
        
#    print('e=',rešitev,'\n','energija=',energija(R,z,PHI,dt,rešitev))
    return R,PHI,mali,os,bali

@jit(nopython=True)
def Obrezovanje(ena,dt):
    '''ena=R'''
    i=len(ena)-1
    while (np.abs(ena[i])-np.abs(ena[i-1]))>0:
        i=i-1
    ena[i:]=np.multiply(np.ones(len(ena)-i),ena[i])
    i=len(ena)
    os=np.arange(0,i*dt,dt)
    volumenR=np.sqrt(trapz(np.square(ena),os,dx=dt))
    ena=np.divide(ena,volumenR)
    
    kk=(1--(phi(ena,i*dt,dt,i))*(i*dt))/(i*dt)
    return ena, kk, os


risalec_funkcije(0.001,20,10,-100,0,0.01)
ena,dva,tri,os,štiri=HartreeFock(0.001,20,10,-44,-40,20)
#%%
#    SOLVER ZA NAVADNO SCHRÖDINGEREJVO ENAČBO

# sedaj bom definiral NUMEROVA
@jit(nopython=True)
def k1(x,z,e,dt):
    ''' To je K ki ga uporabiš pri Numerovu'''
    if x!=0:
        return 2*z/x+2*e
    if x==0:
        return 2*z/(10**-10)+2*e

@jit(nopython=True)
def solver1(z,e,PHI,dt):
    ''' To je Numerov
        PHI - začetni pogoj
    '''
#    fi=np.array([PHI[1]*dt,PHI[2]*dt*2,PHI[3]*dt*3])
    a1=1
    a2=np.array([-(z) for i in np.arange(3)])
    a3=np.array([(-e)/6 for i in np.arange(3)])
    a4=np.array([(2*e*(+z))/18 for i in np.arange(3)])
#    a5=np.array([(e*(3*e))/360 for i in np.arange(3)])
    x=np.arange(0,len(PHI)*dt,dt)
    
    dolžina=len(x)
    Y=np.zeros(dolžina)
    Y[0]=a1*x[0]+a2[0]*x[0]**2+a3[0]*x[0]**3+a4[0]*x[0]**4
    Y[1]=a1*x[1]+a2[1]*x[1]**2+a3[1]*x[1]**3+a4[1]*x[1]**4
    Y[2]=a1*x[2]+a2[2]*x[2]**2+a3[2]*x[2]**3+a4[2]*x[2]**4
    
    K=np.array([k1(j,z,e,dt) for j in x])
    
    Y=HitriLoop(K,Y,dt,dolžina)
    return Y

@jit(nopython=True) 
def strelec1(x,*args):
    '''s tem straljamo in args=(z,R,dt)
    '''
    return solver1(args[0],x,args[1],args[2])[-1]

def risalec_funkcije1(dt,l,z,x0,xmax,dx):
    seznam,graf=hitri_risalec1(dt,l,z,x0,xmax,dx)
#    graf=np.array([strelec(x,z,R,dt) for x in seznam])
    plt.plot(seznam,np.abs(graf))
    naslov('Vrednost funkcije, ki ji iščemo ničlo za $Li$','$\\epsilon$','|f|')
    plt.yscale('log')
  
@jit(nopython=True)
def hitri_risalec1(dt,l,z,x0,xmax,dx):
    R=R1(np.arange(0,l,dt),z)
    seznam=np.arange(x0,xmax,dx)
    graf=np.zeros(len(seznam))
    for x in np.arange(len(seznam)):
        graf[x]=strelec1(seznam[x],z,R,dt)
    return seznam,graf

def HartreeFock1(dt,z,l,meja_minus,meja_plus,koraki=20):
    R=R1(np.arange(0,l,dt),z)
    volumenR=np.sqrt(trapz(np.square(R),np.arange(0,l,dt),dx=dt))
    R=np.divide(R,volumenR)
    mali=[]
    os=np.arange(0,len(R)*dt,dt)
    PHI3=np.zeros(len(os))
    # tu se začne zanka
    for bruno in np.arange(koraki):
        rešitev=bisect(strelec1, meja_minus, meja_plus, args=(z,PHI3,dt,), xtol=1e-12, rtol=4.4408920985006262e-15, maxiter=100, full_output=False, disp=True)
        R=solver1(z,rešitev,PHI3,dt)
        volumenR=np.sqrt(trapz(np.square(R),os,dx=dt))
        R=np.divide(R,volumenR)
        mali.append(rešitev )
        print('e=',rešitev)
#    print('e=',rešitev,'\n','energija=',energija1(R,z,PHI3,dt))
    return R,PHI3,mali

def energija1(R,z,PHI,dt):
    '''izračunajmo še končno energijo.
    '''
    E=13.6058
    R2=np.square(R)
    return 2*E*(trapz(np.square(np.gradient(R,dt)),dx=dt)-2*z*trapz(np.divide(R2[1:],np.arange(dt,len(R)*dt,dt)),dx=dt))

#risalec_funkcije1(0.01,30,1,-0.6,-0.4,0.000001)
ena1,dva1,tri1=HartreeFock1(0.001,1,20,-1,-0.2,2)
#%%
# zdaj pa nekaj testov

slika=[]
for dt in [0.2,0.1,0.05,0.01,0.005,0.001,0.0005]:
    ena,dva,tri=HartreeFock1(dt,1,30,-1,-0.2,2)
    slika.append(tri[-1])

plt.plot([0.2,0.1,0.05,0.01,0.005,0.001,0.0005],np.abs(np.divide(np.subtract(slika,-0.5),0.5)))
plt.yscale('log')
plt.xscale('log')
naslov('Z=1 napaka od $dt$','dt','$|(\\epsilon+0.5)/0.5|$')


plt.plot(np.arange(0,30,0.0005),ena)
#plt.yscale('log')
#plt.xscale('log')
naslov('Rešitev za Z=1 u(r)','r','u(r)')

#%% to pa je novajša, kjer odrežem in potem prilepim kar ostane in nakoncu nardim prosenovo aproksimacijo

sns.set_palette(sns.color_palette("hot", 10))



for dt in [0.001]:
    ena1,dva1,tri1=HartreeFock1(dt,1,20,-1,-0.2,2)
    R,kk,os = Obrezovanje(ena1,dt)
    i=len(ena1)
    A,A1=np.zeros(i),np.zeros(i)
    for j in np.arange(i):
        A[j]=-(phi(R,j*dt,dt,j))*j*dt+kk*j*dt
        A1[j]=-(j*dt+1)*np.exp(-2*j*dt)+1

    
    print(i*dt)
    l=round(i*dt,3)
    plt.plot(os,np.abs(np.divide(np.subtract(A,A1),A1)),label='dt='+str(dt)+', $r_{max}$'+str(l))
    plt.yscale('log')


naslov('Napaka U(r) za Z=1 Integral2 $r_{tot}=30$','r','$|(U(r)-U_{ex}(r))/U_{ex}(r)|$')



#%% to je tista starinska, kjer samo odrežem in ne podaljšam
sns.set_palette(sns.color_palette("hot", 10))
    
for dt in [0.2,0.1,0.05,0.01,0.005,0.001]:
    ena,dva,tri=HartreeFock1(dt,1,40,-1,-0.2,2)
    i=len(ena)-1
    while (np.abs(ena[i])-np.abs(ena[i-1]))>0:
        i=i-1
    R=np.zeros(len(ena))
    R=ena[:i]
    os=np.arange(0,i*dt,dt)
    volumenR=np.sqrt(trapz(np.square(R),os,dx=dt))
    R=np.divide(R,volumenR)
    
    kk=(1--(phi(R,i*dt,dt,i))*(i*dt))/(i*dt)
    A,A1=[],[]
    for j in np.arange(i):
        A.append(-(phi(R,j*dt,dt,j))*j*dt+kk*j*dt)
        A1.append(-(j*dt+1)*np.exp(-2*j*dt)+1)
    #plt.plot(A)
    #plt.plot(A1)
    print(len(A),len(A1))
    
    print(i*dt)
    l=round(i*dt,3)
    plt.plot(os,np.abs(np.divide(np.subtract(A,A1),A1)),label='dt='+str(dt)+', $r_{max}$'+str(l))
    plt.yscale('log')


naslov('Napaka U(r) za Z=1 Integral2 $r_{tot}=30$','r','$|(U(r)-U_{ex}(r))/U_{ex}(r)|$')

#%%
plt.plot(os,A,color='black',label='Integral2')
plt.plot(os,A1,color='red',label='eksaktno')
naslov('U(r) za Z=1 Integral2 dt=0.001','r','U(r)')
#plt.xscale('log')
