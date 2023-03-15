import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

def driver():


    # f = lambda x: np.exp(x)

    # N = 3
    # ''' interval'''
    # a = 0
    # b = 1

    # f=lambda x: 1.0/(1.0+((10.0*x)**2))
    f=lambda x: np.sinc(5*x)
    a=-1
    b=1
    N=30
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
        
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)

    ''' create vector with exact values'''
    fex = f(xeval)

    ''' Doing monomial expansion with vandermonde matrix'''
    V = np.zeros((N+1,N+1))
    for j in range(0,N+1):
        V[:,j] = np.power(xint,j)

    alpha = la.solve(V,yint)

    Veval = np.zeros((Neval+1,N+1))

    for j in range(0,N+1):
        Veval[:,j] = np.power(xeval,j)
    
    yeval_v=np.dot(Veval,alpha)

       

    plt.figure()    
    plt.plot(xeval,fex,'g',label='exact')
    plt.plot(xeval,yeval_l,'b',label='lagrange') 
    plt.plot(xeval,yeval_dd,'r',label='Newton DD')
    plt.plot(xeval,yeval_v,'y',label='Monomial')
    plt.legend()

    plt.figure() 
    err_l = np.abs(yeval_l-fex)
    err_dd = np.abs(yeval_dd-fex)
    err_v = np.abs(yeval_v-fex)
    # err_l = math.log(abs(yeval_l-fex))
    # err_dd = math.log(abs(yeval_dd-fex))
    # err_v = math.log(abs(yeval_v-fex))
    plt.semilogy(xeval,err_l,'b',label='lagrange')
    plt.semilogy(xeval,err_dd,'r',label='Newton DD')
    plt.semilogy(xeval,err_v,'y',label='Monomial')
    plt.legend()
    plt.show()
   

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  
    


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j]))
    return y
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

       

driver()        
