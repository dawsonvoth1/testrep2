import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 1000
 
     ''' Right hand side'''
     b1 = np.random.rand(N,10)
     b2 = np.array(b1)
     A = np.random.rand(N,N)

     p, l, u = scila.lu(A)
     start1 = time.time()
     lu, piv = scila.lu_factor(A)
     end1 = time.time()
     fact = end1 - start1
     start1 = time.time()
     x2 = scila.lu_solve((lu,piv), b2)
     end1 = time.time()
     solving = end1 - start1
     print(fact + solving, fact, solving)
     test2 = np.matmul(A,x2)
     r2 = la.norm(test2-b2)

     start2 = time.time()
     x1 = scila.solve(A,b1)
     end2 = time.time()
     print(end2 - start2)
     test1 = np.matmul(A,x1)
     r1 = la.norm(test1-b1)
     
     print(r1, r2)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
