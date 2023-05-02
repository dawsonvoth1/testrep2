import numpy as np
import math
from numpy.linalg import norm
import matplotlib.pyplot as plt

def evalF(x): 
    F = np.array([x[0]+math.cos(x[0]*x[1]*x[2])-1.,
                  math.pow((1.-x[0]),1/4)+x[1]+0.05*(x[2]**2)-0.15*x[2]-1.,
                  -(x[0]**2)-0.1*(x[1]**2)+0.01*x[1]+x[2]-1.])
    return F

def evalJ(x):
    J = np.array([[-x[1]*x[2]*math.sin(x[0]*x[1]*x[2])+1., -x[0]*x[2]*math.sin(x[0]*x[1]*x[2]), -x[0]*x[1]*math.sin(x[0]*x[1]*x[2])],
                    [(-1.)/(4*math.pow((1-x[0]),3/4)), 1., 0.1*x[2]-0.15],
                    [-2.*x[0], 0.01-0.2*x[1], 1.]])
    return J

# sum(f^2)
def evalg(x):
    F = evalF(x)
    g = F[0]**2 + F[1]**2 +F[2]**2
    return g

# doesnt check for 0 gradient
def steepest_descent(x0, tol, Nmax):
    k = 0
    for k in range(Nmax):
        J0 = evalJ(x0)
        F0 = evalF(x0) # iteration of function
        grad = 2.0*(J0.transpose()).dot(F0) # gradient
        b1=0
        b3=1
        h1 = evalg(x0-b1*grad) # just for show it is really just evalg(x0)
        h3 = evalg(x0-b3*grad)
        while norm(h1) <= norm(h3):
            b3 = b3/2
            h3 = evalg(x0-b3*grad)
            if b3 < tol/2:
                print("no likely improvement")
                xstar = x0
                its = k
                ier = -2
                return [xstar,ier,its]
        b2 = (b1+b3)/2
        h2 = evalg(x0-b2*grad)
        s1 = (h2-h1)/b2
        s2 = (h3-h2)/(b3-b2)
        s3 = (s2-s1)/b3
        # step 11 from book
        b0 = .5*(b2 - (s1/s3)) # critical point from book
        h0 = evalg(x0-b0*grad)
        B = b0
        # step 12 from book
        # M = min([h0,h1,h2,h3])
        # if M == h0:
        #     B=b0
        # elif M == h1:
        #     B=b1
        # elif M == h2:
        #     B = b2
        # elif M == h3:
        #     B = b3
        # h0 = evalg(x0-b0*grad)

        #  step 13 from book
        xk = x0-B*grad
        # step 14 from book
        if(norm(h0-h1) < tol):
            xstar = xk
            ier = 0
            its = k
            return [xstar,ier,its]
        x0 = xk
    xstar = x0
    its = k
    ier = -1
    return [xstar,ier,its]


x0 = np.array([0,0,0])

Nmax = 100
# tol = 5e-2
tol=1e-6
[xstar,ier,its] = steepest_descent(x0,tol,Nmax)
print("xstar: ",xstar)
print("its: ",its)


# def steepest_descent(x0, tol, Nmax):
#     k = 1
#     xn = np.zeros((Nmax,2))
#     while k < Nmax:
#         J = evalJ(x0)
#         g1 = evalF(x0) # iteration of function
#         z = 2.0*(J.transpose()).dot(g1) # gradient
#         z0 = norm(z, 2)
#         if x0 == 0:
#             ier = 1 # zero gradient
#         z = z/z0
#         a1 = 0
#         a3 = 1.
#         g3 = evalF(x0-(a3*z))
#         while norm(g3) >= norm(g1):
#             a3 = a3/2.
#             g3 = evalF(x0-(a3*z))
#             if a3 < tol/2.0:
#                 # no like improvement
#                 ier = 2
#                 return [x0, xn, ier]
#         a2 = a3/2.0
#         g2 = evalF(x0-(a2*z))
#         h1 = (g2-g1)/a2
#         h2 = (g3-g2)/(a3-a2)
#         h3 = (h2-h1)/a3
#         a0 = 0.5*(a2-(h1/h3))
#         g0 = evalF(x0-(a0*z))
#         # step 12 in book
#         M = min([g0,g1,g2,g3])
#         print("m ",M)
#         a=.1
#         if M == g0:
#             a=a0
#         elif M == g1:
#             a=a1
#         elif M == g2:
#             a == a2
#         elif M == g3:
#             a = a3
#         print("a ",a)
#         x0=x0-(a*z)

#         # try psuedo code from lec notes

