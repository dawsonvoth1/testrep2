# I was getting 2nd order for cos(x) for both types but when using x**2 I was getting 1st order for both

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm 

def driver():
    h=0.01*2.**(-np.arange(0.,10.))
    f = lambda x: math.cos(x)
    g = lambda x: x**2
    g1 = 4.
    gs = 2.
    gfd=np.zeros(10)
    gcd=np.zeros(10)
    s = math.pi/2.
    fd=np.zeros(10)
    cd=np.zeros(10)
    f1=-1.0
    for i in range(10):
        # gfd[i]=abs(forward_diff(g,gs,h[i]))
        # gcd[i]=abs(centered_diff(g,gs,h[i]))
        fd[i]=abs(forward_diff(f,s,h[i]))
        cd[i]=abs(centered_diff(f,s,h[i]))
        # print(gfd[i])
        # print(gcd[i])
        print(fd[i])
        print(cd[i])


    plt.plot(h,fd)
    plt.plot(h,cd)
    # plt.plot(h,gfd)
    # plt.plot(h,gcd)
    plt.show()

def forward_diff(f,s,h):
    return (f(s+h)-f(s))/(h)

def centered_diff(f,s,h):
    return (f(s+h)-f(s-h))/(2*h)


if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()