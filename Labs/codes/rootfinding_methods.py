# import libraries
import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt

def driver():

    # example function
    def fun(x):
        return x + np.cos(x) - 3;

    def dfun(x):
        return 1-np.sin(x);

    # params to run bisection and newton
    x0=3.5; a=3; b=4; tol=1e-15; nmax=1000;

    #apply bisection, plot log|rlist - r|
    (rlb,rb,info,itb)=bisection(fun,a,b,tol,nmax);

    idx=np.arange(itb);
    plt.plot(idx,np.log10(np.abs(rlb[idx]-rb)),'r-o');
    plt.show();

    # apply Newton
    (rln,rn,info,itn)=newton(fun,dfun,x0,tol,nmax);

    # plot log|rlist-r|
    idn=np.arange(itn);
    plt.plot(idn,np.log10(np.abs(rln[idn]-rn)),'b-o');
    plt.show();

#######################################################################
# Rootfinding methods
# Bisection
def bisection(f,a,b,tol,Nmax):
    '''
    Inputs:
        f,a,b       - function and endpoints of initial interval
        tol, Nmax   - bisection stops when interval length < tol
                    - or if Nmax iterations have occured
    Returns:
        astar - approximation of root
        ier   - error message
            - ier = 1 => cannot tell if there is a root in the interval
            - ier = 0 == success
            - ier = 2 => ran out of iterations
            - ier = 3 => other error ==== You can explain
    '''

    '''     first verify there is a root we can find in the interval '''
    p = np.zeros(Nmax+1);

    fa = f(a); fb = f(b);
    ######################################
    # if conditions for bisection aren't met, exit (ier=1)
    if (fa*fb>0):
        ier = 1;
        astar = a;
        p[0] = astar;
        return [p,astar, ier,count]

    ######################################
    # if f(a)=0 or f(b)=0, exit (ier=0)
    ''' verify end point is not a root '''
    if (fa == 0):
        astar = a;
        ier =0;
        p[0] = astar;
        return [p,astar, ier,count]

    if (fb ==0):
        astar = b;
        ier = 0;
        p[0] = astar;
        return [p,astar, ier,count]
    #####################################

    # iteration starts (while count<Nmax)
    count = 0;
    while (count < Nmax):
        c = 0.5*(a+b);
        fc = f(c);
        p[count+1] = c;

        # if midpoint is a zero, exit (ier=0)
        if (fc ==0):
            astar = c;
            ier = 0;
            p[count+1] = astar;
            return [p,astar, ier,count]

        if (fa*fc<0):
            b = c;
        elif (fb*fc<0):
            a = c;
            fa = fc;
        else:
            # if fc=0, exit (ier=3)
            astar = c;
            ier = 3;
            p[count+1] = astar;
            return [p,astar, ier,count]

        # once our interval is smaller than tol, exit (ier=0)
        if (abs(b-a)<tol):
            astar = a;
            ier =0;
            p[count+1] = astar;
            return [p,astar, ier,count]

        count = count +1;

    # If we are here, our algorithm ran out of iter. exit (ier=2)
    astar = a;
    ier = 2;
    return [p,astar,ier,count];

# Newton
def newton(f,fp,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
        - 0 if we met tol
        - 1 if we hit Nmax iterations (fail)
        """
    p = np.zeros(Nmax+1);
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1;
        pstar = p1;
        info = 1;
    return [p,pstar,info,it]

################################################################################
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver();
