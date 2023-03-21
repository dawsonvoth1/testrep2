import numpy as np


def driver():
    x = np.linspace(0,10,100)
    ind = np.where(x<=2)
    xeval = np.linspace(0,10,1000)
    xint = np.linspace(0,10,11)


def sub_evals(xeval, low, high):
    tmp1 = np.where(xeval>=low)[0]
    tmp2 = np.where(xeval<high)[0]
    tmp3 = np.concatenate((tmp1,tmp2))
    return tmp3

def slope_intercept(x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    f = lambda x: m*(x-x0)-y0
    return f

driver()        
