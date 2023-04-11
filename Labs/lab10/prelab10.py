import numpy as np

def eval_legendre(x, n):
    p = np.zeros(n+1)
    p[0] = 1
    p[1] = x

    i = 1
    while i < n:
        p[i+1] = (1/(i+1)) * ((2*i+1)*x*p[i] - i*p[i-1])
        i=i+1

    return p

ans = eval_legendre(2, 2)
print(ans)