import numpy as np
import math

def steepest_descent(f, grad, x0, tol, max_iter, alpha_func):
    """
    f: function to minimize, takes a numpy array x as input and returns a scalar
    grad: gradient of f, takes a numpy array x as input and returns a numpy array
    x0: initial guess
    tol: tolerance for stopping criterion
    max_iter: maximum number of iterations
    alpha_func: function to compute the step size alpha, takes a numpy array x and a direction d as input and returns a scalar
    """
    x = x0
    iter_count = 0
    while iter_count < max_iter:
        grad_f = grad(x)
        d = -grad_f
        alpha = alpha_func(x, d)
        x_new = x + alpha * d
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        iter_count += 1
    return x_new

def solve_nonlinear_system(f, grad, x0, tol=1e-6, max_iter=1000, alpha_func=None):
    if alpha_func is None:
        # Use minimum of a quadratic approximation as the step size function
        def alpha_func(x, d):
            grad_f = grad(x)
            H_f = np.array([[grad(lambda x: grad_f[i](x)[j])(x) for j in range(len(x))] for i in range(len(x))])
            return -np.dot(grad_f, d) / np.dot(d, np.dot(H_f, d))

    return steepest_descent(f, grad, x0, tol, max_iter, alpha_func)


def f(x):
    return np.array([x[0] + np.cos(x[0]*x[1]*x[2]) - 1.,
                     (1-x[0])**0.25 + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1.,
                     -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1.])

def grad_f(x):
    return np.array([np.array([1. - x[1]*x[2]*np.sin(x[0]*x[1]*x[2]), -x[0]*x[2]*np.sin(x[0]*x[1]*x[2]), -x[0]*x[1]*np.sin(x[0]*x[1]*x[2])]),
                     np.array([-0.25*(math.pow((1-x[0]),-3/4)), 1., 0.1*x[2] - 0.15]),
                     np.array([-2.*x[0] + 0.01, -0.2*x[1] + 1., 1.])])

x0 = np.array([1.0, 1.0, 1.0])
tol = 1e-6
max_iter = 1000
result = solve_nonlinear_system(f, grad_f, x0, tol, max_iter)

print("Solution: ", result)
