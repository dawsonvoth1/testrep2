import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

def least_squares_example_driver(example = 0):
    if example == 0:
        xs = np.linspace(1, 10, 10)
        ys = np.array([1.3, 3.5, 4.2, 5.0, 7.0, 8.8, 10.1, 12.5, 13.0, 15.6])
        least_squares_degree = 5
        zs = np.linspace(0, 10, 201)
    elif example == 1:
        xs = np.linspace(0, 1, 5)
        ys = np.array([1, 1.284, 1.6487, 2.117, 2.7183])
        least_squares_degree = 4
        zs = np.linspace(0, 1, 201)
    
    interp_basis = lagrange_basis(xs)
    interp_coeffs = get_interpolation_coefficients(xs, ys, interp_basis)
    
    
    least_squares_basis = monomial_basis(least_squares_degree+1)
    least_squares_coeff = get_least_squares_coefficients(xs, ys, least_squares_basis)
    
    
    plt.figure()
    plt.plot(xs, ys, 'b*', label='data')
    plt.plot(zs, linear_combination_eval(zs, interp_coeffs, interp_basis), 'r-', label='interpolant')
    plt.plot(zs, linear_combination_eval(zs, least_squares_coeff, least_squares_basis), 'k--', label=f'degree {least_squares_degree} LS')
    
    plt.title(f'Example {example} Interpolation vs Least Squares')
    plt.legend()    
    #plt.show()

def monomial_basis(n):
    return [lambda x, i=i: x**i for i in range(n)]

def lagrange_polynomial(x, x_i, other_points):
    # f_i(x_j) = 0 for i =/= j
    # f_j(x_j) = 1
    # phi(x) = product of (x - x_j)/(x_i - x_j)
    y = 1 # start with 1 when making a product in a loop
    for x_j in other_points:
        y *= (x - x_j)/(x_i - x_j)
    return y

def remove_point_from_list(x_i, points):
    return [x_j for x_j in points if x_j != x_i]

def lagrange_basis(points):
    return [lambda x, x_i=x_i, other_points=remove_point_from_list(x_i, points): 
                lagrange_polynomial(x, x_i, other_points)
            for x_i in points]
    
def get_interpolation_coefficients(xs, ys, basis):
    M = np.array([[phi(x) for phi in basis] for x in xs])
    return la.solve(M, ys)

def linear_combination_eval(zs, coeffs, basis):
    return sum(c*phi(zs) for c, phi in zip(coeffs, basis))

def get_least_squares_coefficients(xs, ys, basis):
    A = np.array([[phi(x) for phi in basis] for x in xs])
    return la.solve(A.T@A, A.T@ys)

if __name__ == '__main__':
    least_squares_example_driver(example=0)
    least_squares_example_driver(example=1)
    plt.show()

