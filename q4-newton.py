#   THIRD-PARTY LIBRARIES
from typing import Callable
import autograd
import autograd.numpy as np
#import pandas as pd
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from IPython.display import display


#   GLOBAL VARIABLES
eps = 10**-4
N = 100




class DiffFunctions():

    def grad(self, f:Callable):
        g = autograd.grad
        def grad_f(x, y):
            x, y = float(x), float(y)
            return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
        return grad_f

DFunc = DiffFunctions()





#   FUNCTIONS
def Newton(F:Callable, x0:float, y0:float, eps:float=eps, N:int=N) -> tuple:

    """
    This function solves the equation F(x,y) = 0 around (x0,y0) using the Newton algorithm.

    :param F: The function to solve for
    :param x0: The initial x-axis coordinate
    :param y0: The initial y-axis coordinate
    :param eps: The acceptable precision of the algorithm
    :param N: The maximum number of iterations (will raise an error if exceeded)

    :returns: The solution to the equation F(x,y) = 0, to a precision of eps
    """

    #   0. Troubleshooting types (ugly)
    x0, y0 = float(x0), float(y0)

    #   1. Defining an iteration counter
    iter_counter = 0

    #   2. Running the method in a loop to refine the calculation
    while True:

        #<--- to fill --->#
        #THIS DOESN'T WORK YET, LOL
        
        #   2.1. Generating the gradient of F (n-dimensional derivative)
        gradF = DFunc.grad(F)

        #   2.2. Getting the value of the gradient of F at point (x0,y0)
        gradF_appl = gradF(x0,y0)

        #   2.3. Calculating new coordinates for a better approximation of the solution
        x = x0 - F(x0, y0) / gradF_appl[0]
        y = y0 - F(x0, y0) / gradF_appl[1]

        print(x, y)

        #   2.4. Breaking the loop and returning the solution when the precision condition (with eps) is met
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        
        #   2.5. Setting the values for the next iteration
        x0, y0 = x, y

        #   2.6. Incrementing the iteration counter
        iter_counter += 1
    

    #   3. Raising an error when no solution is found and the max number of iterations is exceeded
    raise ValueError(f'No convergence in {N} steps.')


def F(x, y):
    return x + y

val = Newton(F, -1, -1)
print(val)





