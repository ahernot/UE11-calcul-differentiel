#   THIRD-PARTY LIBRARIES
import autograd
import autograd.numpy as np
#import pandas as pd
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from IPython.display import display


#   GLOBAL VARIABLES
#eps = 
N = 10**4


#   FUNCTIONS
def Newton(F:func, x0:double, y0:double, eps:double=eps, N:int=N) -> tuple:

    """
    This function solves the equation F(x,y) = 0 around (x0,y0) using the Newton algorithm.

    :param F: The function to solve for
    :param x0: The initial x-axis coordinate
    :param y0: The initial y-axis coordinate
    :param eps: The acceptable precision of the algorithm
    :param N: The maximum number of iterations (will raise an error)

    :returns: The solution to the equation F(x,y) = 0, to a precision of eps
    """

    #   1. Defining an iteration counter
    iter_counter = 0

    #   2. Running the method in a loop
    while True:
        
        #<--- to fill --->#

        #   2.X. Stopping the loop when the precision condition (eps) of the solution is met
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        
        #   2.X. Setting the values for the next iteration
        x0, y0 = x, y

        #   2.X. Incrementing the iteration counter
        iter_counter += 1
    

    #   3. Raising an error when no solution is found and the max number of iterations is exceeded
    raise ValueError(f'No convergence in {N} steps.')