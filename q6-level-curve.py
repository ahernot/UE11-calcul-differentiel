#   THIRD-PARTY LIBRARIES
from typing import Callable
import autograd
import autograd.numpy as np
#import pandas as pd
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from IPython.display import display


#   GLOBAL VARIABLES
eps = 10**-6
N = 10**4


from q4_newton import Newton





#   FUNCTIONS
def level_curve(f:Callable, x0:float, y0:float, delta:float=0.1, N:int=1000, eps:float=eps):

    """
    This function solves the equation F(x,y) = 0 around (x0,y0) using the Newton algorithm.

    :param f: The function to solve for
    :param x0: The initial x-axis coordinate
    :param y0: The initial y-axis coordinate
    :param eps: The acceptable precision of the algorithm
    :param N: The maximum number of iterations (will raise an error if exceeded)

    :returns: The solution to the equation F(x,y) = 0, to a precision of eps
    """


def f0(x, y):
    return x + y

def f1(x1, x2):
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2

val = Newton(f1, 0.8, 0.8)