#   THIRD-PARTY LIBRARIES
from typing import Callable

import numpy as npy # importing the numpy functions which are not overwritten by autograd.numpy (and thus are otherwise missing)

import autograd
import autograd.numpy as np

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [7, 7] # [width, height] (inches)

from IPython.display import display



#   GLOBAL VARIABLES
N = 100
eps = 10**-7



class DiffFunctions():

    def gradient(self, f:Callable):
        g = autograd.grad
        def grad_f(x, y):
            x, y = float(x), float(y) # troubleshooting types (ugly)
            return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
        return grad_f


    def jacobian(self, f:Callable):
        j = autograd.jacobian
        def J_f(x, y):
            x, y = float(x), float(y) # troubleshooting types (ugly)
            return np.array([ j(f, 0)(x, y) , j(f, 1)(x, y) ]) .T
        return J_f



class GeomFunctions():

    def linarray_2d(self, x_range:tuple, y_range:tuple, steps:tuple) -> np.ndarray:
        """
        This function returns a grid of 2D points in range [x_start, x_stop) and [y_start, y_stop).

        :param x_range: The x-axis range of the grid [x_start, x_stop)
        :param y_range: The y-axis range of the grid [y_start, y_stop)
        :param steps: The step length on each axis (x_step, y_step)

        :returns: The grid of points

        """
        x_step, y_step = steps
        lin_array = npy.mgrid[x_range[0] : x_range[1] : x_step , y_range[0] : y_range[1] : y_step]
        return lin_array.transpose(1, 2, 0)

    
    def display_contour_inline(self, f:Callable, x:float, y:float, levels):
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        #fig, ax = plt.subplots()
        contour_set = plt.contour(X, Y, Z, colors="grey", linestyles="dashed", levels=levels)
        #ax.clabel(contour_set)

        plt.grid(True)
        plt.xlabel("$x$") 
        plt.ylabel("$y$")
        #plt.gca().set_aspect("equal")



DFunc = DiffFunctions()
GFunc = GeomFunctions()



#   Fonctions d'application

def f0(x: float or int, y: float or int) -> float:
    return x + y

def f1(x: float or int, y: float or int) -> float:
    return 3.0 * x * x - 2.0 * x * y + 3.0 * y * y



def Newton(F:Callable, x0:float, y0:float, eps:float=eps, N:int=N) -> tuple:
    """
    This function solves the equation F(x,y) = 0 around (x0,y0) using the Newton algorithm.

    :param F: The function to solve for — it must necessarily have a multi-dimensional image in a np.ndarray type
    :param x0: The initial x-axis coordinate
    :param y0: The initial y-axis coordinate
    :param eps: The acceptable precision of the algorithm
    :param N: The maximum number of iterations (will raise an error if exceeded)

    :returns: The solution to the equation F(x,y) = 0, to a precision of eps
    """

    #   0. Troubleshooting types (ugly)
    x0, y0 = float(x0), float(y0)

    #   1. Defining the X0 point
    X0 = np.array([x0, y0])

    #   2. Generating the jacobian matrix of F (n-dimensional derivative)
    jacF = DFunc.jacobian(F)

    #   3. Running the method in a loop to refine the calculation
    for iter_counter in range(N):

        #   3.1. Inverting F's jacobian matrix
        try:
            jacF_inv = npy.linalg.inv( jacF( *(X0.tolist()) ) )
        except npy.linalg.linalg.LinAlgError:
            raise ValueError('The function to solve for has got a singular jacobian matrix in the desired point.')

        #   3.2. Dot product between jacF and F(X0)
        F_dot = npy.dot( jacF_inv, F( *(X0.tolist()) ) )

        #   3.3. Computing the new X point
        X = X0 - F_dot

        #   3.4. Exiting the function once the desired precision is reached
        if npy.linalg.norm( X - X0 ) <= eps:
            return tuple(X.tolist())

        #   3.5. Performing end-of-loop actions
        X0 = X.copy()

    #   4. Raising an error when no solution is found and the max number of iterations is exceeded
    raise ValueError(f'No convergence in {N} steps.')






# Generating a grid to apply the function to
lin_array = GFunc.linarray_2d((-0.75, 0.75), (-0.75, 0.75), (0.2, 0.2))#0.05, 0.05))

def get_F(f:Callable, level:float = 0.0, offset:float = 0.0) -> Callable:
    def F(x: float or int, y: float or int) -> np.ndarray:
        return np.array([ f(x, y) - level , x - y - offset ])
    return F



plt.figure()
plt.xlim(-1, 1)
plt.ylim(-1, 1)


GFunc.display_contour(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)
#   Varying the seed point
x_list = []
y_list = []
F = get_F(f1, level=0.8)
# very ugly because Newton isn't vectorized
for line in lin_array:
    for col in lin_array:
        for point in col:
            try:
                val = Newton(F, *(point.tolist()))
                x_list.append(val[0])
                y_list.append(val[1])    
            except: continue
plt.scatter(x_list, y_list)
plt.show()

"""
#   Varying offsets
x_list = []
y_list = []
for offset in npy.linspace(0., 1., 20):
    F = get_F(f1, level=0.8, offset=offset)
    try:
        val = Newton(F, 0.8, 0.8, N=100)
        x_list.append(val[0])
        y_list.append(val[1])    
    except: continue
plt.scatter(x_list, y_list)
plt.show()
"""