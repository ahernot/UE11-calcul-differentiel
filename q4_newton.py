#   THIRD-PARTY LIBRARIES
from typing import Callable

import numpy as npy # importing the numpy functions which are not overwritten by autograd.numpy (and thus are otherwise missing)

import autograd
import autograd.numpy as np

#import pandas as pd
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from IPython.display import display


#   GLOBAL VARIABLES
eps = 10**-6
N = 10**4




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

    def linarray_2d(x_range:tuple, y_range:tuple, steps:tuple) -> np.ndarray:
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


    def check_intersection(self, segment_1_1: tuple, segment_1_2: tuple, segment_2_1: tuple, segment_2_2: tuple) -> float or None:
        # DOESN'T WORK FOR VERTICAL SEGMENTS

        #   Unpacking the variables for the first segment
        x_1, y_1 = segment_1_1
        x_2, y_2 = segment_1_2
        if x_1 > x_2:  x_1, y_1, x_2, y_2 = x_2, y_2, x_1, y_1

        #   Unpacking the variables for the second segment
        x_3, y_3 = segment_2_1
        x_4, y_4 = segment_2_2
        if x_3 > x_4:  x_3, y_3, x_4, y_4 = x_4, y_4, x_3, y_3

        #   Calculating the gradients
        delta_1 = (y_2 - y_1) / (x_2 - x_1)
        delta_2 = (y_4 - y_3) / (x_4 - x_3)

        #   Checking if parallel
        if delta_1 == delta_2:
            y0_1 = y_1 - x_1 * delta_1
            y0_2 = y_3 - x_3 * delta_2

            #   Parallel and overlapping
            if y0_1 == y0_2:

                #   careful if distinct segments!!!!!!!!!!!!!! TO FINISH
                if x_2 <= x_3:
                    return True
                return False

            #   Parallel and distinct
            else:  # statement not needed
                return False

        #   Analytical resolution to find the possible intersection of the infinite lines
        x = ((x_1 * delta_1 - y_1) - (x_3 * delta_2 - y_3)) / (delta_1 - delta_2)  # résolution analytique

        #   Condition for intersection of the SEGMENTS
        if x_1 <= x <= x_2 and x_3 <= x <= x_4:
            # y = delta_1 * (x - x_1) + y_1
            # return x, y
            return True

        return False

DFunc = DiffFunctions()
GFunc = GeomFunctions()





#   FUNCTIONS
def newton_anatole(F:Callable, x0:float, y0:float, eps:float=eps, N:int=N, debug:bool=False) -> np.ndarray:

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

    #   3. Defining an iteration counter
    iter_counter = 0

    #   4. Running the method in a loop to refine the calculation
    while True:

        #   4.1. Inverting F's jacobian matrix
        try: jacF_inv = npy.linalg.inv( jacF( *(X0.tolist()) ) )
        except npy.linalg.linalg.LinAlgError: print('Error encountered: the JacF matrix is singular.')

        #   4.2. Dot product between jacF and F(X0)
        F_dot = npy.dot( jacF_inv, F( *(X0.tolist()) ) )

        #   4.3. Computing the new X point
        X = X0 - F_dot

        #   4.4. Exiting the function once the desired precision is reached
        if npy.linalg.norm( X - X0 ) <= eps:
            if debug: print(f'Solution found for (x,y) = {tuple(X.tolist())}, with value f(x,y) = {F(*(X.tolist()))[0]}\nIterations required for calculation: {iter_counter} / {N} ({round(iter_counter / N * 100, 1)}%)')  # DEBUG
            return X

        #   4.5. Incrementing the iteration counter and performing end-of-loop actions
        iter_counter += 1
        X0 = X.copy()

        #   4.6. Breaking the loop when the max number of iterations is reached
        if iter_counter >= N:
            break
    

    #   3. Raising an error when no solution is found and the max number of iterations is exceeded
    raise ValueError(f'No convergence in {N} steps.')

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
            raise ValueError('The function to solve for has got a singular jacobian matrix.')

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





def f0(x: float or int, y: float or int) -> float:
    return x + y

def f1(x: float or int, y: float or int) -> float:
    return 3.0 * x * x - 2.0 * x * y + 3.0 * y * y

c = 0
def F(x: float or int, y: float or int) -> np.ndarray:
    return np.array([ f1(x, y) - c , x - y ])


val = Newton(F, 0.8, 0.8)
print(val)





def level_curve(f: Callable, x0: float or int, y0: float or int, delta:float or int=0.1, eps: float or int=eps, N: int=100) -> np.ndarray:

    #   1. Initialising the contour
    contour = npy.empty((2, N))
    X0 = np.array([x0, y0])
    contour[:, 0] = X0

    print(contour)

    #   2.1. Initialising the loop's variables
    Xi = X0.copy()

    #   2.2. Looping
    for index in range(1, N):

        #   2.2.0. Defining the F conditioning function
        def F(x, y):
            X = np.array([x, y])

            dist = npy.linalg.norm( X - Xi )
            condition = dist - delta ** 2

            return np.array([f1(x, y) - c, condition])

        #   2.2.1. Computing a tangent vector to the curve
        delta_f = DFunc.gradient(f)( *(Xi.tolist()) )
        delta_f_mod = delta_f[::-1] * np.array([1, -1])
        tang_f = delta * delta_f_mod / npy.linalg.norm(delta_f) # vecteur tangent de départ

        #   2.2.2. Computing Xf
        Xf = Xi + tang_f

        #   2.2.3. Finding the nearest suitable intersection using Newton's method
        xf, yf = Xf.tolist()
        Xf = newton(F, xf, yf, eps, N)

        #   2.2.4. Adding the point to the contour
        contour[:, index] = xf, yf

        #   2.2.5. End-of-loop actions
        Xi = Xf.copy()

    return contour

"""
display_contour(
    f1,
    x=numpy.linspace(-1.0, 1.0, 100),
    y=numpy.linspace(-1.0, 1.0, 100),
    levels=10  # 10 levels, automatically selected
)
"""

#level_curve()

#3 CONSECUTIFS AVEC 1, 2, 3!!!!!!!!







x_range = (-5, 6)
x_step  = 1
y_range = (-5, 6)
y_step  = 1

