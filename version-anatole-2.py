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


#   Loading the auxiliary classes for easy use
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


"""
plt.figure()
plt.xlim(-1, 1)
plt.ylim(-1, 1)


GFunc.display_contour_inline(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)
"""

"""
##   Varying the seed point
x_list, y_list = [], [] # results
xp_list, yp_list = [], [] # seed points

F = get_F(f1, level=0.8)
# very ugly because Newton isn't vectorized
for line in lin_array:
    for col in lin_array:
        for point in col:

            #   Adding the seed points
            xp_list.append(point.tolist()[0])
            yp_list.append(point.tolist()[1])

            #   Adding the results (if available)
            try:
                val = Newton(F, *(point.tolist()))
                x_list.append(val[0])
                y_list.append(val[1])    
            except: continue
 
plt.scatter(xp_list, yp_list, c='red', marker='x', linewidths=0.5) # plotting the seed points
plt.scatter(x_list, y_list, c='blue') # plotting the results
plt.show()
"""

"""
#   Varying offsets
x_list, y_list = [], [] # results

for offset in npy.linspace(0., 1., 20):
    #   Calculating F with the given offset
    F = get_F(f1, level=0.8, offset=offset)

    #   Adding the results (if available)
    try:
        val = Newton(F, 0.8, 0.8, N=100)
        x_list.append(val[0])
        y_list.append(val[1])    
    except: continue

plt.scatter(x_list, y_list, c='blue') # plotting the results
plt.show()
"""





def level_curve(f: Callable, x0: float or int, y0: float or int, delta:float or int=0.1, eps: float or int=eps, N: int=100) -> np.ndarray:

    #   1. Initialising the contour
    contour = npy.empty((2, N))
    contour[:, 0] = np.array([x0, y0])

    #   2.1. Initialising the loop's variables
    xi, yi = x0, y0

    #   2.2. Looping
    for index in range(1, N):

        #   2.2.0. Defining the F conditioning function
        def F(x, y):
            condition = np.sqrt((x - xi)**2 + (y - yi)**2) - delta # intersection between the circle and the level line
            return np.array([f(x, y) - c, condition])

        #   2.2.1. Computing a tangent vector to the curve
        delta_f = DFunc.gradient(f)( xi, yi )
        delta_f_mod = delta_f[::-1] * np.array([1, -1])
        tang_f = delta * delta_f_mod / npy.linalg.norm(delta_f)

        #   2.2.2. Computing Xf (in order to get the right intersection)
        xf = xi + tang_f[0]
        yf = yi + tang_f[1]

        #   2.2.3. Finding the nearest suitable intersection using Newton's method
        xi, yi = Newton(F, xf, yf, eps, N)
        #try: xi, yi = Newton(F, xf, yf, eps, N)
        #except: break

        #   2.2.4. Adding the point to the contour
        contour[:, index] = np.array([xi, yi])

    return contour

"""
c = 0.8
F = get_F(f1, level=0.8)
x0, y0 = Newton(F, 0.8, 0.8)
contour = level_curve(f1, x0, y0)

plt.figure()
plt.xlim(-1, 1)
plt.ylim(-1, 1)

GFunc.display_contour_inline(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)


plt.scatter(contour[0].tolist(), contour[1].tolist())
plt.show()
"""






# From https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

def ccw(A:tuple or list or np.ndarray, B:tuple or list or np.ndarray, C:tuple or list or np.ndarray):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_intersection(A1:tuple or list or np.ndarray, A2:tuple or list or np.ndarray, B1:tuple or list or np.ndarray, B2:tuple or list or np.ndarray):
    return ccw(A1, B1, B2) != ccw(A2, B1, B2) and ccw(A1, A2, B1) != ccw(A1, A2, B2)



def level_curve_2(f: Callable, x0: float or int, y0: float or int, delta:float or int=0.1, eps: float or int=eps, N: int=100, max_iter:int=10**4) -> np.ndarray:

    #   1. Initialising the contour
    contour_list = []# = npy.empty((2, N))
    contour_list.append([x0, y0])

    #   2.1. Initialising the loop's variables
    xi, yi = x0, y0
    index = 1

    #   2.2. Looping
    while True:

        #   2.2.0. Defining the F conditioning function
        def F(x, y):
            condition = np.sqrt((x - xi)**2 + (y - yi)**2) - delta # intersection between the circle and the level line
            return np.array([f(x, y) - c, condition])

        #   2.2.1. Computing a tangent vector to the curve
        delta_f = DFunc.gradient(f)( xi, yi )
        delta_f_mod = delta_f[::-1] * np.array([1, -1])
        tang_f = delta * delta_f_mod / npy.linalg.norm(delta_f)

        #   2.2.2. Computing Xf (in order to get the right intersection)
        xf = xi + tang_f[0]
        yf = yi + tang_f[1]

        #   2.2.3. Finding the nearest suitable intersection using Newton's method
        xi, yi = Newton(F, xf, yf, eps, N)

        #   2.2.4. Adding the point to the contour
        contour_list.append([xi, yi])

        #   2.2.5 Checking for an intersection between the first segment and the last one calculated
        if index > 1:
            intersects = check_intersection(*contour_list[0:2], *contour_list[-2:])

            if intersects:
                return np.array(contour_list).transpose(1, 0)

        #   2.2.6. Breaking the loop after a set number of iterations
        if index >= max_iter:
            break

        #   2.2.7. End-of-loop actions
        index += 1

    #   4. Raising an error (when the max number of iterations is exceeded)
    raise ValueError(f'Could not find endpoint of the level curve after {max_iter} steps. Lower the calculation resolution or increase max_iter.')

"""
plt.figure()
plt.xlim(-1, 1)
plt.ylim(-1, 1)

#   Initialising for the level_curve_2 function, using f1
c = 0.8
F = get_F(f1, level=0.8)
x0, y0 = Newton(F, 0.8, 0.8)

#   Running the level_curve_2 function
contour = level_curve_2(f1, x0, y0)

#   Reference display
GFunc.display_contour_inline(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)

#   Plotting the results
plt.plot(contour[0], contour[1], c='blue')
plt.show()
"""