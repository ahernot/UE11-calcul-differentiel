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
    """
    This function calculates a level curve of a function, with N steps.

    :param f: The function to calculate for
    :param x0: The initial x-axis coordinate
    :param y0: The initial y-axis coordinate
    :param delta: The step size for the calculation
    :param eps: The acceptable precision of the algorithm
    :param N: The maximum number of iterations (Newton resolution algorithm) and of steps

    :returns: The array of points making up the discrete calculation of the level curve
    """

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
    """
    This function automatically calculates a level curve of a function.

    :param f: The function to calculate for
    :param x0: The initial x-axis coordinate
    :param y0: The initial y-axis coordinate
    :param delta: The step size for the calculation
    :param eps: The acceptable precision of the algorithm
    :param N: The maximum number of iterations for the Newton resolution algorithm
    :param max_iter: The maximum number of steps

    :returns: The array of points making up the discrete calculation of the level curve
    """

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

def gamma(t:np.ndarray, P1:tuple or np.ndarray, P2:tuple or np.ndarray, u1:tuple or np.ndarray, u2:tuple or np.ndarray) -> np.ndarray:
    """
    This function generates a polynomial interpolation between two points in a 2D plane.

    :param t: The interpolated curve's normalized parameter
    :param P1: The first point tuple
    :param P2: The second point tuple
    :param u1: The first derivative vector
    :param u2: The second derivative vector

    :returns: The interpolated points' array
    """

    #   1. Unpacking the variables for easier processing
    u11, u12 = u1
    u21, u22 = u2
    x1, y1 = P1
    x2, y2 = P2

    #   2. Calculating the determinant of the (u1, u2) couple
    denom = (u12 * u21) - (u11 * u22) 

    #   3.1. Using the second-degree polynomial interpolation method when possible 
    if not np.isclose(denom, 0):
        #   3.1.1. Calculating the values of k1 and k2
        k1 = 2 * ( ((y2 - y1) * u21) - ((x2 - x1) * u22) ) / denom
        k2 = 2 * ( ((x2 - x1) * u12) - ((y2 - y1) * u11) ) / denom

        #   3.1.2. Calculating the explicit values of the a, b, c, d, e, f parameters
        a = x1
        b = k1 * u11
        c = k2 * u21 + x1 - x2
        d = y1
        e = k1 * u12
        f = k2 * u22 + y1 - y2

        #   3.1.3. Calculating the values of the x, y variables
        x = a + b * t + c * t**2
        y = d + e * t + f * t**2

    #   3.2. Switching to a linear interpolation method (ignoring the derivative vectors) if u1 and u2 are parallel vectors
    else:
        #   3.2.1. Calculating the explicit values of the a, b, c, d parameters
        a = x1
        b = x2 - x1
        c = y1
        d = y2 - y1
        
        #   3.2.2. Calculating the values of the x, y variables
        x = a + b * t
        y = c + d * t

    #   4. Concatenating the results into a single coordinates array
    points_interpolated = np.empty((2, t.shape[0]))
    points_interpolated[0, :] = x
    points_interpolated[1, :] = y
    
    return points_interpolated

"""
#   Initialising the plot
plt.figure()
plt.xlim(-1, 6)
plt.ylim(-3, 4)

#   Defining the points
P1 = (0, 0)
P2 = (5, 0)

#   Defining the derivatives
u1 = (1, 1)
u2 = (2, -3)

#   Plotting the points and derivative unit vectors
u1_norm = np.linalg.norm(u1)
u2_norm = np.linalg.norm(u2)
plt.plot([P1[0], P1[0] + u1[0]/u1_norm], [P1[1], P1[1] + u1[1]/u1_norm], c='gray')
plt.plot([P2[0], P2[0] + u2[0]/u2_norm], [P2[1], P2[1] + u2[1]/u2_norm], c='gray')
plt.scatter(*P1, c='red')
plt.scatter(*P2, c='red')

#   Generating an array of discrete normalized values to calculate the interpolation results for
t = np.linspace(0, 1, 30)

#   Running the gamma function
points = gamma(t, P1, P2, u1, u2)

#   Plotting the results
plt.plot(points[0], points[1], c='blue')
plt.show()
"""



def level_curve_3(f: Callable, x0: float or int, y0: float or int, delta:float or int=0.1, eps: float or int=eps, N: int=100, max_iter:int=10**4, oversampling:int = 1) -> np.ndarray:
    """
    This function automatically calculates a level curve of a function and interpolates-in points to visually refine the output.

    :param f: The function to calculate for
    :param x0: The initial x-axis coordinate
    :param y0: The initial y-axis coordinate
    :param delta: The step size for the calculation
    :param eps: The acceptable precision of the algorithm
    :param N: The maximum number of iterations for the Newton resolution algorithm
    :param max_iter: The maximum number of steps
    :param oversampling: The number of points to calculate per step (1 for no interpolation)

    :returns: The array of points making up the interpolated discrete calculation of the level curve
    """

    #   0. Calculating the number of interpolated points to add between each calculated point
    points_to_oversample = oversampling - 1
    
    #   1. Running the basic level_curve function to generate the starting points
    points_array = level_curve_2(f, x0, y0, delta, eps, N, max_iter)

    #   2. Fast-track return to avoid running empty loops
    if oversampling == 1: return points_array

    #   3. Generating the final array
    starting_width = points_array.shape[1]
    final_width = starting_width + (starting_width - 1) * points_to_oversample
    final_array = np.empty((2, final_width))
    final_array[:, -1] = points_array[:, -1]
    
    #   4. Filling the final array
    for point_id in range(starting_width - 1):

        #   4.1. Extracting the bounds for the interpolation
        point_start = points_array[:, point_id] # P1
        point_stop = points_array[:, point_id + 1] # P2

        #   4.2. Getting the adjacent points when possible, to calculate the derivatives
        #   4.2.1. To the left
        if point_id == 0: point_left = point_start
        else: point_left = points_array[:, point_id - 1]
        #   4.2.2. To the right
        if point_id == starting_width - 2: point_right = point_stop
        else: point_right = points_array[:, point_id + 2]

        #   4.3.1. Calculating the slopes, to the left and to the right
        slope_left = (point_stop[1] - point_left[1]) / (point_stop[0] - point_left[0])
        slope_right = (point_right[1] - point_start[1]) / (point_right[0] - point_start[0])
        #   4.3.2. Calculating the derivative vectors (non-unit vectors, doesn't matter), to the left and to the right
        der_start = np.array([1., slope_left])
        der_stop = np.array([1., slope_right])
        
        #   4.4. Generating the array of normalised parameters used to calculate the interpolated points
        t_array = np.linspace(0, 1, points_to_oversample)

        #   4.5. Calculating the array of interpolated points
        points_interpolated = gamma(t_array, point_start, point_stop, der_start, der_stop)
        
        #   4.6. Adding the calculated points to the final array
        final_array[:, point_id * oversampling] = point_start
        final_array[:, point_id * oversampling + 1 : (point_id + 1) * oversampling] = points_interpolated

    return final_array



plt.figure()
plt.xlim(-1, 1)
plt.ylim(-1, 1)

#   Initialising for the level_curve_2 function, using f1
c = 0.8
F = get_F(f1, level=0.8)
x0, y0 = Newton(F, 0.8, 0.8)

#   Running the level_curve_2 function
contour_old = level_curve_2(f1, x0, y0)
contour_new = level_curve_3(f1, x0, y0, oversampling = 10)

#   Reference display
GFunc.display_contour_inline(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)

#   Plotting the results
plt.plot(contour_old[0], contour_old[1], c='gray')
plt.plot(contour_new[0], contour_new[1], c='blue')
plt.show()
