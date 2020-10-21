""""
#   THIRD-PARTY LIBRARIES
from typing import Callable
import autograd
import autograd.numpy as np
import numpy
#import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display


#   GLOBAL VARIABLES
eps = 10**-6
N = 10**4


plt.rcParams['figure.figsize'] = [10, 10] # [width, height] (inches). 

# ---------------------

#Display monitoring
def display_contour(f, x, y, levels):
    X, Y = numpy.meshgrid(x, y)
    Z = f(X, Y)
    fig, ax = plt.subplots()
    contour_set = plt.contour(
        X, Y, Z, colors="grey", linestyles="dashed", 
        levels=levels 
    )
    ax.clabel(contour_set)
    plt.grid(True)
    plt.xlabel("$x_1$") 
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal")

#grad function
def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f

#jacobian function
def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f

#Parameters
N = 100
eps = 10**(-3) #A justifier avec fin Calcul diff II (nb flottants & erreurs)
c = 0.8

#Task 1
def Newton(F, x0, y0, eps=eps, N=N): #A vectoriser

    J_F = J(F)

    for i in range(N):

        J_Finv = numpy.linalg.inv(J_F(x0, y0))
        #Application formule de récurrence
        F_0 = F(x0, y0)
        x = x0 - numpy.dot(J_Finv, np.array( [F_0[0], F_0[1]] ))[0]
        y = y0 - numpy.dot(J_Finv, np.array( [F_0[0], F_0[1]] ))[1]

        if numpy.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

#Task 2
def f1(x, y):
    return 3.0 * x * x - 2.0 * x * y + 3.0 * y * y

def F(x, y):
    return np.array([f1(x, y) - c, x - y])

xf, yf = Newton(F, 0.8, 0.8)
print((xf, yf))

#A terminer avec pleins d'essais

display_contour(
    f1, 
    x=numpy.linspace(-1.0, 1.0, 100), 
    y=numpy.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)

#Task 3
def level_curve(f, x0, y0, delta=0.1, eps=eps, N=100):

    contour = numpy.zeros((2,N))
    contour[0, 0], contour[1, 0] = x0, y0
    xi, yi = x0, y0

    for i in range(1, N):

        def F(x, y):
            dist = (x - xi)**2 + (y - yi)**2
            return np.array([f1(x, y) - c, dist - delta**2])

        delta1_f = grad(f)(xi, yi)[0]
        delta2_f = grad(f)(xi, yi)[1]
    
        tang = ( np.array([delta2_f, -delta1_f]) / numpy.sqrt(delta1_f**2 + delta2_f**2) ) * delta #Vecteur tangent de départ
        xf = xi + tang[0]
        yf = yi + tang[1]
        xf, yf = Newton(F, xf, yf, eps, N)

        contour[0, i], contour[1, i] = xf, yf
        xi, yi = xf, yf

    return contour

contour = level_curve(f1, xf, yf)
plt.scatter(contour[0,:], contour[1,:])
plt.show()

#Task 4
def intersect(v1, v2, u1, u2):
    x_v1, y_v1 = v1[0], v1[1]
    x_u1, y_u2 = u1[0], u1[1]
    x_v2, y_v2 = v2[0], v2[1]
    x_u2, y_u2 = u2[0], u2[1]








# Amélioration de level_curve
def level_curve(f, x0, y0, delta=0.1, eps=eps, N=100):
    contour = numpy.zeros((2, N))
    contour[0, 0], contour[1, 0] = x0, y0
    xi, yi = x0, y0

    for i in range(1, N):
        def F(x, y):
            dist = (x - xi) ** 2 + (y - yi) ** 2
            return np.array([f1(x, y) - c, dist - delta ** 2])

        delta1_f = grad(f)(xi, yi)[0]
        delta2_f = grad(f)(xi, yi)[1]

        tang = (np.array([delta2_f, -delta1_f]) / numpy.sqrt(
            delta1_f ** 2 + delta2_f ** 2)) * delta  # Vecteur tangent de départ
        xf = xi + tang[0]
        yf = yi + tang[1]
        xf, yf = Newton(F, xf, yf, eps, N)

        contour[0, i], contour[1, i] = xf, yf
        xi, yi = xf, yf


    return contour


"""


def check_intersection(segment_1_1:tuple, segment_1_2:tuple, segment_2_1:tuple, segment_2_2:tuple) -> float or None:

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

    #   Analytical resolution to find the possible intersection of the infinite lines
    x = ((x_1 * delta_1 - y_1) - (x_3 * delta_2 - y_3)) / (delta_1 - delta_2) # résolution analytique

    #   Condition for intersection of the SEGMENTS
    if x_1 <= x <= x_2 and x_3 <= x <= x_4:
        return x

    return None


x = check_intersection(
    (1, 1),
    (7, 2),
    (6, 3),
    (3, 4)
)
print(x)

