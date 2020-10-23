#   THIRD-PARTY LIBRARIES
from typing import Callable
import autograd
import autograd.numpy as np
import numpy
#import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display




# Display monitoring
def display_contour(f: Callable, x: float or int, y: float or int, levels):
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


# grad function
def grad(f):
    g = autograd.grad

    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])

    return grad_f


# jacobian function
def J(f):
    j = autograd.jacobian

    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T

    return J_f






# Parameters
N = 100
eps = 10 ** (-3)  # A justifier avec fin Calcul diff II (nb flottants & erreurs)
c = 0.8


# Task 1
def Newton(f:Callable, x0:float or int, y0:float or int, eps:float or int=eps, N:int=N):
    """
    This function...
    :param f:
    :param x0:
    :param y0:
    :param eps:
    :param N: Maximum number of iterations
    :return:
    """

    #   1. Calcul de la matrice jacobienne de F
    Jf = J(f)

    for i in range(N):

        #   2.1. Inversion de la matrice jacobienne
        Jf_inv = numpy.linalg.inv( Jf(x0, y0) )
        f0 = f(x0, y0)

        #   2.2. Application de la formule de récurrence (Newton)
        f_dot = numpy.dot( Jf_inv, np.array([f0[0], f0[1]]) )
        #   2.3.
        x = x0 - f_dot[0]
        y = y0 - f_dot[1]

        #   2.4. Exiting the function once the desired precision is reached
        if numpy.sqrt((x - x0) ** 2 + (y - y0) ** 2) <= eps:
            return x, y

        x0, y0 = x, y

    else:
        raise ValueError(f'No convergence in {N} steps.')





# Task 2
def f1(x, y):
    return 3.0 * x * x - 2.0 * x * y + 3.0 * y * y


def F(x, y):
    return np.array([f1(x, y) - c, x - y])


xf, yf = Newton(F, 0.8, 0.8)
print((xf, yf))
# À terminer avec pleins d'essais





display_contour(
    f1,
    x=numpy.linspace(-1.0, 1.0, 100),
    y=numpy.linspace(-1.0, 1.0, 100),
    levels=10  # 10 levels, automatically selected
)




# Task 3
def level_curve(f:Callable, x0, y0, delta=0.1, eps=eps, N=100):
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








####################


def check_intersection(segment_1_1:tuple, segment_1_2:tuple, segment_2_1:tuple, segment_2_2:tuple) -> tuple or None:

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
    if x >= max(x_1, x_3) and x <= min(x_2, x_4): # if x_1 <= x <= x_2 and x_3 <= x <= x_4:
        #y = delta_1 * (x - x_1) + y_1
        #return x, y
        return True

    return False



# Task x
def level_curve_1(f:Callable, x0, y0, delta=0.1, eps=eps) -> list:
    overlaps = 0 # overlap counter
    overlaps_to_break = 1 # 2 # number of overlaps required to break the loop

    contour = [(x0, y0)]
    xi, yi = x0, y0

    while True:

        def F(x:float, y:float) -> numpy.ndarray: #change to have inline argument *[x, y] with numpy???????
            dist = (x - xi) ** 2 + (y - yi) ** 2
            return np.array([f1(x, y) - c, dist - delta ** 2])



        delta_f = grad(f)(xi, yi)
        delta_f = grad(f)(xi, yi)

        grad_f = numpy.dot( grad(f)(xi, yi) , numpy.array([1, -1]))

        tang = (np.array([delta2_f, -delta1_f]) / numpy.sqrt(
            delta1_f ** 2 + delta2_f ** 2)) * delta  # Vecteur tangent de départ

        xf = xi + tang[0]
        yf = yi + tang[1]
        xf, yf = Newton(F, xf, yf, eps, N)


        #   Adding the next calculated point to the contour
        contour.append((xf, yf))


        #   Check if intersects with first segment
        if len(contour) > 2:
            intersects = check_intersection(contour[0], contour[1], contour[-2], contour[-1]) ## to implement directly because doesn't need to re-compute the first segment every time

            if intersects:
                overlaps += 1
            else:
                overlaps = 0

        if overlaps >= overlaps_to_break:
            break

        xi, yi = xf, yf


    return contour



"""
def gamma(t, P1:tuple, P2:tuple, u1:tuple, u2:tuple) -> tuple:

    x1, x2 = P1
    y1, y2 = P2

    u11, u12 = u1
    u21, u22 = u2

    a = x1
    d = y1
    #condition si u12 ou u22 = 0

    # et, par résolution du système linéaire :
    e = ((2*y2 - 2*y1)*(u21/u22) + 2*(x2-x1)) / (3*u11/u12 + u21/u22) #refaire le calcul, au cas où
    b = e * u11 / u12
    c = x2 - x1 - b
    f = y2 - y1 - e

    x = a + b * t + c * t ** 2
    y = d + e * t + f * t ** 2

    return (x, y)
"""

###
t = numpy.linspace(0, 1, 100)

"""
ff = gamma(
    t,
    P1 = (0, 0),
    P2 = (5, 5),
    u1 = (3*(10**-0.5), 1*(10**-0.5)),
    u2 = (3*(34**-0.5), -5*(34**-0.5))
)
"""


plt.figure()
plt.scatter([x for x in contour[0]], [y for y in contour[1]])
plt.show()
