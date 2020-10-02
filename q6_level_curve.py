#   THIRD-PARTY LIBRARIES
from typing import Callable
import autograd
import autograd.numpy as np
#import pandas as pd
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from IPython.display import display


#   OTHER IMPORTS
from q4_newton import Newton


#   GLOBAL VARIABLES
eps = 10**-6
N = 10**4




#   FUNCTIONS
def level_curve(f:Callable, x0:float, y0:float, delta:float=0.1, N:int=100, eps:float=eps):

    """
    This function solves the equation F(x,y) = 0 around (x0,y0) using the Newton algorithm.

    :param f: The function to solve for
    :param x0: The initial x-axis coordinate
    :param y0: The initial y-axis coordinate
    :param delta: The requested distance between the initial point and the solution
    :param eps: The acceptable precision of the algorithm
    :param N: The number of points to plot on the level line

    :returns: ???
    """

    #   1. On génère le cercle de rayon delta et de centre (x0, y0) et on cherche son intersection avec la ligne de niveau

    X, Y = x0, y0


    def generate_circle(xc:float, yc:float, radius:float):
        """
        This function generates a circle function.
        :param xc: The x-axis coordinate of the circle's center
        :param yc: The y-axis coordinate of the circle's center
        :param radius: The radius of the circle
        :returns: The circle function
        """

        def circle(angle:float) -> tuple:
            """
            This function generates a point on the circle given an angle from the x-axis (anti-clockwise, in radians).
            :param angle: The angle from the x-axis, in radians
            :retuns: The coordinates of the point on the circle
            """
            return (xc + radius * np.cos(angle), yc + radius * np.sin(angle)) #this is easier to understand with a simple drawing
        
        return circle



    for point_nb in range(N):
        
        new_circle = generate_circle(X, Y)

        # now we look for the angles so that f(circle) = c : 1-dim Newton...





        #on raisonne avec les angles??