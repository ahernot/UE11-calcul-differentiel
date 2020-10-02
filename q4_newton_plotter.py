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

def f0(x, y):
    return x + y

def f1(x1, x2):
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 - 0.8

val = Newton(f1, 0.8, 0.8)
print(val)