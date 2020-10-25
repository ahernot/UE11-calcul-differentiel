# Third-Party Libraries
# ---------------------

# Autograd & Numpy
import autograd
import autograd.numpy as np
import numpy #Besoin réel ?

# Pandas
import pandas as pd

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10] # [width, height] (inches). 

# Jupyter & IPython
from IPython.display import display

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
    plt.xlabel("$x$") 
    plt.ylabel("$y$")
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

#Task 1
def Newton(F, x0, y0, eps=eps, N=N):

    J_F = J(F) #Jacobian matrix

    for i in range(N):

        J_Finv = numpy.linalg.inv(J_F(x0, y0))
        #Relations of recurrence
        x = x0 - numpy.dot(J_Finv, np.array( [F(x0, y0)[0], F(x0, y0)[1]] ))[0]
        y = y0 - numpy.dot(J_Finv, np.array( [F(x0, y0)[0], F(x0, y0)[1]] ))[1]

        #Testing end condition
        if numpy.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

#Task 2

def f1(x, y):
    return 3.0 * x * x - 2.0 * x * y + 3.0 * y * y

def f2(x, y):
    return (x - 1)**2 + (x - y**2)**2

def F1(x, y):
    return np.array([f1(x, y) - c, x - y])

def F1_bis(x, y):
    return np.array([f1(x, y) - c, 2*x - y])

def F1_bis_bis(x, y):
    return np.array([f1(x, y) - c, x**2 + y])

def F2(x, y):
    return np.array([f2(x, y) - c, x - 1])

def NewtonVisual(F, x0, y0, eps=eps, N=N):

    J_F = J(F) #Jacobian matrix
    tab = np.empty((2,N)) #Memory tab
    tab[0, 0], tab[1, 0] = x0, y0

    for i in range(1, N):

        J_Finv = numpy.linalg.inv(J_F(x0, y0))
        #Relations of recurrence
        x = x0 - numpy.dot(J_Finv, np.array( [F(x0, y0)[0], F(x0, y0)[1]] ))[0]
        y = y0 - numpy.dot(J_Finv, np.array( [F(x0, y0)[0], F(x0, y0)[1]] ))[1]

        tab[:, i] = [x, y]

        #Testing end condition
        if numpy.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return tab[:, :i]
        
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

#Modification of the initial point

#---> with F1 (c = 0.8 and x1 = x2)

c = 0.8
tabTest = np.array([ [0.8,   0, -0.3,  0.5, -0.25, -0.8], 
                     [0.8, 0.8, -0.8, 0.25,     0,    0] ])
tabX_F1 = numpy.linspace(-1.0, 1.0, 100)
tabY_F1 = tabX_F1

#First test
display_contour(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)
plt.plot(tabX_F1, tabY_F1, color='black', linestyle='-.', label='x2 = x1')
xi, yi = tabTest[0, 0], tabTest[1, 0]
tabX, tabY = NewtonVisual(F1, xi, yi)[0, :], NewtonVisual(F1, xi, yi)[1, :]
plt.plot(tabX, tabY, 'o-')
plt.legend(loc='best')
plt.show()

#Second test
display_contour(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)
plt.plot(tabX_F1, tabY_F1, color='black', linestyle='-.', label='x2 = x1')

for i in range(1, np.shape(tabTest)[1] ):

    xi, yi = tabTest[0, i], tabTest[1, i]
    tabX, tabY = NewtonVisual(F1, xi, yi)[0, :], NewtonVisual(F1, xi, yi)[1, :] #A optimiser (2 calculs)
    plt.plot(tabX, tabY, 'o-')

plt.legend(loc='best')
plt.show()

#---> with F1_bis (c = 0.8 and 2x1 = x2)

tabX_F1_bis = np.linspace(-0.5, 0.5, 100)
tabY_F1_bis = tabX_F1_bis*2
tabTest = np.array([ [0.8,   0, -0.3,  -0.5], 
                     [0.8, 0.8, -0.8, -0.25] ])
display_contour(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=10 # 10 levels, automatically selected
)
plt.plot(tabX_F1_bis, tabY_F1_bis, color='black', linestyle='-.', label='x2 = 2*x1')

for i in range( np.shape(tabTest)[1] ):

    xi, yi = tabTest[0, i], tabTest[1, i]
    tabX, tabY = NewtonVisual(F1_bis, xi, yi)[0, :], NewtonVisual(F1_bis, xi, yi)[1, :] #A optimiser (2 calculs)
    plt.plot(tabX, tabY, 'o-')

plt.legend(loc='best')
plt.show()

#---> with F1_bis_bis (c = 1 and x1**2 = x2)

c = 0.5
tabX_F1_bis_bis = np.linspace(-1.0, 1.0, 100)
tabY_F1_bis_bis = -tabX_F1_bis_bis**2
tabTest = np.array([ [-0.6,  0.7, 0.5, -0.8], 
                     [-0.8, -0.8, 0.25, 0] ])
display_contour(
    f1, 
    x=np.linspace(-1.0, 1.0, 100), 
    y=np.linspace(-1.0, 1.0, 100), 
    levels=15 # 10 levels, automatically selected
)
plt.plot(tabX_F1_bis_bis, tabY_F1_bis_bis, color='black', linestyle='-.', label='x2 = -(x1)²')

for i in range( np.shape(tabTest)[1] ):

    xi, yi = tabTest[0, i], tabTest[1, i]
    tabX, tabY = NewtonVisual(F1_bis_bis, xi, yi)[0, :], NewtonVisual(F1_bis_bis, xi, yi)[1, :] #A optimiser (2 calculs)
    plt.plot(tabX, tabY, 'o-')

plt.legend(loc='best')
plt.show()

#---> with F2 (c = 0.125 and x1 = 1)

c = 0.125
tabX_F2 = np.array([1.0]*100)
tabY_F2 = np.linspace(-2.0, 2.0, 100)
tabTest = np.array([ [1.5,    0, 2.5, -0.5], 
                     [0.5, -1.5, 1.5,  1.5] ])
display_contour(
    f2, 
    x=np.linspace(-1.0, 3.0, 100), 
    y=np.linspace(-2.0, 2.0, 100), 
    levels=[2**i for i in range(-3, 8)] # levels: [0.125, 0.25, ..., 64, 128]
)
plt.plot(tabX_F2, tabY_F2, color='black', linestyle='-.', label='x1 = 1')

for i in range( np.shape(tabTest)[1] ):

    xi, yi = tabTest[0, i], tabTest[1, i]
    tabX, tabY = NewtonVisual(F2, xi, yi)[0, :], NewtonVisual(F2, xi, yi)[1, :] #A optimiser (2 calculs)
    plt.plot(tabX, tabY, 'o-')

plt.legend(loc='best')
plt.show()

#Task 3
def level_curve(f, x0, y0, delta=0.1, eps=eps, N=100):

    contour = np.zeros((2,N))
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

"""
xf, yf = Newton(F1, 0.8, 0.8)
contour = level_curve(f1, xf, yf)
plt.scatter(contour[0,:], contour[1,:])
plt.show()
"""


#Task 4
def intersect(u1, u2, v1, v2):

    #Extraction coordonnées
    x_u1, y_u1 = u1[0], u1[1]
    x_u2, y_u2 = u2[0], u2[1]

    x_v1, y_v1 = v1[0], v1[1]
    x_v2, y_v2 = v2[0], v2[1]

    A = numpy.array([ [x_u2 - x_u1, -( x_v2 - x_v1 )],
                      [y_u2 - y_u1, -( y_v2 - y_v1 )] ]) #Matrice représentative du système
    try:
        Ainv = numpy.linalg.inv(A)
        sol = numpy.dot( Ainv, numpy.array([ x_v1 - x_u1, y_v1 - y_u1]))
        t_u, t_v = sol[0], sol[1]

        if (0 <= t_u <= 1) and (0 <= t_v <= 1):
            return True
        else:
            return False
    except numpy.linalg.LinAlgError: #Si infinité de solutions (superposition)
        return True

def level_curve_auto(f, x0, y0, delta=0.1, eps=eps, N=100):

    contour = numpy.zeros((2,N))
    contour[0, 0], contour[1, 0] = x0, y0
    xi, yi = x0, y0
    finTour = False
    i = 0 #Variable de sécurité (boucle infinie)

    while not(finTour) and (i < N):

        def F(x, y):
            dist = (x - xi)**2 + (y - yi)**2
            return np.array([f1(x, y) - c, dist - delta**2])

        #Vecteur tangent de départ
        delta1_f = grad(f)(xi, yi)[0]
        delta2_f = grad(f)(xi, yi)[1]
        tang = ( np.array([delta2_f, -delta1_f]) / numpy.sqrt(delta1_f**2 + delta2_f**2) ) * delta

        #Création point de départ pour la méthode de Newton
        xf = xi + tang[0]
        yf = yi + tang[1]
        xf, yf = Newton(F, xf, yf, eps, N)

        contour[0, i], contour[1, i] = xf, yf #Ajout point de la courbe

        if (i > 2) and ( intersect( contour[:,0], contour[:,1], contour[:,i-1], contour[:,i]) ): #Condition de fin de tour
            finTour = True

        xi, yi = xf, yf #Passage au point suivant
        i += 1

    return contour

"""
print(intersect( (0, 2), (0, 1), (0, 1), (0, 2) ))  
contour = level_curve_auto(f1, xf, yf)
plt.scatter(contour[0,:], contour[1,:])
plt.show()
"""