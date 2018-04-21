#!/usr/bin/env python
from __future__ import division
import matplotlib

matplotlib.use('TkAgg')
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from scipy import integrate

"""Info: Code solves lane-emden differential equation for following boundary conditions:
        psi(0) = 0; dpsi/dzi(zi=0) = 0. 

        The external pressure is determine from the following expression for the
        critical mass: 1.18*sigma^4/sqrt(G^3*P_o).
        A temperature of 20K is assume for the medium.
        The sound of speed is found from: sqrt(RT/mu); where R is the gas constant and mu is the
        molecular weight.

        After solving the ODE, the code will solve for dimensionless mass 'm'
        given an M and P_o. The value of rho_c is estimated. 
        Then, the code solves for zi_o for a given rho_c until the desired 
        TOTAL critical mass is found. This will give us the value of the BE boundary.   

By Aisha Mahmoud-Perez
Star Formation
"""
mu = 1.5  # any value between 1-2
T = 10  # temperature of gas in K
R = 8.31E7  # gas constant in cgs units
G = 6.67E-8
k = 1.3806488e-16
m_H2 = 1.672621777e-24
pc = 3.08567758e18




def f(x):

    res = 4 *np.pi*pow(x,2)*pow(1+pow(x/2.88,2),-1.47)
    return res

X = np.arange(0.0,10.0,0.001)


def F(x):
    res = np.zeros_like(x)
    for i,val in enumerate(x):
        y,err = integrate.quad(f,0,val)
        res[i]=y
    return res



theta_points = np.array(X)


# list_for_fit_theta =[]
# list_for_fit_d_theta =[]
# for i in range(0,len(X)):
#
#    list_for_fit_theta.append((X[i],F(X[i]) ))

# get x and y vectors
x = X
y = F(X)
# calculate polynomial
z = np.polyfit(x,y, 8)
ffit = np.poly1d(z)
for i in range(0,len(x)):
    print(X[i], y[i])
plt.plot(X,F(X),'o', x, ffit(x))

plt.show()

