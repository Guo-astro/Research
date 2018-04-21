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


# ----Functions-----#
def get_ext_pressure(T, m_crit):
    p = 79.4 * T ** 4 * k / m_crit ** 2  # in SI units.
    p = p * 10  # convert to bar.
    return p


def get_dmnless_mass(m_crit, p_out, c_sound):
    p_out = p_out / 10.0  # Pascals
    c_sound = c_sound  # SI units
    m = (np.sqrt(p_out) * G ** (1.5) * m_crit) / c_sound ** 4
    return m


def get_zi(m_dmnless, rho_out, rho_in, psi_diff):
    con = rho_out / (4 * np.pi * rho_in)
    zi_squared = m_dmnless / (np.sqrt(con) * psi_diff)
    return zi_squared


def get_mass(rho_in, c_sound, zi_sq, psi_diff):
    con1 = 4 * np.pi * rho_in
    con2 = c_sound ** 2 / (4 * np.pi * G * rho_in)
    m = con1 * con2 ** (1.5) * zi_sq * psi_diff
    m = m  # change to kg
    m_s = m / 1.98E33  # chanege to solar masses
    return m_s


# ----Constants----#

cs = np.sqrt(R * T / mu)  # cgs units
P_o = get_ext_pressure(T, 1)
rho_o = P_o / (cs ** 2)


# ----Solve Lane-Emden----#

def write_as_ascii(list_dic):
    """
    Write all results as text file
    """
    n_dic = len(list_dic)
    if not os.path.exists('output'): os.mkdir('output')

    for i_dic, dic in enumerate(list_dic):
        filename = 'output/Poly-n{0}.txt'.format(dic['n'])
        list_col = []
        xi = dic['xi']
        theta = dic['theta']
        d_theta = dic['d_theta']
        m = len(xi)
        for i in range(m):
            txt = '{0:016.12f}  {1:016.12f}  {2:016.12f} \n'.format(xi[i], theta[i], d_theta[i])
            list_col.append(txt)
        with open(filename, 'w') as w:
            w.writelines(list_col)

    return None


def plot_theta(list_dic):
    """
    Plot all polytropic solutions
    """
    import itertools

    n_dic = len(list_dic)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(6, 8), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.98, wspace=0.04, hspace=0.04)
    list_col = itertools.cycle(['black', 'blue', 'red', 'green', 'cyan', 'grey', 'purple', 'pink'])
    list_ls = itertools.cycle(['solid', 'dashed', 'dashdot', 'dotted'])

    for i_dic, dic in enumerate(list_dic):
        xi = dic['xi']
        theta = dic['theta']
        d_theta = dic['d_theta']

        ls = list_ls.next()
        cl = list_col.next()
        lbl = r'n=' + str(dic['n'])

        top.plot(xi, theta, linestyle=ls, color=cl, lw=2, label=lbl)
        bot.plot(xi, d_theta, linestyle=ls, color=cl, lw=2)

    top.set_xlim(-0.5, 15)
    top.set_xticklabels(())
    top.set_ylim(-0.1, 1.1)
    top.set_ylabel(r'$\theta(\xi)$')
    bot.set_xlim(-0.5, 15)
    bot.set_xlabel(r'Dimensionless Radius $\xi$')
    bot.set_ylim(-0.6, 0.05)
    bot.set_ylabel(r'$d\theta(\xi)\,/\,d\xi$')

    leg1 = top.legend(loc=1)

    plt.savefig('Lane-Emden.png')
    print(' New plot: "Lane-Emden.png" created')
    plt.close()

    return None


def get_next_func(step, func, deriv):
    """
    (theta)i+1 = (theta)i + d_xi *[ (d_theta/d_xi)i+1].
    """
    return func + step * deriv


def get_next_deriv(xi, step, deriv, theta):
    """
    (d_theta/d_xi)i+1 = (d_theta/d_xi)i - ([ (2 /xi)i . (d_theta/d_xi)i ] + theta^n ) * d_xi
    """
    return deriv - step * ((2. / xi) * deriv + np.exp(theta))


def integrate(n=1.0):
    """
    integrate the Lane Emden equations, and return the solution as a dictionary
    """

    list_xi = []
    list_theta = []
    list_d_theta = []

    # Initial Conditions
    xi_start = 1e-6
    theta = 0.0
    d_theta = 0.0

    # Stepsize
    xi_step = 1e-3

    i = 0
    list_xi.append(xi_start)
    list_theta.append(theta)
    list_d_theta.append(d_theta)

    next_theta = theta

    # while next_theta >= 0.0:
    xi_mesh = np.linspace(xi_start, 200, 200000)  # find the array of values of rho_c. Guesstimate!

    for idx in range(0, len(xi_mesh)):
        xi = xi_mesh[idx]
        next_d_theta = get_next_deriv(xi, xi_step,  d_theta, theta)
        next_theta = get_next_func(xi_step, theta, next_d_theta)

        if (next_theta > theta):
            print('theta increasing: ', i, xi, theta, next_theta, next_d_theta)

        i += 1
        xi += xi_step
        theta = next_theta
        d_theta = next_d_theta
        list_xi.append(xi)
        list_theta.append(next_theta)
        list_d_theta.append(next_d_theta)

    print(' Solution for n={0} found after {1} steps'.format(n, i))
    list_xi = np.array(list_xi)
    list_theta = np.array(list_theta)
    list_d_theta = np.array(list_d_theta)
    dic = {'n': n, 'xi': list_xi, 'theta': list_theta, 'd_theta': list_d_theta}

    return dic


y_init = [0, 0]  # initial conditions

list_dic = integrate()

list_theta = list_dic['theta']  # dump psi values here
list_d_theta = list_dic['d_theta']  # dump dpsi values here
t = list_dic['xi']  # create array of values. Equidistant integration steps.

plt.xlim([0.1, 20 ])
plt.ylim([0.001, 1.0 ])

plt.plot(t, np.exp(list_theta), 'o', t, np.power(1 + np.power(t / 2.88, 2), -1.47))

plt.loglog(t, np.exp(list_theta), 'o', t, np.power(1 + np.power(t / 2.88, 2), -1.47))

for i in range(0,len(t) -190000):
    print(t[i], np.exp(list_theta[i]))


#
# # plt.plot(x,y1,'o', x_new, y1_new)
# plt.show()
# list_for_fit_theta =[]
# list_for_fit_d_theta =[]
#
# for i in range(0,len(t)):
#
#    list_for_fit_theta.append((t[i], np.exp(psi[i])))
#    list_for_fit_d_theta.append((t[i], dpsi[i]))
#
#    # print(t[i], psi[i])
# #
#
#
#
#
# #
# theta_points = np.array(list_for_fit_theta)
# d_theta_points = np.array(list_for_fit_d_theta)
#
# # get x and y vectors
# x = theta_points[:, 0]
# y = theta_points[:, 1]
#
# y1=d_theta_points[:,1]
# # calculate polynomial
# z = np.polyfit(x,y, 100)
# f = np.poly1d(z)
# print(f)
#
# z1 = np.polyfit(x,y1, 8)
# f1 = np.poly1d(z1)
# print(f1)
# # calculate new x's and y's
#
#
#
# # ----Plot basic Lane-Emden----#
# font = {'family': 'DejaVu Sans',
#         'size': 13}
#
#
# def fit_theta(x):
#     i = 5.174e-16;
#     h = - 4.433e-13;
#     g = + 1.574e-10;
#     f = - 2.992e-08;
#     e = 3.286e-06;
#     d = - 0.000209;
#     c = 0.007358;
#     b = - 0.1254;
#     a = + 0.7485;
#     fitted = i * pow(x, 8) + h * pow(x, 7) + g * pow(x, 6) + f * pow(x, 5) + e * pow(x, 4) + d * pow(x, 3) + c * pow(x,
#                                                                                                                      2) + b * pow(
#         x, 1) + a
#     return fitted
#
# def fit_d_theta(x):
#     i = 1.193e-16;
#     h = - 9.659e-14;
#     g = + 3.172e-11;
#     f = - 5.37e-09;
#     e = + 4.87e-07;
#     d = - 2.107e-05;
#     c = + 0.0001497;
#     b = + 0.01751;
#     a = - 0.4394;
#     fitted =i*pow(x,8) +h *pow(x,7)+ g*pow(x,6) +f*pow(x,5)+ e*pow(x,4) +d*pow(x,3)+ c*pow(x,2)+b*pow(x,1)+a
#     return fitted
#
# x_new = np.linspace(x[0], x[-1], 200001)
# y_new = fit_theta(x_new)
# y1_new = fit_d_theta(x_new)
# plt.xlim([0.1, 100 ])
# plt.ylim([0.001, 1.0 ])
#
# plt.loglog(x,y,'o', x, f(x))
#
# plt.show()

# plt.rc('font', **font)
# plt.figure(1)
# plt.plot(t, psi, color='LightSeaGreen', linewidth=2, label='$\psi$')
# # plt.plot(t, rho_frac, color='Plum', linewidth=2, label='$\\rho$/$\\rho_c$')
# plt.xlabel('Nondimentsional radius $\\xi$')
# plt.legend()


#
# # ----Find zi_o----#
# rho_c = np.linspace(1.5E-20, 2.9E-18, 2000)  # find the array of values of rho_c. Guesstimate!
# rho_c_len = len(rho_c)
# dpsi_len = len(dpsi)
# # Loop trough each value of rho_c and find a value of zi^2. Use that zi^2 to find
# # the total enclosed mass. If the mass desired is reached, save the
# # rho_c and zi_sq (and their indexes)  and exit the loop.
# # Convert the zi_sq into a physical radius.
# z_real = 0.0
# rho_c_real = 0.0
# dimensionless_mass = get_dmnless_mass(0.5 * 1.98E30, P_o, cs)  # do not use solar units
# for i in range(rho_c_len):
#     final_mass_computed = 0.48
#     for n in range(1, dpsi_len):
#         dimensionless_radius_squared = get_zi(dimensionless_mass, rho_o, rho_c[i], dpsi[n])
#         if dimensionless_radius_squared < 6.4:
#             mass_cloud = get_mass(rho_c[i], cs, dimensionless_radius_squared, dpsi[n])
#             if mass_cloud > 0.4999 and mass_cloud < 0.5001:
#                 if mass_cloud > final_mass_computed:
#                     final_mass_computed = mass_cloud
#                     total_mass = mass_cloud
#                     index_rho_c = i
#                     index_dpsi = n
#                     z_real = np.sqrt(dimensionless_radius_squared)
#                     rho_c_real = rho_c[i]
#
#
# def physical_r(z_real, rho_c_real, cs):
#     r_c = cs / np.sqrt(4. * np.pi * G * rho_c_real)
#     return z_real * r_c
#
#
# # ----Get physical values for plotting----#
# radius_au = physical_r(z_real, rho_c_real, cs)
# radius_cutoff = []
# rho_frac_cutoff = []
# rho_c_m = rho_c_real * 1E-4  # convert to kg/m^3
# for m in range(1, len(t)):
#     radius_list = physical_r(t, rho_c_real, cs)
#     if radius_list[m] < radius_au:
#         rho_frac_cutoff.append(rho_c_m * rho_frac[m] / m_H2 / 1E6)
#         radius_cutoff.append(radius_list[m] / pc)
#
# # ----Plot results----#
#
#
#
# plt.rc('font', **font)
# plt.figure(2)
# plt.plot(radius_cutoff, np.log(rho_frac_cutoff), color='b', linewidth=2, label='log($\\rho$/$\\rho_c$)')
# # plt.xlim(-0.3, 4.3)
# # plt.ylim(-1.2, 0)
# plt.xlabel('Radius (AU)')
# plt.legend(loc='lower left')
#
# rho_frac_cutoff = np.array(rho_frac_cutoff)
# radius_cutoff = np.array(radius_cutoff)
#
# plt.figure(3)
# plt.plot(radius_cutoff, rho_frac_cutoff, color='b', linewidth=2, label='$\\rho(r)$')
# plt.xlabel('Radius (AU)')
# plt.ylabel('$\\rho$(r) $/cm^3$')
# plt.legend()
#
# plt.rc('font', **font)
# plt.figure(4)
# plt.plot(np.log10(radius_cutoff), np.log10(rho_frac_cutoff / rho_c_m), color='b', linewidth=2,
#          label='log($\\rho$/$\\rho_c$)')
# plt.xlabel('Radius (AU)')
# plt.legend(loc='lower left')
plt.show()

