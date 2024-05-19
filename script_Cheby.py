#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 07:14:48 2024

@author: mariano
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import analyze_sys, pretty_print_lti, tf2sos_analog, pretty_print_SOS

from pytc2.general import Chebyshev_polynomials, s, w, print_subtitle
import sympy as sp
from IPython.display import display

#%% datos del problema

alfa_max = 1 # dB
alfa_min = 12 # dB
ws = 2

#%% cuentas auxiliares

# epsilon cuadrado
eps_sq = 10**(alfa_max/10)-1
eps = np.sqrt(eps_sq)

for nn in range(2,5):
    
    alfa_min_c = 10*np.log10(1 + eps_sq * np.cosh(nn * np.arccosh(ws))**2 )
    # print( 'nn {:d} - alfa_min_cheby {:f}'.format(nn, alfa_min_c) )

    alfa_min_b = 10*np.log10(1 + eps_sq * ws**(2*nn))
    print( 'nn {:d} - alfa_min_butter {:f} - alfa_min_cheby {:f}'.format(nn, alfa_min_b, alfa_min_c) )

    # repasar décadas y octavas!!
    # 20*np.log10([1, 2, 4, 8, 16])
    # 20*np.log10([1, 10, 100, 1000])    

#%% elijo un orden luego de iterar ...

nn = 3

#%% forma simbólica la más natural viniendo desde el lápiz y papel

chebn_expr = Chebyshev_polynomials(nn)

print(sp.expand(chebn_expr))
display(sp.expand(chebn_expr))
# preguntar si pueden visualizar LaTex, sino usar print

#
Tcsq_jw = 1/(1 + eps_sq*chebn_expr**2 )
print(sp.expand(Tcsq_jw))
# display(sp.expand(Tcsq_jw))

j = sp.I

Tcsq_s = Tcsq_jw.subs(w, s/j)
print(sp.expand(Tcsq_s))
display(sp.expand(Tcsq_s))


#%% forma numérica. mucho menos clara

# asumo que hice la recursión en papel
Cn3 = np.array([4., 0., -3., 0])
Cn3sq = np.polymul( Cn3, Cn3)
Tcsq_den_jw = np.polyadd( np.array([1.]), Cn3sq * eps_sq ) 

# convierto a s
Tcsq_den_s = Tcsq_den_jw * np.array([-1,-1,1,1,-1,-1,1])
print(Tcsq_den_s)

roots_Tcsq_den_s = np.roots(Tcsq_den_s)
print(roots_Tcsq_den_s)

# filtro T(s) reteniendo solo polos en el SPI
roots_Tcsq_den_s = roots_Tcsq_den_s[np.real(roots_Tcsq_den_s) < 0]
print(roots_Tcsq_den_s)

z,p,k = sig.cheb1ap(nn, alfa_max)
num_cheb, den_cheb = sig.zpk2tf(z,p,k)

#%% análisis de lo obtenido

filter_names = []
all_sys = []

this_aprox = 'Cheby'
this_label = this_aprox + '_ord_' + str(nn) + '_rip_' + str(alfa_max) + '_att_' + str(alfa_min)

sos_cheb = tf2sos_analog(num_cheb, den_cheb)

filter_names.append(this_label)
all_sys.append(sig.TransferFunction(num_cheb, den_cheb))

analyze_sys( all_sys, filter_names )

print_subtitle(this_label)
# factorizamos en SOS's
pretty_print_SOS(sos_cheb, mode='omegayq')



