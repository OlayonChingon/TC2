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

from pytc2.general import Chebyshev_polynomials, s, w, print_subtitle #aca ya crea instancias s y w de sympy
import sympy as sp
from IPython.display import display, Math, Markdown

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

#calculo los coeficente Cn de cheby
chebn_expr = Chebyshev_polynomials(nn)
print(sp.expand(chebn_expr))
display(sp.expand(chebn_expr)) #visualizacion en latex

print("\n" * 3) #introduce tres espacios en blanco


#Conformo la funcion de transferencia al cuadrado en jw
Tcsq_jw = 1 / (1 + eps_sq * chebn_expr**2) #transferencia cheby square (al cuadrado)
#print(sp.expand(Tcsq_jw))
#display(sp.expand(Tcsq_jw))
display(Math( r' \left \| T_{c}{(\omega )} \right \|^{2} = ' + sp.latex(sp.expand(Tcsq_jw)) ))

print("\n" * 3)

#remplazo w=s/j
j = sp.I   # asignacion de j como la unidad imaginaria de sympy

Tcsq_s = Tcsq_jw.subs(w, s/j) # utiliza la función .subs() de Sympy para reemplazar la variable w en 
                              # Tcsq_jw por s/j
#print(sp.expand(Tcsq_s))
#display(sp.expand(Tcsq_s))
display(Math( r' \left \| T_{c}{(s)} \right \|^{2} = ' + sp.latex(sp.expand(Tcsq_s)) ))

#%% forma numérica. Tengo que trabajar con los coeficientes del polinomio

# Construir array de coeficientedes denominador del polinomio de  chebychev (Tcsq_den_s )
Cn3 = np.array([4., 0., -3., 0])  #construido a partir de los cn 4w³-3w
Cn3sq = np.polymul( Cn3, Cn3)
Tcsq_den_jw = np.polyadd( np.array([1.]), Cn3sq * eps_sq ) 

# convierto a s
Tcsq_den_s = Tcsq_den_jw * np.array([-1,-1,1,1,-1,-1,1]) # pasar a s es aplicar la ley de signos resultande 
                                                         #de las pot de j. en este caso 
                                                         #j⁶=-1, j⁵=-1, j⁴=1, j³=1, j²=-1, j⁰=1
                                                         #quedando [-1,-1, 1, 1, -1, -1, 1]

print(Tcsq_den_s) #polinomio cuyos coificientes son los de sp.expand(Tcsq_s) del analisis simbolico

roots_Tcsq_den_s = np.roots(Tcsq_den_s) #hallando las raices de este array de coeficientes obtengo los polos
print(roots_Tcsq_den_s)

# filtro T(s) reteniendo solo polos en el semi plano izquierdo
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
#%%Graficos
H1 = sig.TransferFunction( num_cheb, den_cheb )
analyze_sys(H1)


