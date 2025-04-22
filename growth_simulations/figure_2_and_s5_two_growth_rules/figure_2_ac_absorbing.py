import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# function for numerical scheme in custom package 
from euler_scheme_1d_with_growth.functions_for_numerics import *

# GROWTH RULE
import euler_scheme_1d_with_growth.one_morph as one_morph

default_figsize = plt.rcParams['figure.figsize']

# for better plots
plt.rcParams['text.usetex'] = True 

# assume to be 1
D     = 1
beta  = 1
alpha = 1
lam   = math.sqrt(D/beta) 

# supported by experiments
x0    = 0.4*lam 
w     = 0.4*lam

# minimal anitial size: minimal possible x0 + w, minimal plausible 2 * (x0 + w/2), minimal prefered (2 * x0 + 2 * w)
L0    = 1.8 * lam

thresholds_to_test = np.linspace(0.04, 0.12, 5)

# for plotting
Lfinals = np.linspace(0, 4 , 100)


small_number = 1e-6

# space x
ndx  = 500
dx   = L0/ndx

# time t
tmax         = 10
dt           = 0.5*1e-5
t            = np.arange(0, tmax+small_number, dt)
ndt          = len(t)

# save only ndt_to_save time points
ndt_to_save = 1000

print(D * dt / dx**2)
print(f"D={D}, dt={dt}, dx={dx}")
print(f"alpha={alpha}, beta={beta}, lam={lam}")
print(f"x0={x0}, w={w}")

thresholds = np.array( [
    one_morph.threshold_vs_Lfinal_absorbing(Lfinal, x0, lam, alpha, w, beta, D) 
    for Lfinal in Lfinals] )
print( thresholds )

print( thresholds_to_test )
Lfinals_to_reach   = np.array([
    one_morph.Lfinal_absorbing(threshold, x0, lam, alpha, w, beta, D) for threshold in thresholds_to_test
])

plt.plot( thresholds*beta/alpha/w, Lfinals/lam )
plt.plot( thresholds_to_test*beta/alpha/w, Lfinals_to_reach/lam, '.' )
plt.hlines( L0, 0, 2, linestyles='dashed')
plt.hlines(x0+w, 0, 2)
plt.xlabel(r'Threshold, $\Theta \beta / \alpha w$')
plt.ylabel(r'Final system size, $L^*/\lambda$')
# plt.xlim( 0, 1.2 )
# plt.ylim( 0, 3 )
plt.show()