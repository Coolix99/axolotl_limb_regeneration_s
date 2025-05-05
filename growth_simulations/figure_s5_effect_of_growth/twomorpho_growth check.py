import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
import math
import shutil
import os


from euler_scheme_1d_with_growth import two_morph as two_morph

def loop():
    # Parameters
    D     = 1
    beta  = 1
    alpha = 1
    lam   = math.sqrt(D / beta)

    x0    = 0.5 * lam
    w     = 0.5 * lam
    L0    = 1.8 * (x0 + w)
    threshold_growth = 0.2

    small_number = 1e-6
    L_final_g0=two_morph.Lfinal(threshold_growth, x0, lam, alpha, w, beta, D)
    all_L_final_rel=[]
    for ndx in [100,200,300,500,600,700,900,1100,1500,2000,2500, 3000, 4000]:

   
        dx   = L0 / ndx

        tmax = 5
        dt   = dx*dx/2

        t    = np.arange(0, tmax + small_number, dt)
        ndt  = len(t)

        ndt_to_save = min(3000,ndt)
    
        g =  5*beta

    
    
        print(f'th_{threshold_growth}_g_{g}')
        simulation = two_morph.two_morph_euler(
            D, beta, alpha, L0, x0, w, g, threshold_growth,
            ndx, dx, tmax, dt, t, ndt, ndt_to_save
        )
        print('expected',)
        simulation.run_simulation()
        print(ndx,simulation.Lt[-1])
        print(ndx,simulation.Lt[-1]/L_final_g0)
        all_L_final_rel.append(simulation.Lt[-1]/L_final_g0)

    plt.plot([100,200,300,500,600,700],all_L_final_rel)
    plt.xlabel('ndx')
    plt.ylabel('L_final/L_final_expected')
    plt.title(f'growth = {g}, threshold = {threshold_growth}')
    plt.show()

# th_0.2_g_1: 
# 100 0.9717195257888565
# 200 0.9731684383089295
# 300 0.9799439138766213
# 500 0.9826863634550059
# 600 0.9878290949334267
# 700 0.9892968417745257

# th_0.2_g_5
# 100 0.9298856842594323
# 200 0.943491566749725
# 300 0.9489337814018177
# 500 0.9565842488884835
# 600 0.9591431675506469
# 700 0.9610107280906546
# 900 0.963495443870222
# 1100 0.9658227122346277
# 1500 0.9676129440917681
# 2000 0.9707316818857342
# 2500 0.9726535878055533
# 3000 0.9741248875401709
# 4000 0.9762800556964006


#loop()
from scipy.stats import linregress
def plot():
    # Data
    ndx_1 = np.array([100, 200, 300, 500, 600, 700])
    y_g1 = np.array([0.9717195257888565, 0.9731684383089295, 0.9799439138766213, 
            0.9826863634550059, 0.9878290949334267, 0.9892968417745257])
    
    ndx_2 = np.array([100, 200, 300, 500, 600, 700, 900, 1100, 1500, 2000, 2500, 3000, 4000])
    y_g5 = np.array([0.9298856842594323, 0.943491566749725, 0.9489337814018177, 
            0.9565842488884835, 0.9591431675506469, 0.9610107280906546,
            0.963495443870222, 0.9658227122346277, 0.9676129440917681,
            0.9707316818857342, 0.9726535878055533, 0.9741248875401709,
            0.9762800556964006])

    # Compute 1 - y
    one_minus_y_g1 = [1 - val for val in y_g1]
    one_minus_y_g5 = [1 - val for val in y_g5]

    # Regression for g_1
    slope_g1, intercept_g1, *_ = linregress(ndx_1, one_minus_y_g1)
    x_zero_g1 = -intercept_g1 / slope_g1
    y_fit_g1 = slope_g1 * ndx_1 + intercept_g1 

    # Regression for g_5
    slope_g5, intercept_g5, *_ = linregress(ndx_2, one_minus_y_g5)
    x_zero_g5 = -intercept_g5 / slope_g5
    y_fit_g5 = slope_g5 * ndx_2 + intercept_g5

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Top plot
    ax1.plot(ndx_1, y_g1, marker='o', label='th_0.2_g_1')
    ax1.plot(ndx_2, y_g5, marker='s', label='th_0.2_g_5')
    ax1.set_ylabel('L_final / L_final_expected')
    ax1.set_title('growth = g, threshold = 0.2')
    ax1.legend()
    ax1.grid(True)

    # Bottom plot
    ax2.plot(ndx_1, one_minus_y_g1, marker='o', label='1 - th_0.2_g_1')
    ax2.plot(ndx_2, one_minus_y_g5, marker='s', label='1 - th_0.2_g_5')
    ax2.plot(ndx_1, y_fit_g1, linestyle='--', label=f'LR g_1 (→ 0 at {x_zero_g1:.1f})')
    ax2.plot(ndx_2, y_fit_g5, linestyle='--', label=f'LR g_5 (→ 0 at {x_zero_g5:.1f})')
    ax2.axvline(x_zero_g1, linestyle=':', color='gray')
    ax2.axvline(x_zero_g5, linestyle=':', color='gray')
    ax2.text(x_zero_g1, 0, f'{x_zero_g1:.1f}', va='bottom', ha='right', fontsize=8, rotation=90)
    ax2.text(x_zero_g5, 0, f'{x_zero_g5:.1f}', va='bottom', ha='right', fontsize=8, rotation=90)
    ax2.set_xlabel('ndx')
    ax2.set_ylabel('1 - L_final / L_final_expected')
    ax2.legend()
    ax2.set_ylim(0, max(max(one_minus_y_g1), max(one_minus_y_g5)) * 1.1)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Inverse x and log-log regression
    inv_ndx_1 = 1 / ndx_1
    inv_ndx_2 = 1 / ndx_2

    logx_1 = np.log(inv_ndx_1)
    logy_1 = np.log(one_minus_y_g1)

    logx_2 = np.log(inv_ndx_2)
    logy_2 = np.log(one_minus_y_g5)

    # Power law (log-log linear fit)
    slope_p1, intercept_p1, *_ = linregress(logx_1, logy_1)
    slope_p2, intercept_p2, *_ = linregress(logx_2, logy_2)

    # Reconstruct fit lines in original scale
    y_fit_p1 = np.exp(intercept_p1) * inv_ndx_1**slope_p1
    y_fit_p2 = np.exp(intercept_p2) * inv_ndx_2**slope_p2

    # Plot
    plt.figure(figsize=(6, 5))
    plt.loglog(inv_ndx_1, one_minus_y_g1, 'o', label='th_0.2_g_1')
    plt.loglog(inv_ndx_2, one_minus_y_g5, 's', label='th_0.2_g_5')
    plt.loglog(inv_ndx_1, y_fit_p1, '--', label=f'Power fit g_1 (exp={slope_p1:.2f})')
    plt.loglog(inv_ndx_2, y_fit_p2, '--', label=f'Power fit g_5 (exp={slope_p2:.2f})')
    plt.xlabel('1 / ndx')
    plt.ylabel('1 - L_final / L_final_expected')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.title('Log-Log Plot with Power Law Fit')
    plt.tight_layout()
    plt.show()

plot()