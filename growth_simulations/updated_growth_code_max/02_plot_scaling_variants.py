import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import newton

def c_star_ref(x, alpha, beta, L, lambda_, w, x0):
    """Reference implementation of c*(x) with reflecting boundaries."""
    prefactor = alpha / (beta * np.sinh(L / lambda_))
    
    x = np.asarray(x)
    c = np.zeros_like(x, dtype=float)

    mask1 = (x >= 0) & (x < x0)
    mask2 = (x >= x0) & (x < x0 + w)
    mask3 = (x >= x0 + w) & (x <= L)

    c[mask1] = (2 * np.sinh(w / (2*lambda_)) *
                np.cosh((2*L - w - 2*x0) / (2*lambda_)) *
                np.cosh(x[mask1] / lambda_))

    c[mask2] = (np.sinh(L/lambda_)
                - np.sinh((L - w - x0) / lambda_) * np.cosh(x[mask2] / lambda_)
                - np.sinh(x0 / lambda_) * np.cosh((L - x[mask2]) / lambda_))

    c[mask3] = (2 * np.sinh(w / (2*lambda_)) *
                np.cosh((w + 2*x0) / (2*lambda_)) *
                np.cosh((L - x[mask3]) / lambda_))

    return prefactor * c

def inv_c(c,alpha, beta, L, lambda_, w, x0):
    A=c*beta*np.sinh(L/lambda_)/alpha/2/np.sinh(w/(2*lambda_))/np.cosh((w+2*x0)/(2*lambda_))
    return 1-lambda_ * np.arccosh(A)/L


def determine_Lfinal(theta, lambda_, alpha, beta, w, x0, c_star_func):
    def f(Lfinal):
        return c_star_func(Lfinal/2, alpha, beta, Lfinal, lambda_, w, x0) - theta

    # choose an initial guess (e.g. proportional to lambda_)
    L0 = 10

    # Newton's method root finding
    Lfinal = newton(f, L0)
    return Lfinal
     


USE_TEX = True  # enable LaTeX rendering
if USE_TEX:
    plt.rcParams['text.usetex'] = True





# ==============================
# Main script
# ==============================
def main():
    L0s = np.linspace(1.5, 4.0, 10)  # L0 values to test
   
    x0_rel = 0.2
    w_rel = 0.2
    alpha = 1 
    beta = 1
    lambda_rel = 0.3

    theta=0.15
    
    results_v1=[]
    results_v2=[]
    results_v3=[]
    results_v4=[]

    for L0 in L0s:
        lambda_static = lambda_rel * L0
        w_static = w_rel * L0
        x0_static = x0_rel * L0

        #V1
        def f1(L):
            wshh=w_rel * L
            x0shh=x0_rel * L
            lambda_shh=lambda_rel * L 

            wfgf=w_static
            x0fgf=x0_static
            lambda_fgf=lambda_static
            # print(c_star_ref(L*0.5, alpha, beta, L, lambda_shh, wshh, x0shh))
            # print(inv_c(theta,alpha, beta, L, lambda_shh, wshh, x0shh))
            
            return inv_c(theta,alpha, beta, L, lambda_shh, wshh, x0shh)-(1-inv_c(theta,alpha, beta, L, lambda_fgf, wfgf, x0fgf))
        Lfinal=np.nan
        best=np.inf
        for L in np.linspace(1.0*L0, 4.0*L0, 1000):
            if np.abs(f1(L)) < best:
                best = np.abs(f1(L))
                Lfinal = L
          
        print(L0,Lfinal,best)
        results_v1.append(Lfinal)

        #V2
        def f2(L):
            wshh=w_rel * L
            x0shh=x0_rel * L
            lambda_shh=lambda_static

            wfgf=w_static
            x0fgf=x0_static
            lambda_fgf=lambda_rel * L 
            
            return inv_c(theta,alpha, beta, L, lambda_shh, wshh, x0shh)-(1-inv_c(theta,alpha, beta, L, lambda_fgf, wfgf, x0fgf))
        Lfinal=np.nan
        best=np.inf
        for L in np.linspace(1.0*L0, 4.0*L0, 1000):
            if np.abs(f2(L)) < best:
                best = np.abs(f2(L))
                Lfinal = L
          
        print(L0,Lfinal,best)
        results_v2.append(Lfinal)
    

        #V3
        def f3(L):
            wshh=w_rel * L
            x0shh=x0_rel * L
            lambda_shh=lambda_static

            wfgf=w_static
            x0fgf=x0_rel * L
            lambda_fgf=lambda_rel * L 
            
            return inv_c(theta,alpha, beta, L, lambda_shh, wshh, x0shh)-(1-inv_c(theta,alpha, beta, L, lambda_fgf, wfgf, x0fgf))
        Lfinal=np.nan
        best=np.inf
        for L in np.linspace(1.0*L0, 4.0*L0, 1000):
            if np.abs(f3(L)) < best:
                best = np.abs(f3(L))
                Lfinal = L
          
        print(L0,Lfinal,best)
        results_v3.append(Lfinal)

        #V4
        def f4(L):
            wshh=w_rel * L
            x0shh=x0_rel * L
            lambda_shh=lambda_rel * L 

            wfgf=w_static
            x0fgf=x0_rel * L
            lambda_fgf=lambda_rel * L 
            
            return inv_c(theta,alpha, beta, L, lambda_shh, wshh, x0shh)-(1-inv_c(theta,alpha, beta, L, lambda_fgf, wfgf, x0fgf))
        Lfinal=np.nan
        best=np.inf
        for L in np.linspace(1.0*L0, 4.0*L0, 1000):
            if np.abs(f4(L)) < best:
                best = np.abs(f4(L))
                Lfinal = L
          
        print(L0,Lfinal,best)
        results_v4.append(Lfinal)

    results_v1=np.array(results_v1)
    results_v2=np.array(results_v2)
    results_v3=np.array(results_v3)
    results_v4=np.array(results_v4)


    colors = ['C0', 'C1', 'C2', 'C3']
    fig = plt.figure(figsize=(5.5 * 0.3937, 5.5 * 0.3937))
    ax = plt.subplot()
    ax.plot(L0s, results_v1, '--', label="Variante 1", color=colors[0])
    ax.plot(L0s, results_v2, '--', label="Variante 2", color=colors[1])
    ax.plot(L0s, results_v3, '--', label="Variante 3", color=colors[2])
    ax.plot(L0s, results_v4, '--', label="Variante 4", color=colors[3])

    x = np.linspace(0, 4.0, 200)  # cover x-range of plot
    ax.fill_between(x, 0, x, color="grey", alpha=0.3)
    
    ax.set_xlabel(r"Initial system size")
    ax.set_ylabel(r"Final size $L^*(x)$")

    ax.legend()
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 6.0)
    #plt.subplots_adjust(left=0.18, bottom=0.27)

    os.makedirs("fig", exist_ok=True)
    plt.savefig(f'fig/scaling_var.pdf', bbox_inches='tight')
    plt.savefig(f'fig/scaling_var.eps', bbox_inches='tight')
    plt.savefig(f'fig/scaling_var.svg', bbox_inches='tight')
    plt.show()


    colors = ['C0', 'C1', 'C2', 'C3']
    fig = plt.figure(figsize=(5.5 * 0.3937, 5.5 * 0.3937))
    ax = plt.subplot()
    ax.plot(L0s, np.gradient(results_v1, L0s), '--', label="Variante 1", color=colors[0])
    ax.plot(L0s, np.gradient(results_v2, L0s), '--', label="Variante 2", color=colors[1])
    ax.plot(L0s, np.gradient(results_v3, L0s), '--', label="Variante 3", color=colors[2])
    ax.plot(L0s, np.gradient(results_v4, L0s), '--', label="Variante 4", color=colors[3])


    ax.set_xlabel(r"Initial system size")
    ax.set_ylabel(r"Derivative of final size $L^*(x)$")

    ax.legend()
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 4.0)
    
    #plt.subplots_adjust(left=0.18, bottom=0.27)

    os.makedirs("fig", exist_ok=True)
    plt.savefig(f'fig/scaling_var_der.pdf', bbox_inches='tight')
    plt.savefig(f'fig/scaling_var_der.eps', bbox_inches='tight')
    plt.savefig(f'fig/scaling_var_der.svg', bbox_inches='tight')
    plt.show()

    

if __name__ == "__main__":
    main()
