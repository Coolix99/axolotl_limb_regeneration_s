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

def c_star_open(x, alpha, beta, L, lambda_, w, x0):
    """Open boundaries (infinite domain): solution on R with exponential tails.
    Equivalent to using the infinite-domain Green's function.

    D C'' - beta C + s(x) = 0, s(x)=alpha on [x0, x0+w], 0 otherwise.
    lambda_ = sqrt(D/beta).
    """
    x = np.asarray(x, dtype=float)
    c = np.zeros_like(x, dtype=float)

    pref = alpha / (2.0 * beta)
    a = x0
    b = x0 + w

    m1 = (x < a)
    m2 = (x >= a) & (x <= b)
    m3 = (x > b)

    # Outside the source: single exponential
    c[m1] = pref * np.exp(x[m1] / lambda_) * (np.exp(-x0/lambda_)-np.exp(-(x0+w)/lambda_))
    c[m3] = pref * np.exp(-x[m3] / lambda_) * (np.exp((x0+w)/lambda_)-np.exp(x0/lambda_))

    # Inside the source: sum of two exponentials towards each edge
    if np.any(m2):
        xm = x[m2]
        c[m2] = pref * (np.exp(xm/lambda_) * (np.exp(-xm/lambda_)-np.exp(-(x0+w)/lambda_))
                        +np.exp(-xm/lambda_)* (np.exp(xm/lambda_)-np.exp(x0/lambda_)))
    return c


def c_star_absorbing(x, alpha, beta, L, lambda_, w, x0):
    """Absorbing (Dirichlet) boundaries: C(0)=C(L)=0 on a finite interval [0,L].

    Closed form from the Dirichlet Green's function on [0,L]:
      C(x) = (alpha/beta) / sinh(L/lambda) * (piecewise hyperbolic combo)
    """
    x = np.asarray(x, dtype=float)
    c = np.zeros_like(x, dtype=float)

    a = x0
    b = x0 + w
    den = np.sinh(L / lambda_)
    pref = (alpha / beta) / den

    m1 = (x >= 0) & (x < a)
    m2 = (x >= a) & (x <= b)
    m3 = (x > b) & (x <= L)

    # Region 1: [0, a)
    if np.any(m1):
        xm = x[m1]
        c[m1] = pref * (
            np.sinh(xm / lambda_) *
            (np.cosh((L - a) / lambda_) - np.cosh((L - b) / lambda_))
        )

    # Region 2: [a, b]
    if np.any(m2):
        xm = x[m2]
        c[m2] = pref * (
            np.sinh((L - xm) / lambda_) * (np.cosh(xm / lambda_) - np.cosh(a / lambda_))
            + np.sinh(xm / lambda_) * (np.cosh((L - xm) / lambda_) - np.cosh((L - b) / lambda_))
        )

    # Region 3: (b, L]
    if np.any(m3):
        xm = x[m3]
        c[m3] = pref * (
            np.sinh((L - xm) / lambda_) *
            (np.cosh(b / lambda_) - np.cosh(a / lambda_))
        )

    return c


def determine_theta_for_growth_arrest(lambda_, Lfinal, alpha, beta, w, x0):
    return c_star_open(Lfinal/2, alpha, beta, Lfinal, lambda_, w, x0)


def determine_Lfinal(theta, lambda_, alpha, beta, w, x0, c_star_func):
    def f(Lfinal):
        return c_star_func(Lfinal/2, alpha, beta, Lfinal, lambda_, w, x0) - theta

    # choose an initial guess (e.g. proportional to lambda_)
    L0 = 3.5

    # Newton's method root finding
    Lfinal = newton(f, L0)
    return Lfinal
     
def determine_t90(theta, lambda_, alpha, beta, w, x0, c_star_func,Lf,L0):
    Ltarget=Lf*0.9
    L=L0
    g=1.0
    t=0
    nt=100
    dL=(Ltarget-L0)/nt
    ns=200
    for i in range(nt):
        x= np.linspace(L/2, L, ns)
        c = c_star_func(x, alpha, beta, L, lambda_, w, x0)
        fraction_growth=np.sum(c>=theta)/ns
        dL_dt=L*g*fraction_growth
        dt= dL/dL_dt
        t+=dt
        L+=dL
    return t

USE_TEX = True  # enable LaTeX rendering
if USE_TEX:
    plt.rcParams['text.usetex'] = True





# ==============================
# Main script
# ==============================
def main():
    L0 = 1.8
    Lfinal = 2 * L0
    x0 = 0.4
    w= 0.4
    alpha = 1 
    beta = 1

    # L=1.8*1
    # lambda_ = 1.359
   

    # print(c_star_absorbing(x0-0.0001, alpha, beta, L, lambda_, w, x0),c_star_absorbing(x0+0.0001, alpha, beta, L, lambda_, w, x0))
    # print(c_star_absorbing(x0+w-0.0001, alpha, beta, L, lambda_, w, x0),c_star_absorbing(x0+w+0.0001, alpha, beta, L, lambda_, w, x0))

    # x= np.linspace(0, L, 100)
    # c_open = c_star_absorbing(x, alpha, beta, L, lambda_, w, x0)
    # plt.plot(x, c_open, label="open boundary", color='C0')
    # plt.show()

    #raise
    # sweep lambda between 0.1*Lfinal and Lfinal
    lambdas = np.linspace(0.01 * Lfinal, 0.25*Lfinal, 10)

    results_ref = []
    results_abs = []
    results_t90_open = []
    results_t90_ref = []
    results_t90_abs = []

    for lambda_ in lambdas:
        # determine theta for growth arrest with c_star_open
        theta = determine_theta_for_growth_arrest(lambda_, Lfinal, alpha, beta, w, x0)
        
        Lf_open = determine_Lfinal(theta, lambda_, alpha, beta, w, x0, c_star_open) #just check 
        print(f"lambda: {lambda_}, theta: {theta}, Lf_open: {Lf_open}")

        Lf_ref = determine_Lfinal(theta, lambda_, alpha, beta, w, x0, c_star_ref)
        Lf_abs = determine_Lfinal(theta, lambda_, alpha, beta, w, x0, c_star_absorbing)

        t90_open=determine_t90(theta, lambda_, alpha, beta, w, x0, c_star_open,Lfinal,L0)
        t90_ref=determine_t90(theta, lambda_, alpha, beta, w, x0, c_star_ref,Lf_ref,L0)
        t90_abs=determine_t90(theta, lambda_, alpha, beta, w, x0, c_star_absorbing,Lf_abs,L0)

        results_ref.append(Lf_ref)
        results_abs.append(Lf_abs)
        results_t90_open.append(t90_open)
        results_t90_ref.append(t90_ref)
        results_t90_abs.append(t90_abs)


    results_ref=np.array(results_ref)
    results_abs=np.array(results_abs)
    results_t90_open=np.array(results_t90_open)
    results_t90_ref=np.array(results_t90_ref)   
    results_t90_abs=np.array(results_t90_abs)


    print(lambdas / Lfinal, results_ref, results_abs)
    
    colors = ['C0', 'C1', 'C2', 'C3']
    fig = plt.figure(figsize=(9 * 0.3937, 4.5 * 0.3937))
    ax = plt.subplot()
    ax.plot(lambdas / L0, results_ref/ Lfinal, '--', label="reflecting", color=colors[0])
    ax.plot(lambdas / L0, results_abs/ Lfinal, '--', label="absorbing", color=colors[1])
    ax.axhline(1.0, linestyle="--", color="black", label="open")
    
    ax.set_xlabel(r"$\lambda/L_0$")
    ax.set_ylabel(r"$L_{\mathrm{final}}/2L_0$")

    ax.legend()

    #plt.subplots_adjust(left=0.18, bottom=0.27)

    os.makedirs("fig", exist_ok=True)
    plt.savefig(f'fig/L_bc.pdf', bbox_inches='tight')
    plt.savefig(f'fig/L_bc.eps', bbox_inches='tight')
    plt.savefig(f'fig/L_bc.svg', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(9 * 0.3937, 4.5 * 0.3937))
    ax = plt.subplot()
    ax.plot(lambdas / L0, results_t90_ref/ results_t90_open, '--', label="reflecting", color=colors[0])
    ax.plot(lambdas / L0, results_t90_abs/ results_t90_open, '--', label="absorbing", color=colors[1])
    ax.axhline(1.0, linestyle="--", color="black", label="open")
    
    ax.set_xlabel(r"$\lambda/L_0$")
    ax.set_ylabel(r"$t_{90}/t_{90,\mathrm{open}}$")

    ax.legend()

    #plt.subplots_adjust(left=0.18, bottom=0.27)

    os.makedirs("fig", exist_ok=True)
    plt.savefig(f'fig/t_bc.pdf', bbox_inches='tight')
    plt.savefig(f'fig/t_bc.eps', bbox_inches='tight')
    plt.savefig(f'fig/t_bc.svg', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
