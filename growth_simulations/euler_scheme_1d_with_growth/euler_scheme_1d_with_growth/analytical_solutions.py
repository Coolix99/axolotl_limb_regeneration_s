import numpy as np

def hill(A, K, n):
    return A**n/(A**n + K**n)

def nheaviside(x, accuracy):
    return np.where(x > -accuracy, 1, 0)

def nrect(x, x0, w, accuracy):
    return nheaviside(w + x0 - x, accuracy) * nheaviside(x - x0, accuracy)

def steady_solution(x, x0, lam, alpha, w, beta, Diff, L):

    # left  from the source
    part1 = 2 * np.sinh(w / (2 * lam)) * np.cosh((2 * L - 2 * x0 - w) / (2 * lam)) * np.cosh(x / lam)
    part1 = part1 * (x <= x0)

    # inside the source
    part2 = np.sinh(L / lam) - (np.sinh((L - w - x0) / lam) * np.cosh(x / lam) + np.sinh(x0 / lam) * np.cosh((L - x) / lam))
    part2 = part2 * (x > x0) * (x <= x0 + w)

    # right from the source
    part3 = 2 * np.sinh(w / (2 * lam)) * np.cosh((w + 2 * x0) / (2 * lam)) * np.cosh((L - x) / lam)
    part3 = part3 * (x > x0 + w)

    result = (alpha / (beta * np.sinh(L / lam))) * (part1 + part2 + part3) * (x0 + w <= L)
    
    return result

def steady_absorbing(x, x0, lam, alpha, w, beta, Diff, L):
    A = (alpha / beta) * (
        np.cosh(x0 / lam) / np.tanh(L / lam)
        - np.cosh((-L + w + x0) / lam) / np.sinh(L / lam)
    )

    B = - (alpha / beta) * np.cosh(x0 / lam)

    C1 = (alpha / beta) * (
        np.cosh((L - x0) / lam) - np.cosh((-L + w + x0) / lam)
    ) / np.sinh(L / lam)

    C2 = - (alpha / beta) * (
        np.cosh(x0 / lam) - np.cosh((w + x0) / lam)
    ) / np.sinh(L / lam)

    # left from the source
    part1 = np.sinh(x / lam) * C1
    part1 = part1 * (x <= x0)

    # inside the source
    part2 = A * np.sinh(x/ lam) + B * np.cosh(x / lam) + alpha/beta
    part2 = part2 * (x > x0) * (x <= x0 + w)

    # right from the source
    part3 = np.sinh((L - x) / lam) * C2
    part3 = part3 * (x > x0 + w)

    result =  (part1 + part2 + part3) * (x0 + w <= L)
    
    return result

def steady_open(x, x0, lam, alpha, w, beta, Diff, L):
    A = -alpha * np.exp(-(w + x0) / lam) / (2 * beta)

    B = -alpha * np.exp(x0 / lam) / (2 * beta)

    C1 = alpha * (np.exp(w / lam) - 1) * np.exp(-(w + x0) / lam) / (2 * beta)

    C2 = alpha * (-np.exp(x0 / lam) + np.exp((w + x0) / lam)) * np.exp(-L / lam) / (2 * beta)

    # left from the source
    part1 = np.exp(x / lam) * C1
    part1 = part1 * (x <= x0)

    # inside the source
    part2 = A * np.exp(x/ lam) + B * np.exp(-x / lam) + alpha/beta
    part2 = part2 * (x > x0) * (x <= x0 + w)

    # right from the source
    part3 = np.exp((L - x) / lam) * C2
    part3 = part3 * (x > x0 + w)

    result =  (part1 + part2 + part3) * (x0 + w <= L)
    
    return result

def main():
    import matplotlib.pyplot as plt
    L = 4.5
    x0 = 0.2
    w = 0.5
    beta = 1.0
    alpha = 1.0
    Diff = 1.0
    lam = np.sqrt(Diff / beta)

    x = np.linspace(0, L, 1000)

    y_reflecting = steady_solution(x, x0, lam, alpha, w, beta, Diff, L)
    y_absorbing = steady_absorbing(x, x0, lam, alpha, w, beta, Diff, L)
    y_open = steady_open(x, x0, lam, alpha, w, beta, Diff, L)

    plt.plot(x, y_reflecting, label="Reflecting BC")
    plt.plot(x, y_absorbing, label="Absorbing BC")
    plt.plot(x, y_open, label="Open BC")

    plt.title("Steady-State Solutions with Different Boundary Conditions")
    plt.xlabel("x")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
