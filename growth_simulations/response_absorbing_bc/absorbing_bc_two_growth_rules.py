import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Constants
theta = 0.1        # given theta
x0 = 1.0            # fixed position
j_tilde = 1.0       # base production rate
lambda_tilde = 1.0  # base pattern length
D_tilde = 1.0       # base diffusion constant

# Define the equation to solve for L
def equation_L(L, theta, j, lambd, D):
    numerator = np.sinh(x0 / lambd) * np.sinh(L / (2 * lambd))
    denominator = np.sinh(L / lambd)
    return theta - (j * lambd / D) * numerator / denominator

# Range of scaling variable s
s_values = np.linspace(0.1, 2.0, 100)
L_A = []
L_B = []
L_C = []

for s in s_values:
    # Scenario A: j = s * j_tilde
    j_A = s * j_tilde
    lambda_A = lambda_tilde
    D_A = D_tilde

    # Scenario B: j and lambda scale with s
    j_B = s * j_tilde
    lambda_B = s * lambda_tilde
    D_B = D_tilde

    # Scenario C: j and lambda scale with s, D scales with s^2
    j_C = s * j_tilde
    lambda_C = s * lambda_tilde
    D_C = s**2 * D_tilde

    # Solve numerically for L in each case
    for j, lambd, D, L_list in [(j_A, lambda_A, D_A, L_A),
                                (j_B, lambda_B, D_B, L_B),
                                (j_C, lambda_C, D_C, L_C)]:
        try:
            sol = root_scalar(equation_L, args=(theta, j, lambd, D),
                              bracket=[0.001, 50], method='brentq')
            L_list.append(sol.root if sol.converged else np.nan)
        except:
            L_list.append(np.nan)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(s_values, L_A, label="A: source scaling", lw=2)
plt.plot(s_values, L_B, label="B: beta scaling", lw=2)
plt.plot(s_values, L_C, label="C: D scaling", lw=2)
plt.xlabel("Scaling variable $s$")
plt.ylabel("System size $L$")
plt.title("System size $L(s)$ for different scaling scenarios")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
