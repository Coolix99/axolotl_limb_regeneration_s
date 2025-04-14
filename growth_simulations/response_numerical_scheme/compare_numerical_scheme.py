import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 1.0
g = 0.1
L0 = 10.0
T = 2.0
dt_fixed = 0.0005
dt_moving = 0.005
nt_fixed = int(T / dt_fixed)
nt_moving = int(T / dt_moving)
nx_fixed = 100
nx_moving = 100

# Track L(t)
Lt_fixed = L0 * (1 + g * dt_fixed) ** np.arange(nt_fixed)
Lt_moving = L0 * (1 + g * dt_moving) ** np.arange(nt_moving)

# Fixed grid (larger domain)
x_fixed_full = np.linspace(0, Lt_fixed[-1]*1.1, nx_fixed)
dx_fixed = x_fixed_full[1] - x_fixed_full[0]
v_fixed = g * x_fixed_full  # position-dependent velocity

# Initial conditions
def init_cond(x): return np.exp(-((x - L0/2)**2) / 0.5)

c_fixed = init_cond(x_fixed_full)
c_moving = init_cond(np.linspace(0, L0, nx_moving))

# --- Fixed Grid Scheme ---
for n in range(nt_fixed):
    L_current = Lt_fixed[n]
    iL = np.searchsorted(x_fixed_full, L_current)
    inside = x_fixed_full <= L_current

    lap = (np.roll(c_fixed, -1) - 2 * c_fixed + np.roll(c_fixed, 1)) / dx_fixed**2
    grad = (np.roll(c_fixed, -1) - np.roll(c_fixed, 1)) / (2 * dx_fixed)
    v_fixed = g * x_fixed_full  # Update v in case x domain shifts
    
    c_fixed += dt_fixed * (D * lap - v_fixed * grad - g * c_fixed)
    c_fixed[~inside] = 0
    
    # Reflecting BCs
    c_fixed[0] = c_fixed[1]
    c_fixed[iL] = c_fixed[iL - 1]

# --- Moving Bin Scheme ---
x_moving = np.linspace(0, L0, nx_moving)

for n in range(nt_moving):
    x_moving *= (1 + g * dt_moving)

    dxR = np.roll(x_moving, -1) - x_moving
    dxL = x_moving - np.roll(x_moving, 1)

    # Reflecting BCs
    c_moving[0] = c_moving[1]
    c_moving[-1] = c_moving[-2]

    cL = np.roll(c_moving, 1)
    cR = np.roll(c_moving, -1)
    lap = 2 / (dxR + dxL) * ((cR - c_moving)/dxR - (c_moving - cL)/dxL)

    c_moving += dt_moving * (D * lap - g * c_moving)

# --- Plot & Compare ---
plt.plot(x_fixed_full, c_fixed, label="Fixed grid (explicit v)")
plt.plot(x_moving, c_moving, '--', label="Moving bins (ours)")
plt.xlabel("x")
plt.ylabel("Concentration")
plt.legend()
plt.title("Convection-diffusion with growth (reflecting BCs)")
plt.show()

from numpy.linalg import norm
x_interp = np.linspace(0, Lt_fixed[-1], nx_fixed)
c_interp = np.interp(x_interp, x_moving, c_moving)
c_fixed_crop = np.interp(x_interp, x_fixed_full, c_fixed)

print(f"Relative error (L2): {norm(c_interp - c_fixed_crop)/norm(c_fixed_crop):.2e}")
print(f"Grid points used: fixed={len(x_fixed_full)}, moving={len(x_moving)}")
