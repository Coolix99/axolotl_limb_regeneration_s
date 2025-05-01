import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
import math
import shutil
import os

from euler_scheme_1d_with_growth import two_morph_gstatic as two_morph


def loop():
    # Parameters
    D     = 1
    beta  = 1
    alpha = 1
    lam   = math.sqrt(D / beta)

    x0    = 0.5 * lam
    w     = 0.5 * lam
    L0    = 1.8 * (x0 + w)
    thresholds_to_test = np.array([0.18,0.2,0.22])

    small_number = 1e-6

    ndx  = 500
    dx   = L0 / ndx

    tmax = 5
    dt   = 0.5e-5
    t    = np.arange(0, tmax + small_number, dt)
    ndt  = len(t)

    ndt_to_save = 1000
    g_to_test = np.array([0.5, 1, 2, 3.5, 5]) * beta * L0

    output_folder = 'growth_simulations/figure_s5_effect_of_growth/results_of_numerics/two_morph/dynamics/'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    file_path = os.path.join(output_folder, "parameters.txt")
    with open(file_path, 'w') as file:
        file.write("Model Parameters:\n")
        file.write(f"D: {D}\n")
        file.write(f"beta: {beta}\n")
        file.write(f"alpha: {alpha}\n")
        file.write(f"lambda (sqrt(D/beta)): {lam}\n")
        file.write(f"x0: {x0}\n")
        file.write(f"w: {w}\n")
        file.write(f"L0: {L0}\n")
        file.write(f"small_number: {small_number}\n\n")

        file.write("Space Parameters:\n")
        file.write(f"ndx: {ndx}\n")
        file.write(f"dx: {dx}\n\n")

        file.write("Time Parameters:\n")
        file.write(f"tmax: {tmax}\n")
        file.write(f"dt: {dt}\n")
        file.write(f"t (array length): {len(t)}\n")
        file.write(f"ndt: {ndt}\n")
        file.write(f"ndt_to_save: {ndt_to_save}\n")

    for g in g_to_test:
        for threshold_growth in thresholds_to_test:
            print(f'th_{threshold_growth}_g_{g}')
            simulation = two_morph.two_morph_euler(
                D, beta, alpha, L0, x0, w, g, threshold_growth,
                ndx, dx, tmax, dt, t, ndt, ndt_to_save
            )
            simulation.run_simulation()
            simulation.save_results(folder_base=output_folder)

if __name__ == "__main__":
    loop()
