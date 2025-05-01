import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import euler_scheme_1d_with_growth.two_morph as two_morph
from matplotlib_defaults import *


def read_simulation_parameters(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.endswith(":"):
                key, value = line.split(": ")
                if key == "lambda (sqrt(D/beta))":
                    params['lam'] = float(value)
                elif key == "t (array length)":
                    continue
                elif "." in value or "e" in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
    return params


def list_visible_folders(path):
    return [
        item for item in os.listdir(path)
        if not item.startswith(".") and os.path.isdir(os.path.join(path, item))
    ]


def extract_thresholds_and_g(folder):
    folders = list_visible_folders(folder)
    thresholds = np.unique([float(name.split('_')[1]) for name in folders])
    g_values = np.unique([float(name.split('_')[-1]) for name in folders])
    return thresholds, g_values


def simulate_dynamics(folder, thresholds_to_test, g_to_test):
    Lfinal_simulated = {
        g: dict(zip(thresholds_to_test, np.zeros(len(thresholds_to_test))))
        for g in g_to_test
    }
    
    nrows = len(thresholds_to_test)
    fig, ax = plt.subplots(nrows, 1, figsize=(default_figsize["large"][0]/3, default_figsize["large"][1]), dpi=200, sharex=True, sharey=True)

    for i, ng in enumerate(g_to_test):
        for j, nth in enumerate(thresholds_to_test):
            for nfolder in os.listdir(folder):
                if (str(ng) in nfolder) and (str(nth) in nfolder):
                    nfolder_path = os.path.join(folder, nfolder)
                    time_to_save = np.load(os.path.join(nfolder_path, 'time_to_save.npy'))
                    Lt = np.load(os.path.join(nfolder_path, 'Lt.npy'))
                    Lfinal_simulated[ng][nth] = Lt[-1]

            ax[j].plot(time_to_save, Lt, label=f'gL? = {ng}')
            ax[j].set_title(f'th = {np.round(nth, 2)}', fontsize=6)
            ax[j].set_xlabel('time, t')
            ax[j].set_ylabel('size, L(t)')
            ax[j].legend()
            ax[j].set_ylim(0, 3.5)

    plt.tight_layout()
    plt.show()

    return Lfinal_simulated


def plot_Lfinal_comparison(thresholds, thresholds_to_test, Lfinals, Lfinals_to_reach, Lfinal_simulated, beta, alpha, w, lam, x0):
    fig = plt.figure(figsize=default_figsize['small'])
    ax = plt.subplot()

    ax.plot(thresholds * beta / alpha / w, Lfinals / lam)
    ax.plot(thresholds_to_test * beta / alpha / w, Lfinals_to_reach / lam, '.')

    for ng in Lfinal_simulated:
        ax.plot(
            np.array(list(Lfinal_simulated[ng].keys())) * beta / alpha / w,
            np.array([Lfinal_simulated[ng][nth] for nth in Lfinal_simulated[ng].keys()]) / lam,
            'x',
            markerfacecolor='none',
            label=f'g = {ng}'
        )

    ax.fill_between(np.linspace(0, 2), 0, x0 + w, color='gray', alpha=0.5)

    ax.set_xlabel(r'Threshold, $\Theta \beta / \alpha w$')
    ax.set_ylabel(r'Final system size, $L^*/\lambda$')
    plt.xlim(0, 1.2)
    plt.ylim(0, 4)
    ax.legend(loc='upper right')

    plt.savefig('fig/two_morph_gstatic_Lfinal_dynamics.pdf', bbox_inches='tight')
    plt.show()


def plot_ratio_simulated_to_expected(Lfinal_simulated, thresholds_to_test, g_to_test, Lfinals_to_reach, colormap='viridis'):
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    fig, ax = plt.subplots()
    norm = plt.Normalize(min(thresholds_to_test), max(thresholds_to_test))
    for i, th in enumerate(thresholds_to_test):
        ratios = [Lfinal_simulated[g][th] / Lfinals_to_reach[i] for g in g_to_test]
        ax.plot(g_to_test, ratios, 'o-', color=cmap(norm(th)), label=f'Th={th}')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('g')
    ax.set_ylabel(r'Simulated $L^* / L_{\mathrm{expected}}$')
    ax.set_title('Ratio of Simulated to Expected Final Size')
    ax.legend(title="Threshold")
    plt.tight_layout()
    plt.show()

def main():
    plt.rcParams['text.usetex'] = True

    folder = 'growth_simulations/figure_s5_effect_of_growth/results_of_numerics/two_morph_gstatic/dynamics/'
    folder = 'growth_simulations/figure_s5_effect_of_growth/results_of_numerics/two_morph_gmultiplicative/dynamics/'
    #folder = 'growth_simulations/figure_s5_effect_of_growth/results_of_numerics/two_morph/dynamics/'
    file_path = os.path.join(folder, "parameters.txt")

    params = read_simulation_parameters(file_path)
    lam = params.pop('lam')
    x0 = params['x0']
    L0 = params['L0']
    alpha = params['alpha']
    beta = params['beta']
    w = params['w']
    D = params['D']

    thresholds_to_test, g_to_test = extract_thresholds_and_g(folder)
    Lfinals = np.linspace(0, 4 * L0, 100)

    thresholds = np.array([
        two_morph.threshold_vs_Lfinal(Lfinal, x0, lam, alpha, w, beta, D)
        for Lfinal in Lfinals
    ])

    Lfinals_to_reach = np.array([
        two_morph.Lfinal(threshold, x0, lam, alpha, w, beta, D)
        for threshold in thresholds_to_test
    ])

    Lfinal_simulated = simulate_dynamics(folder, thresholds_to_test, g_to_test)
    plot_Lfinal_comparison(thresholds, thresholds_to_test, Lfinals, Lfinals_to_reach, Lfinal_simulated, beta, alpha, w, lam, x0)
    plot_ratio_simulated_to_expected(Lfinal_simulated, thresholds_to_test, g_to_test, Lfinals_to_reach)

if __name__ == "__main__":
    main()

