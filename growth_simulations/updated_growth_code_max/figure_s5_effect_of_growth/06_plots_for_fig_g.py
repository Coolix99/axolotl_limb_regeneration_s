import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from euler_scheme_1d_with_growth import two_morph, one_morph
from matplotlib_defaults import *

USE_TEX = True  # Set to True to enable LaTeX rendering
if USE_TEX:
    plt.rcParams['text.usetex'] = True


def parse_parameters(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.endswith(":"):
                continue
            try:
                key, value = line.split(": ", 1)
                if key == "lambda (sqrt(D/beta))":
                    params['lam'] = float(value)
                elif "." in value or "e" in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                continue
    return params

def list_visible_folders(path):
    return [
        item for item in os.listdir(path)
        if not item.startswith(".") and os.path.isdir(os.path.join(path, item))
    ]

def compute_thresholds(params, Lfinals):
    return np.array([
        two_morph.threshold_vs_Lfinal(Lf, params['x0'], params['lam'], params['alpha'], params['w'], params['beta'], params['D'])
        for Lf in Lfinals
    ])

def compute_thresholds_one(params, Lfinals):
    return np.array([
        one_morph.threshold_vs_Lfinal(Lf, params['x0'], params['lam'], params['alpha'], params['w'], params['beta'], params['D'])
        for Lf in Lfinals
    ])

def compute_Lfinals_to_reach(params, thresholds):
    return np.array([
        two_morph.Lfinal(th, params['x0'], params['lam'], params['alpha'], params['w'], params['beta'], params['D'])
        for th in thresholds
    ])

def compute_Lfinals_to_reach_one(params, thresholds):
    return np.array([
        one_morph.Lfinal(th, params['x0'], params['lam'], params['alpha'], params['w'], params['beta'], params['D'])
        for th in thresholds
    ])


def collect_simulated_data(folder, thresholds_to_test, g_to_test):
    collected_data = {g: {} for g in g_to_test}

    for ng in g_to_test:
        for nth in thresholds_to_test:
            for nfolder in os.listdir(folder):
                if str(ng) in nfolder and str(nth) in nfolder:
                    nfolder_path = os.path.join(folder, nfolder)

                    # Load arrays
                    time_to_save = np.load(os.path.join(nfolder_path, 'time_to_save.npy'))
                    x_gridt = np.load(os.path.join(nfolder_path, 'x_gridt.npy'))
                    growth_region = np.load(os.path.join(nfolder_path, 'grt.npy'))
                    sn = np.load(os.path.join(nfolder_path, 'st.npy'))
                    fn = np.load(os.path.join(nfolder_path, 'ft.npy'))

                    # Store in dictionary
                    collected_data[ng][nth] = {
                        "time": time_to_save,
                        "x_gridt": x_gridt,
                        "growth_region": growth_region,
                        "sn": sn,
                        "fn": fn
                    }
    return collected_data

def collect_simulated_data_one(folder, thresholds_to_test, g_to_test):
    collected_data = {g: {} for g in g_to_test}

    for ng in g_to_test:
        for nth in thresholds_to_test:
            for nfolder in os.listdir(folder):
                if str(ng) in nfolder and str(nth) in nfolder:
                    nfolder_path = os.path.join(folder, nfolder)

                    # Load arrays
                    time_to_save = np.load(os.path.join(nfolder_path, 'time_to_save.npy'))
                    x_gridt = np.load(os.path.join(nfolder_path, 'x_gridt.npy'))
                    growth_region = np.load(os.path.join(nfolder_path, 'grt.npy'))
                    sn = np.load(os.path.join(nfolder_path, 'st.npy'))

                    # Store in dictionary
                    collected_data[ng][nth] = {
                        "time": time_to_save,
                        "x_gridt": x_gridt,
                        "growth_region": growth_region,
                        "sn": sn,
                    }
    return collected_data

def plot_gx(gx, subdir):
    colors =  ['C0', 'C1', 'C2','C3']
    fig = plt.figure(figsize=(5.5 * 0.3937, 4.5 * 0.3937))
    ax = plt.subplot()
  

    for i, (target, (x, g)) in enumerate(gx.items()):
        L = np.max(x)
        color = colors[i % len(colors)]

        # plot g(x) as dashed
        ax.plot(x, g, linestyle='--', color=color, label=fr"$L/L_f={target}$", linewidth=2)

        # add vertical line at L, dotted, same color
        ax.axvline(L, linestyle=':', color=color, linewidth=1)


    ax.set_xlabel(r'Position $x$')
    ax.set_ylabel(r'Growth rate $g(x)$')
    
    ax.set_ylim(0,1)
    ax.set_xlim(left=0)
    #ax.legend(loc='lower right') 
    plt.subplots_adjust(left=0.18,bottom=0.27)
    plt.savefig(f'fig/gx_{subdir}.pdf', bbox_inches='tight')
    plt.savefig(f'fig/gx_{subdir}.eps', bbox_inches='tight')
    plt.savefig(f'fig/gx_{subdir}.svg', bbox_inches='tight')
    plt.show()



def two_morph_g(g,th,sn,fn,gr,L):
    return gr*g

def two_morph_stat_g(g,th,sn,fn,gr,L):
    return gr*g/L

def two_morph_mult_g(g,th,sn,fn,gr,L):
    return g * (sn-th) * (fn-th)  * gr 

def get_gx(data,g,th,L_rel_plots, g_function):
    print(data.keys())
    x_gridt = data['x_gridt']
    sn = data['sn']
    fn = data['fn']
    growth_region = data['growth_region']

    # Compute relative length trajectory
    L = np.sum(x_gridt, axis=1)
    L_rel = L / L[-1]

    results = {}
    for target in L_rel_plots:
        # Find index closest to target
        idx0 = np.argmin(np.abs(L_rel - target))
        # Compute g
        g_val = g_function(g,th,sn[idx0], fn[idx0], growth_region[idx0],np.sum(x_gridt[idx0]))
        # Compute spatial grid (cumulative)
        grid = np.cumsum(x_gridt[idx0])
        results[target] = (grid, g_val)
        print(idx0,grid.shape,g_val.shape)
    return results

def main():
    base_folder = 'growth_simulations/updated_growth_code_max/figure_s5_effect_of_growth/results_of_numerics/'
    all_subdirs = [('two_morph',0.5,(0.75,0.85,0.95),two_morph_g),
                    ('two_morph_static',0.9,(0.75,0.85,0.95),two_morph_stat_g), 
                    ('two_morph_gmultiplicative',250.0,(0.75,0.85,0.95),two_morph_mult_g)]
    all_Lfinal_sim = {}
    all_eff_growth = {}

    for (subdir, g, L_rel_plots, g_function) in all_subdirs:
        folder = os.path.join(base_folder, subdir, 'dynamics')
        param_file = os.path.join(folder, "parameters.txt")
        params = parse_parameters(param_file)

        required_keys = ['x0', 'alpha', 'w', 'beta', 'D', 'L0', 'lam']
        for key in required_keys:
            if key not in params:
                raise KeyError(f"Missing required parameter: {key}")

        folders = list_visible_folders(folder)
        thresholds_to_test = np.unique([float(f.split('_')[1]) for f in folders])
        g_to_test = np.unique([float(f.split('_')[-1]) for f in folders])


       
        collected_data = collect_simulated_data(folder, thresholds_to_test, g_to_test)
        gx=get_gx(collected_data[g][0.1379],g,0.1379,L_rel_plots, g_function)
        plot_gx(gx, subdir)
        
    
    
    subdir = 'one_morph'
    g = 0.5
    th=0.25
    L_rel_plots=(0.75,0.85,0.95)

    folder = os.path.join(base_folder, subdir, 'dynamics')
    param_file = os.path.join(folder, "parameters.txt")
    params = parse_parameters(param_file)

    required_keys = ['x0', 'alpha', 'w', 'beta', 'D', 'L0', 'lam']
    for key in required_keys:
        if key not in params:
            raise KeyError(f"Missing required parameter: {key}")

    folders = list_visible_folders(folder)
    thresholds_to_test = np.unique([float(f.split('_')[1]) for f in folders])
    g_to_test = np.unique([float(f.split('_')[-1]) for f in folders])


    
    collected_data = collect_simulated_data_one(folder, thresholds_to_test, g_to_test)

    data=collected_data[g][th]
    x_gridt = data['x_gridt']
    growth_region = data['growth_region']
    L = np.sum(x_gridt, axis=1)
    L_rel = L / L[-1]

    results = {}
    for target in L_rel_plots:
        # Find index closest to target
        idx0 = np.argmin(np.abs(L_rel - target))
        # Compute g
        g_val = growth_region[idx0]*g
        # Compute spatial grid (cumulative)
        grid = np.cumsum(x_gridt[idx0])
        results[target] = (grid, g_val)
        print(idx0,grid.shape,g_val.shape)
    
    plot_gx(results, subdir)

if __name__ == "__main__":
    main()