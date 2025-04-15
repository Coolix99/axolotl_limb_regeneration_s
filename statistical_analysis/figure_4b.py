import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, t

def get_data():
    return pd.read_csv('statistical_analysis/results_from_image_analysis_figure_4_blastema_width_from_3d.csv')

def compute_eta(X, Y, Z):
    slope_XZ, intercept_XZ, *_ = linregress(Z, X)
    eps_X = X - (slope_XZ * Z + intercept_XZ)

    slope_YZ, intercept_YZ, *_ = linregress(Z, Y)
    eps_Y = Y - (slope_YZ * Z + intercept_YZ)

    slope_eps, intercept_eps, *_ = linregress(eps_Y, eps_X)
    pred_eps_X = slope_eps * eps_Y + intercept_eps
    eps_final = eps_X - pred_eps_X

    eta2 = 1 - np.sum(eps_final**2) / np.sum(eps_X**2)

    r = (np.corrcoef(X, Y)[0, 1] - np.corrcoef(X, Z)[0, 1] * np.corrcoef(Y, Z)[0, 1]) / \
        np.sqrt((1 - np.corrcoef(X, Z)[0, 1]**2) * (1 - np.corrcoef(Y, Z)[0, 1]**2))
    print(f"r: {r}")
    n = len(X)
    t_stat = np.sqrt(n - 3) * r / np.sqrt(1 - r**2)
    p_val =  (1 - t.cdf(np.abs(t_stat), df=n - 3)) *2 #two sided
    p_val =  (1 - t.cdf(t_stat, df=n - 3)) #one sided

    return eta2, p_val

def analyze_scaling(df):
    results = {}
    targets = [
        "Volume^1/3(Shh), um",
        "x0(Shh), um",
        "Volume^1/3(Fgf8), um",
        "x0(Fgf8), um",
        "Volume^1/3(Dusp), um"
    ]

    for X_col in targets:
        # Drop NaNs only for this target and shared variables
        sub_df = df[["Exact animal size (snout to tail), cm", "Blastema width, um", X_col]].dropna()
        Y = sub_df["Exact animal size (snout to tail), cm"].values
        Z = sub_df["Blastema width, um"].values
        X = sub_df[X_col].values

        eta_stat, p_stat = compute_eta(X, Y, Z)
        eta_dyn, p_dyn = compute_eta(X, Z, Y)

        results[X_col] = {
            'eta_dyn': eta_dyn,
            'p_dyn': p_dyn,
            'eta_stat': eta_stat,
            'p_stat': p_stat
        }

    return results

def get_significance_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."

def plot_eta(results, font_size=14):
    params = list(results.keys())
    fig, axes = plt.subplots(len(params), 1, figsize=(8, 2.5 * len(params)))

    if len(params) == 1:
        axes = [axes]

    for i, param in enumerate(params):
        eta_dyn = results[param]['eta_dyn']
        eta_stat = results[param]['eta_stat']
        p_dyn = results[param]['p_dyn']
        p_stat = results[param]['p_stat']

        print(f"Parameter: {param}")
        print(f"Dynamic scaling (η²): {eta_dyn:.4f}, p-value: {p_dyn:.4f}")
        print(f"Static scaling (η²): {eta_stat:.4f}, p-value: {p_stat:.4f}")
        print()

        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        split = eta_dyn / (eta_dyn + eta_stat) if (eta_dyn + eta_stat) > 0 else 0.5

        ax.barh(0.5, width=split, left=0, height=0.4, color="#c7f0fc")
        ax.barh(0.5, width=1 - split, left=split, height=0.4, color="#ecf6e1")

        label_dyn = get_significance_label(p_dyn)
        label_stat = get_significance_label(p_stat)

        if label_dyn:
            pos = max(split / 2, 0.03)
            ax.text(pos, 0.5, label_dyn, ha='center', va='center', fontsize=font_size)
        if label_stat:
            pos = min(split + (1 - split) / 2, 0.97)
            ax.text(pos, 0.5, label_stat, ha='center', va='center', fontsize=font_size)

        ax.set_yticks([])
        ax.set_title(param)
        ax.set_xlabel('Explained Variance (η²) Split: Dynamic (left) vs Static (right)')

    plt.tight_layout()
    plt.show()

def save_eta_svgs(results, font_size=14):
    import os
    os.makedirs('statistical_analysis/fig', exist_ok=True)

    for param, res in results.items():
        eta_dyn = res['eta_dyn']
        eta_stat = res['eta_stat']
        p_dyn = res['p_dyn']
        p_stat = res['p_stat']

        split = eta_dyn / (eta_dyn + eta_stat) if (eta_dyn + eta_stat) > 0 else 0.5

        fig, ax = plt.subplots(figsize=(4, 1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.barh(0.5, width=split, left=0, height=0.4, color="#c7f0fc")
        ax.barh(0.5, width=1 - split, left=split, height=0.4, color="#ecf6e1")

        label_dyn = get_significance_label(p_dyn)
        label_stat = get_significance_label(p_stat)

        if label_dyn:
            pos = max(split / 2, 0.03)
            ax.text(pos, 0.5, label_dyn, ha='center', va='center', fontsize=font_size)
        if label_stat:
            pos = min(split + (1 - split) / 2, 0.97)
            ax.text(pos, 0.5, label_stat, ha='center', va='center', fontsize=font_size)

        ax.axis('off')
        plt.tight_layout()

        filename = f"eta_plot_{param.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.svg"
        plt.savefig('statistical_analysis/fig/' + filename, format='svg', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    df = get_data()
    print(df)
    results = analyze_scaling(df)
    plot_eta(results, font_size=14)
    save_eta_svgs(results, font_size=14)