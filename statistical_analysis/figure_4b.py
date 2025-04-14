import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

# Load data
def get_data():
    df = pd.read_csv('statistical_analysis/axolot_data.csv')
    selected_columns = df[[
        "Exact animal size (snout to tail), cm",
        "Blastema size, um",
        "x0(Shh), um",
        "x0(Fgf8), um",
        "Volume^1/3(Shh), um",
        "Volume^1/3(Fgf8), um",
        "Volume^1/3(Dusp), um"
    ]]
    cleaned_df = selected_columns.dropna()
    return cleaned_df

def compute_eta(X, Y, Z):
    beta_XZ = np.polyfit(Z, X, 1)
    eps_X = X - (beta_XZ[0] * Z + beta_XZ[1])

    beta_YZ = np.polyfit(Z, Y, 1)
    eps_Y = Y - (beta_YZ[0] * Z + beta_YZ[1])

    beta_eps = np.polyfit(eps_Y, eps_X, 1)
    pred_eps_X = beta_eps[0] * eps_Y + beta_eps[1]
    eps_final = eps_X - pred_eps_X

    eta2 = 1 - np.sum(eps_final**2) / np.sum(eps_X**2)

    r = np.corrcoef(eps_Y, eps_X)[0, 1]
    n = len(X)
    t_stat = np.sqrt(n - 3) * r / np.sqrt(1 - r**2)
    p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 3))

    return eta2, p_val

def analyze_scaling(df):
    Y = df["Exact animal size (snout to tail), cm"].values
    Z = df["Blastema size, um"].values

    results = {}
    targets = [
        "x0(Shh), um",
        "x0(Fgf8), um",
        "Volume^1/3(Shh), um",
        "Volume^1/3(Fgf8), um",
        "Volume^1/3(Dusp), um"
    ]

    for X_col in targets:
        X = df[X_col].values
        eta_dyn, p_dyn = compute_eta(X, Y, Z)
        eta_stat, p_stat = compute_eta(X, Z, Y)

        results[X_col] = {
            'eta_dyn': eta_dyn,
            'p_dyn': p_dyn,
            'eta_stat': eta_stat,
            'p_stat': p_stat
        }

    return results

def plot_eta(results):
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
        print(f"Dynamic scaling (η²): {eta_dyn:.3f}, p-value: {p_dyn:.3f}")
        print(f"Static scaling (η²): {eta_stat:.3f}, p-value: {p_stat:.3f}")
        print()

        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        split = eta_dyn / (eta_dyn + eta_stat) if (eta_dyn + eta_stat) > 0 else 0.5

        ax.barh(0.5, width=split, left=0, height=0.4, color="#c7f0fc")
        ax.barh(0.5, width=1 - split, left=split, height=0.4, color="#ecf6e1")

        if p_dyn < 0.05:
            ax.text(split / 2, 0.5, '*', ha='center', va='center', fontsize=14)
        if p_stat < 0.05:
            ax.text(split + (1 - split) / 2, 0.5, '*', ha='center', va='center', fontsize=14)

        ax.set_yticks([])
        ax.set_title(param)
        ax.set_xlabel('Explained Variance (η²) Split: Dynamic (left) vs Static (right)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = get_data()
    results = analyze_scaling(df)
    plot_eta(results)