###################
#ATTENTION: This code shows equivalence in case of two sided test
###################


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import t
from causallearn.utils.cit import CIT
import pandas as pd

# Constants
N_SAMPLES = 30
N_EXP = 1000
SIGNIFICANCE_LEVEL = 0.05

def generate_data(m_BC=0.0, n_B=0.0, m_B=1.0, sigma_B=1.0, n_C=0.0, m_AC=1.0, sigma_C=1.0):

    """Generate synthetic data for A, B, and C."""

    A = np.random.uniform(0, 1, N_SAMPLES)

    e_B = np.random.normal(0, sigma_B, N_SAMPLES)

    B = n_B + m_B * A + e_B

    e_C = np.random.normal(0, sigma_C, N_SAMPLES)

    C = n_C + m_AC * A + m_BC * B + e_C

    return A, B, C

def partial_correlation(A, B, C):
    """Compute the partial correlation between B and C given A."""
    residual_B = B - np.polyval(np.polyfit(A, B, 1), A)
    residual_C = C - np.polyval(np.polyfit(A, C, 1), A)
    return np.corrcoef(residual_B, residual_C)[0, 1]

def fisher_z_test(partial_corr, n_samples):
    """Perform the Fisher Z-test."""
    z_value = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
    standard_error = 1 / np.sqrt(n_samples - 4)
    p_value = 2 * (1 - norm.cdf(abs(z_value) / standard_error))

    return z_value, p_value

def ben_test(partial_corr, n_samples):
    t_value = partial_corr * np.sqrt(n_samples - 3) / np.sqrt(1 - partial_corr ** 2)
    degrees_of_freedom = n_samples - 3
    p_value = 2 * (1 - t.cdf(abs(t_value), df=degrees_of_freedom))
    return t_value, p_value

def run_experiment(m_BC, alpha=True):
    """
    Run the statistical experiment and store results in a pandas DataFrame.

    Parameters:
        m_BC (float): The value of m_BC (0 for alpha, non-zero for beta).
        alpha (bool): Whether to compute alpha (false positive) or beta (false negative).

    Returns:
        pd.DataFrame, dict, dict: The results DataFrame, alpha/beta statistics, and their counting errors.

    """
    results = []

    for _ in range(N_EXP):
        A, B, C = generate_data(m_BC=m_BC)
        partial_corr = partial_correlation(A, B, C)

        # Custom Fisher Z-test
        z_value_custom, p_value_custom = fisher_z_test(partial_corr, N_SAMPLES)

        # Ben's test
        t_value_Ben, p_value_Ben = ben_test(partial_corr, N_SAMPLES)

        # Fisher Z-test from causal-learn
        data = np.column_stack([A, B, C])
        fisherz_obj = CIT(data, method="fisherz")
        p_value_causal_learn = fisherz_obj(1, 2, [0]) 

        # Append the results for this experiment
        results.append({
            "partial_corr": partial_corr,
            "z_value_custom": z_value_custom,
            "p_value_custom": p_value_custom,
            "t_value_Ben": t_value_Ben,
            "p_value_Ben": p_value_Ben,
            "p_value_causal_learn": p_value_causal_learn,
            "rejected_custom": int(p_value_custom < SIGNIFICANCE_LEVEL if alpha else p_value_custom >= SIGNIFICANCE_LEVEL),
            "rejected_Ben": int(p_value_Ben < SIGNIFICANCE_LEVEL if alpha else p_value_custom >= SIGNIFICANCE_LEVEL),
            "rejected_causal_learn": int(p_value_causal_learn < SIGNIFICANCE_LEVEL if alpha else p_value_causal_learn >= SIGNIFICANCE_LEVEL),
        })

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Calculate statistics
    alpha_or_beta = {
        "custom": df_results["rejected_custom"].mean(),
        "Ben": df_results["rejected_Ben"].mean(),
        "causal_learn": df_results["rejected_causal_learn"].mean(),
    }

    # Calculate counting error
    error = {
        "custom": np.sqrt(alpha_or_beta["custom"] * (1 - alpha_or_beta["custom"]) / N_EXP),
        "Ben": np.sqrt(alpha_or_beta["Ben"] * (1 - alpha_or_beta["Ben"]) / N_EXP),
        "causal_learn": np.sqrt(alpha_or_beta["causal_learn"] * (1 - alpha_or_beta["causal_learn"]) / N_EXP),
    }

    return df_results, alpha_or_beta, error

def plot_histogram(data, title, xlabel):

    """Plot histogram of partial correlation or Z values."""

    plt.hist(data, bins=30, color='skyblue', edgecolor='black')

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel("Frequency")

    plt.show()



####################################################################################################
df_alpha, alpha, alpha_error = run_experiment(m_BC=0.0, alpha=True)
print("Empirical alpha (custom): {:.4f} ± {:.4f}".format(alpha["custom"], alpha_error["custom"]))
print("Empirical alpha (Ben): {:.4f} ± {:.4f}".format(alpha["Ben"], alpha_error["Ben"]))
print("Empirical alpha (causal-learn): {:.4f} ± {:.4f}".format(alpha["causal_learn"], alpha_error["causal_learn"]))
print(df_alpha.head())  # Preview the DataFrame

df_alpha["partial_corr"].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Partial Correlation Histogram (m_BC=0)")
plt.xlabel("Partial Correlation")
plt.ylabel("Frequency")
plt.show()

df_alpha["z_value_custom"].hist(bins=30, color='lightgreen', edgecolor='black')
plt.title("z-Value Histogram (Custom Fisher Z-Test, m_BC=0)")
plt.xlabel("z-Value")
plt.ylabel("Frequency")
plt.show()


df_alpha["t_value_Ben"].hist(bins=30, color='lightgreen', edgecolor='black')
plt.title("t-Value Histogram (Bens test, m_BC=0)")
plt.xlabel("t-Value")
plt.ylabel("Frequency")
plt.show()



####################################################################################################
# For beta (m_BC = 0.5)
df_beta, beta, beta_error = run_experiment(m_BC=0.5, alpha=False)
print("Empirical beta (custom): {:.4f} ± {:.4f}".format(beta["custom"], beta_error["custom"]))
print("Empirical beta (Ben): {:.4f} ± {:.4f}".format(beta["Ben"], beta_error["Ben"]))
print("Empirical beta (causal-learn): {:.4f} ± {:.4f}".format(beta["causal_learn"], beta_error["causal_learn"]))
print(df_beta.head())  # Preview the DataFrame


df_beta["partial_corr"].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Partial Correlation Histogram (m_BC=0.5)")
plt.xlabel("Partial Correlation")
plt.ylabel("Frequency")
plt.show()


df_beta["z_value_custom"].hist(bins=30, color='lightgreen', edgecolor='black')
plt.title("z-Value Histogram (Custom Fisher Z-Test, m_BC=0.5)")
plt.xlabel("z-Value")
plt.ylabel("Frequency")
plt.show()



df_beta["t_value_Ben"].hist(bins=30, color='lightgreen', edgecolor='black')
plt.title("t-Value Histogram (Bens test, m_BC=0.5)")
plt.xlabel("t-Value")
plt.ylabel("Frequency")
plt.show()