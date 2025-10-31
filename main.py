import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_x_analysis(t, x, title=None):
    xo = x[0]  # Initial value xo

    # Define model function / analytical solution
    def x_func(t, xinf, k):
        return xinf + (xo - xinf) * np.exp(-k * t)

    # Fit curve
    p0 = [min(x), 0.01]
    param, param_cov = curve_fit(x_func, t, x, p0=p0)
    x_inf, k_fit = param

    # Calculate fitted values
    x_fit = x_func(t, *param)
    # R square calculation:
    r2_curve_fit = 1 - np.sum((x - x_fit) ** 2) / np.sum((x - np.mean(x)) ** 2)

    # Log-linear transformation and Regression to verify results
    y_log = np.log(x - x_inf)
    model = LinearRegression(fit_intercept=True)
    model.fit(t.reshape(-1, 1), y_log)
    y_log_fit = model.predict(t.reshape(-1, 1))
    # R square calc
    r2_log_linear = 1 - np.sum((y_log - y_log_fit) ** 2) / np.sum((y_log - np.mean(y_log)) ** 2)

    # Numerical derivative
    dx_dt = np.gradient(x, t)

    # theoretical derivative with k anf xif we found
    def deriv(x):
        return k_fit * (x - x_inf)

    # --------------plots --------------------
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Original data + fitted curve
    axs[0, 0].plot(t, x, 'o', label='Data')
    axs[0, 0].plot(t, x_fit, '--', label='Fit')
    axs[0, 0].set_xlabel("t [min]")
    axs[0, 0].set_ylabel("x(t)")
    axs[0, 0].set_title("Data and Fit")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    param_text = f"x_inf = {x_inf:.4f}\nk = {k_fit:.4f}\n R^2 = {r2_curve_fit:.4f}"
    axs[0, 0].text(0.25, 0.05, param_text, transform=axs[0, 0].transAxes, fontsize=10, verticalalignment='bottom',
                   horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    # 2. Extended model prediction
    t_new = np.linspace(0, max(t) * 10, 200)
    axs[0, 1].plot(t_new, x_func(t_new, *param), '--', label='Model Prediction')
    axs[0, 1].set_xlabel("t [min]")
    axs[0, 1].set_ylabel("x(t) from model")
    axs[0, 1].set_title("Extended Model x(t) = x_inf + (xo - x_inf) * exp(-k * t)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Log-linear plot
    axs[1, 0].plot(t, y_log, 'o', label='Data')
    axs[1, 0].plot(t, y_log_fit, '-', label=f'Linear fit')
    axs[1, 0].set_xlabel("Time t")
    axs[1, 0].set_ylabel("ln(x - x_inf)")
    axs[1, 0].set_title("Log Transformation")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    param_text = f"Estimated\nfrom log fit\nk = {-model.coef_[0]:.4f}\n R^2 = {r2_log_linear:.4f}"
    axs[1, 0].text(0.05, 0.05, param_text, transform=axs[1, 0].transAxes, fontsize=10, verticalalignment='bottom',
                   horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    # 4. Numerical derivative dx/dt vs x
    axs[1, 1].scatter(x, abs(dx_dt), color='red', marker='o', label='Deriv from data points')
    axs[1, 1].plot(x, deriv(x), '--', label='dx/dt = k(x-xe)')
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("dx/dt")
    axs[1, 1].set_title("Numerical Derivative")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Add figure title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
    plt.show()

    # Save the plot and ensure plots folder exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    fig.savefig(f'plots/{title}.png')
    plt.show()
    plt.close(fig)
    return fig


# --- Function to read Excel and call analysis for each x column ---
def analyze_excel(filename):
    """
    Reads Excel file with columns: t, x1, x2, x3, x4
    and generates plots for each x_i.
    """
    df = pd.read_excel(filename)
    t = df['t'].to_numpy(dtype=float)

    derivatives = pd.DataFrame({'t': t})

    # calculate the analysis for each column in the Excel file
    for col in df.columns[1:]:
        x = df[col].to_numpy(dtype=float)
        plot_x_analysis(t, x, title=col)

        # Compute numerical derivative
        dx_dt = np.gradient(x, t)
        derivatives[f'dx_dt_{col}'] = dx_dt

    # Save derivative to Excel
    output_filename = '../derivatives_output.xlsx'
    derivatives.to_excel(output_filename, index=False)
    print(f"\n Derivatives saved to '{output_filename}'")


# Call the function with your filename
analyze_excel('Ξήρανση.xlsx')
