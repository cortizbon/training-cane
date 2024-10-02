import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wishart


# Function to simulate ARMA processes
def simulate_arma(ar_coefs, ma_coefs, n=500):
    ar = np.r_[1, -np.array(ar_coefs)]  # AR parameters
    ma = np.r_[1, np.array(ma_coefs)]   # MA parameters
    arma_process = ArmaProcess(ar, ma)
    return arma_process.generate_sample(nsample=n)

# Plotting function
def plot_time_series_acf_pacf(ts_data, title, ax_ts, ax_acf, ax_pacf):
    ax_ts.plot(ts_data, color='blue')
    ax_ts.set_title(title)
    plot_acf(ts_data, ax=ax_acf, lags=30)
    plot_pacf(ts_data, ax=ax_pacf, lags=30)


# Function to generate random correlation matrix
def generate_correlation_matrix(n):
    # Generate random positive definite matrix using Wishart distribution
    A = np.random.randn(n, n)
    corr_matrix = np.dot(A, A.T)
    
    # Normalize to make it a proper correlation matrix
    D = np.diag(1 / np.sqrt(np.diag(corr_matrix)))
    corr_matrix = D @ corr_matrix @ D
    return corr_matrix

# Function to simulate data based on the correlation matrix
def simulate_data(corr_matrix, n_samples=500):
    mean = np.zeros(corr_matrix.shape[0])
    data = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)
    df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(corr_matrix.shape[0])])
    return df

# Function to run linear regression and display summary
def run_regression(df, dependent_var, independent_vars):
    X = df[independent_vars]
    X = sm.add_constant(X)  # Add constant for intercept
    y = df[dependent_var]
    model = sm.OLS(y, X).fit()
    return model

