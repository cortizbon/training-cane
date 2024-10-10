import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils import simulate_arma, plot_time_series_acf_pacf, simulate_data, run_regression, generate_correlation_matrix

st.set_page_config(layout='wide')

st.title("Simulate processes")


tab1, tab2, tab3 = st.tabs(['ARMA', 'Linear', 'Inference'])
with tab1:

    process = st.selectbox("Seleccione el proceso a simular: ", ['AR(1)','AR(2)', 'MA(1)', 'MA(2)'])

    if process == 'AR(1)':
        par = st.text_input("Ingrese un número |r|<1:", "0.1")
        if abs(float(par)) < 1:
            ts_ar1 = simulate_arma([float(par)], [])
            fig, axes = plt.subplots(3, 1, figsize=(10, 4))
            plot_time_series_acf_pacf(ts_ar1, par, axes[0], axes[1], axes[2])
            fig.tight_layout()
            st.pyplot(fig)
            
        else:
            st.error("No cumple con la condición")

    elif process == 'AR(2)':
        par = st.text_input("Ingrese dos números separados por un espacio, verifique si se cumplen las condiciones: ")
        par1, par2 = [float(i) for i in par.split(",")]
        ts_ar1 = simulate_arma([par1, par2], [])
        fig, axes = plt.subplots(3, 1, figsize=(10, 4))
        plot_time_series_acf_pacf(ts_ar1, par, axes[0], axes[1], axes[2])
        fig.tight_layout()

        st.pyplot(fig)
    elif process == 'MA(1)':
        par = st.text_input("Ingrese un número:")
        ts_ar1 = simulate_arma([], [float(par)])
        fig, axes = plt.subplots(3, 1, figsize=(10, 4))
        plot_time_series_acf_pacf(ts_ar1, par, axes[0], axes[1], axes[2])
        fig.tight_layout()

        st.pyplot(fig)
    else:
        par = st.text_input("Ingrese dos números separados por un espacio: ")
        par1, par2 = [float(i) for i in par.split(",")]
        ts_ar1 = simulate_arma([], [par1, par2])
        fig, axes = plt.subplots(3, 1, figsize=(10, 4))
        plot_time_series_acf_pacf(ts_ar1, par, axes[0], axes[1], axes[2])
        fig.tight_layout()

        st.pyplot(fig)


with tab2:
    n_regs = st.text_input("Ingrese un número entero (número de regresiones): ", "1")
    if st.button("Generate"):
        for i in range(int(n_regs)):
            st.write(f"\n--- Regression {i+1} ---")
            
            # Generate random correlation matrix and simulate data
            corr_matrix = generate_correlation_matrix(4)
            df = simulate_data(corr_matrix)

            # Define dependent and independent variables
            dependent_var = 'Var1'
            independent_vars = ['Var2', 'Var3', 'Var4']

            # Run regression
            model = run_regression(df, dependent_var, independent_vars)
            
            # Display regression summary (R-squared, Adj. R-squared, F-statistic)
            st.write(model.summary())

with tab3:

    # Function to simulate data
    def simulate_variables(n_samples=100, n_variables=5, random_state=None):
        np.random.seed(random_state)
        return np.random.randn(n_samples, n_variables)

    # Function to run a regression and print summary
    def run_regression(Y, X, description="Regression"):
        X = sm.add_constant(X)  # Add intercept
        model = sm.OLS(Y, X)
        results = model.fit()
        st.write(f"=== {description} ===")
        st.write(results.summary())
        st.write()

    # Main simulation process
    n_simulations = 12
    n_samples = 100
    contador = 0

    for i in range(n_simulations):
        print(f"Simulation {i+1}")
        
        # Step 1: Simulate 3 variables for FRP
        data = simulate_variables(n_samples, 5, random_state=i)
        Y = data[:, 0]  # Dependent variable
        X1, X2, X3 = data[:, 1], data[:, 2], data[:, 3]  # FRP variables
        X4, X5 = data[:, 4], np.random.randn(n_samples)  # Additional variables
        
        # Regression 1: 3 variables (FRP)
        run_regression(Y, np.column_stack([X1, X2, X3]), f"Regresión {contador}")
        contador += 1
        # Regression 2: 4 variables (FRP + X4)
        run_regression(Y, np.column_stack([X1, X2, X3, X4]), f"Regresión {contador}")
        contador += 1
        # Regression 3: 5 variables (FRP + X4 + X5)
        run_regression(Y, np.column_stack([X1, X2, X3, X4, X5]), f"Regresión {contador}")
        contador += 1
        # Regression 4: 3 variables (2 from FRP, 1 outside FRP)
        run_regression(Y, np.column_stack([X1, X2, X4]), f"Regresión {contador}")
        contador += 1
        # Regression 5: 2 variables (both outside FRP)
        run_regression(Y, np.column_stack([X4, X5]), f"Regresión {contador}")
        contador += 1
        st.write("="*60)
