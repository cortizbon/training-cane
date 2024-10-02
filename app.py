import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils import simulate_arma, plot_time_series_acf_pacf, simulate_data, run_regression, generate_correlation_matrix

st.set_page_config(layout='wide')

st.title("Simulate processes")


tab1, tab2 = st.tabs(['ARMA', 'Linear'])
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