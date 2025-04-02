import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import streamlit as st

def obtener_datos(stocks):
    '''
    Descarga el precio de cierre de uno o varios activos desde 2010.
    Input: Ticker del activo (string o lista)
    Output: DataFrame de precios
    '''
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

def calcular_rendimientos(df):
    '''
    Calcula los rendimientos diarios del activo.
    Input: DataFrame de precios
    Output: DataFrame de rendimientos (porcentajes)
    '''
    return df.pct_change().dropna()

# Configuración de Streamlit
st.title("Análisis de Rendimientos Diarios 📊")
st.markdown("---")

# Definir el activo (puedes cambiarlo o hacerlo interactivo)
M7 = ['AAPL']  # Ejemplo con Apple

# Obtener datos
df_precios = obtener_datos(M7)
df_rendimientos = calcular_rendimientos(df_precios)

# Mostrar precios y rendimientos (opcional, puedes comentar estas líneas)
st.subheader("Precios de Cierre (Últimos 5 días)")
st.dataframe(df_precios.tail())

st.subheader("Rendimientos Diarios (Últimos 5 días)")
st.dataframe(df_rendimientos.tail())

# Calcular estadísticos
media_diaria = df_rendimientos['AAPL'].mean()
sesgo = skew(df_rendimientos['AAPL'])
exceso_curtosis = kurtosis(df_rendimientos['AAPL'], fisher=True)  # Fisher=True para exceso

# Mostrar resultados en Streamlit
st.markdown("---")
st.subheader("Estadísticos Clave")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="**Media Diaria**", value=f"{media_diaria:.4%}")

with col2:
    st.metric(label="**Sesgo**", value=f"{sesgo:.2f}")

with col3:
    st.metric(label="**Exceso de Curtosis**", value=f"{exceso_curtosis:.2f}")

# Interpretación
st.markdown("---")
st.subheader("Interpretación:")
st.write("- **Media Diaria**: Rendimiento promedio diario del activo.")
st.write("- **Sesgo**: Indica asimetría. >0 = cola derecha; <0 = cola izquierda.")
st.write("- **Exceso de Curtosis**: Mide 'peso' de colas. >0 = más extremos que una normal; <0 = menos extremos.")

# Gráfico opcional de rendimientos
st.markdown("---")
st.subheader("Distribución de Rendimientos")
fig, ax = plt.subplots()
df_rendimientos['AAPL'].hist(bins=50, alpha=0.6, ax=ax)
ax.set_xlabel("Rendimiento Diario")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)


st.subheader("📈 Análisis Financiero Completo")

# Funciones base
def obtener_datos(stocks):
    '''Descarga precios de cierre desde 2010'''
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

def calcular_rendimientos(df):
    '''Calcula rendimientos porcentuales diarios'''
    return df.pct_change().dropna()

# Funciones de riesgo (nuevas)
def calcular_var_parametrico(returns, alpha, distrib='normal'):
    '''Calcula VaR paramétrico'''
    mean = np.mean(returns)
    stdev = np.std(returns)
    
    if distrib == 'normal':
        VaR = norm.ppf(1-alpha, mean, stdev)
    elif distrib == 't':
        df_t = 5  # Grados de libertad ajustables
        VaR = t.ppf(1-alpha, df_t, mean, stdev)
    return VaR

def calcular_var_historico(returns, alpha):
    '''VaR usando método histórico'''
    return returns.quantile(1-alpha)

def calcular_var_montecarlo(returns, alpha, n_sims=10000):
    '''VaR con simulación Monte Carlo'''
    mean = np.mean(returns)
    stdev = np.std(returns)
    sim_returns = np.random.normal(mean, stdev, n_sims)
    return np.percentile(sim_returns, (1-alpha)*100)

def calcular_cvar(returns, VaR):
    '''Calcula Conditional VaR (pérdida esperada)'''
    return returns[returns <= VaR].mean()

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    ticker = st.text_input("Selecciona un ticker", "AAPL")
    conf_levels = st.multiselect(
        "Niveles de confianza para VaR",
        options=[0.90, 0.95, 0.975, 0.99],
        default=[0.95, 0.99]
    )
    metodo_var = st.multiselect(
        "Métodos de VaR a mostrar",
        options=["Paramétrico (Normal)", "Paramétrico (t-Student)", "Histórico", "Monte Carlo"],
        default=["Paramétrico (Normal)", "Histórico"]
    )

# Obtención y cálculo de datos
try:
    df_precios = obtener_datos([ticker])
    retornos = calcular_rendimientos(df_precios)[ticker]
    
    # Sección 1: Estadísticas básicas
    st.header("📊 Estadísticas Descriptivas")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Media diaria", f"{retornos.mean():.4%}")
    with col2:
        st.metric("Volatilidad", f"{retornos.std():.4%}")
    with col3:
        st.metric("Sesgo", f"{skew(retornos):.2f}")
    with col4:
        st.metric("Curtosis", f"{kurtosis(retornos, fisher=True):.2f}")
    
    # Gráfico de distribución
    fig, ax = plt.subplots()
    retornos.hist(bins=50, alpha=0.6, ax=ax)
    ax.set_title(f"Distribución de Rendimientos - {ticker}")
    st.pyplot(fig)
    
    # Sección 2: Análisis de Riesgo
    st.header("⚠️ Análisis de Riesgo (VaR y CVaR)")
    
    if not conf_levels:
        st.warning("Selecciona al menos un nivel de confianza")
    else:
        # Crear DataFrame para resultados
        resultados = pd.DataFrame(index=[f"{int(cl*100)}%" for cl in sorted(conf_levels, reverse=True)],
                                columns=[m for m in metodo_var] + [f"CVaR ({m.split(')')[0]})" for m in metodo_var])
        
        # Calcular para cada nivel de confianza
        for cl in conf_levels:
            row_name = f"{int(cl*100)}%"
            
            if "Paramétrico (Normal)" in metodo_var:
                var_n = calcular_var_parametrico(retornos, cl, 'normal')
                resultados.loc[row_name, "Paramétrico (Normal)"] = var_n * 100
                resultados.loc[row_name, "CVaR (Paramétrico (Normal)"] = calcular_cvar(retornos, var_n) * 100
            
            if "Paramétrico (t-Student)" in metodo_var:
                var_t = calcular_var_parametrico(retornos, cl, 't')
                resultados.loc[row_name, "Paramétrico (t-Student)"] = var_t * 100
                resultados.loc[row_name, "CVaR (Paramétrico (t-Student)"] = calcular_cvar(retornos, var_t) * 100
            
            if "Histórico" in metodo_var:
                var_h = calcular_var_historico(retornos, cl)
                resultados.loc[row_name, "Histórico"] = var_h * 100
                resultados.loc[row_name, "CVaR (Histórico"] = calcular_cvar(retornos, var_h) * 100
            
            if "Monte Carlo" in metodo_var:
                var_mc = calcular_var_montecarlo(retornos, cl)
                resultados.loc[row_name, "Monte Carlo"] = var_mc * 100
                resultados.loc[row_name, "CVaR (Monte Carlo"] = calcular_cvar(retornos, var_mc) * 100
        
        # Mostrar resultados
        st.dataframe(resultados.style.format("{:.2f}%").background_gradient(cmap='Reds'))
        
        # Explicación de métodos
        with st.expander("🔍 Explicación de Métodos"):
            st.markdown("""
            - **VaR Paramétrico (Normal)**: Asume distribución normal de rendimientos
            - **VaR Paramétrico (t-Student)**: Usa distribución t con colas más pesadas (más conservador)
            - **VaR Histórico**: Basado en percentiles históricos reales
            - **VaR Monte Carlo**: Simulación de escenarios aleatorios basados en parámetros históricos
            - **CVaR**: Pérdida esperada cuando se supera el VaR (Expected Shortfall)
            """)

except Exception as e:
    st.error(f"Error al procesar los datos: {str(e)}")

# Sección de precios históricos
st.header("📈 Precios Históricos")
st.line_chart(df_precios)
