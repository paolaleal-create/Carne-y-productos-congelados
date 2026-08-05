from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.utils import load_data, query_with_duckdb


st.set_page_config(page_title="CL Circular Analytics", page_icon=":snowflake:", layout="wide")

st.title("CL Circular | Analitica Estrategica de Carnes Congeladas")
st.markdown(
    """
Plataforma local para identificar **clientes potenciales** con alto riesgo logístico y alto valor económico
en el corredor comercial **Mexico-Estados Unidos**.
"""
)

data = load_data()
monthly = data["monthly"]

kpis = query_with_duckdb(
    monthly,
    """
    SELECT
      SUM(volume_tons) AS volumen_total,
      SUM(value_usd) AS valor_total,
      AVG(incidence_probability) AS riesgo_promedio,
      COUNT(DISTINCT company_id) AS empresas_activas
    FROM trade
    """,
)
row = kpis.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Volumen total (tons)", f"{row['volumen_total']:,.0f}")
col2.metric("Valor total (USD)", f"${row['valor_total']:,.0f}")
col3.metric("Riesgo logistico medio", f"{row['riesgo_promedio']:.2%}")
col4.metric("Empresas activas", f"{int(row['empresas_activas'])}")

st.subheader("Pregunta de negocio")
st.info("Donde estan los clientes potenciales con mayor riesgo logistico y mayor valor economico para CL Circular?")

st.subheader("Paginas")
st.markdown(
    """
1. Mercado y KPIs
2. Ranking de empresas
3. Mapa de flujos
4. Simulador de escenarios
5. Pronosticos
6. Clustering
7. Generador de leads
"""
)

