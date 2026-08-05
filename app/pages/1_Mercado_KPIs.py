from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.utils import load_data, query_with_duckdb


st.title("1. Mercado y KPIs")

data = load_data()
monthly = data["monthly"]

segmentos = ["todos"] + sorted(monthly["segmento"].unique().tolist())
rutas = ["todas"] + sorted(monthly["ruta_logistica"].unique().tolist())

c1, c2 = st.columns(2)
segmento = c1.selectbox("Segmento", segmentos, index=0)
ruta = c2.selectbox("Ruta", rutas, index=0)

filtered = monthly.copy()
if segmento != "todos":
    filtered = filtered[filtered["segmento"] == segmento]
if ruta != "todas":
    filtered = filtered[filtered["ruta_logistica"] == ruta]

kpis = query_with_duckdb(
    filtered,
    """
    SELECT
      SUM(volume_tons) AS volumen_total,
      SUM(value_usd) AS valor_total,
      SUM(shipment_count) AS envios,
      AVG(incidence_probability) AS riesgo
    FROM trade
    """,
).iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Volumen (tons)", f"{kpis['volumen_total']:,.0f}")
col2.metric("Valor (USD)", f"${kpis['valor_total']:,.0f}")
col3.metric("Envios", f"{kpis['envios']:,.0f}")
col4.metric("Riesgo medio", f"{kpis['riesgo']:.2%}")

monthly_trend = query_with_duckdb(
    filtered,
    """
    SELECT month, SUM(volume_tons) AS volumen, SUM(value_usd) AS valor
    FROM trade
    GROUP BY month
    ORDER BY month
    """,
)

fig_vol = px.line(monthly_trend, x="month", y="volumen", title="Evolucion mensual de volumen")
fig_val = px.line(monthly_trend, x="month", y="valor", title="Evolucion mensual de valor")
st.plotly_chart(fig_vol, use_container_width=True)
st.plotly_chart(fig_val, use_container_width=True)

route_risk = query_with_duckdb(
    filtered,
    """
    SELECT
      ruta_logistica,
      SUM(value_usd) AS valor_total,
      AVG(incidence_probability) AS riesgo_promedio
    FROM trade
    GROUP BY ruta_logistica
    ORDER BY valor_total DESC
    """,
)
fig_route = px.bar(
    route_risk,
    x="ruta_logistica",
    y="valor_total",
    color="riesgo_promedio",
    title="Valor economico por ruta y riesgo promedio",
    color_continuous_scale="RdYlGn_r",
)
st.plotly_chart(fig_route, use_container_width=True)

