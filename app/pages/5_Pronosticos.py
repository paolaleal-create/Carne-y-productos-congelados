from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analytics.forecasting import forecast_series, prepare_monthly_series
from app.utils import load_data


st.title("5. Pronosticos")

data = load_data()
monthly = data["monthly"]
companies = data["companies"]

c1, c2, c3, c4 = st.columns(4)
target = c1.selectbox(
    "Variable objetivo",
    ["volume_tons", "value_usd", "shipment_count"],
    format_func=lambda x: {"volume_tons": "volumen", "value_usd": "valor", "shipment_count": "envios"}[x],
)
segment = c2.selectbox("Segmento", ["todos", "bovino", "porcino"], index=0)
model_name = c3.selectbox("Modelo", ["ARIMA", "SARIMA", "Holt-Winters"], index=1)
horizon = c4.selectbox("Horizonte (meses)", [3, 6, 12], index=1)

company_opts = {"Todas": None}
company_opts.update({row.empresa: int(row.company_id) for row in companies.itertuples(index=False)})
company_label = st.selectbox("Empresa (opcional)", list(company_opts.keys()), index=0)
company_id = company_opts[company_label]

series = prepare_monthly_series(monthly, target_col=target, segment=segment, company_id=company_id)
if len(series) < 18:
    st.warning("No hay suficientes datos para pronosticar con la seleccion actual.")
    st.stop()

result = forecast_series(series, model_name=model_name, horizon=horizon)

hist = series
fc = result["forecast_df"]
test = result["test"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Historico"))
if test is not None:
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines", name="Real (holdout)"))
fig.add_trace(go.Scatter(x=fc["month"], y=fc["forecast"], mode="lines", name="Forecast"))
fig.add_trace(
    go.Scatter(
        x=list(fc["month"]) + list(fc["month"][::-1]),
        y=list(fc["upper"]) + list(fc["lower"][::-1]),
        fill="toself",
        fillcolor="rgba(0, 100, 200, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Intervalo",
    )
)
fig.update_layout(title=f"{model_name} - Pronostico de {target}", xaxis_title="Mes", yaxis_title=target)
st.plotly_chart(fig, use_container_width=True)

m1, m2 = st.columns(2)
m1.metric("MAPE", "N/A" if result["mape"] != result["mape"] else f"{result['mape']:.2f}%")
m2.metric("RMSE", "N/A" if result["rmse"] != result["rmse"] else f"{result['rmse']:.2f}")

st.dataframe(fc, use_container_width=True)

