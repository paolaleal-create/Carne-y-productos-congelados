from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analytics.scoring import DEFAULT_WEIGHTS, rank_companies
from app.utils import load_data


st.title("2. Ranking de empresas")
st.caption("Opportunity Score configurable con pesos en tiempo real.")

data = load_data()
monthly = data["monthly"]

st.sidebar.subheader("Pesos del Opportunity Score")
weights = {}
for key, default in DEFAULT_WEIGHTS.items():
    weights[key] = st.sidebar.slider(key, min_value=0.0, max_value=1.0, value=float(default), step=0.01)

lookback = st.sidebar.slider("Ventana de analisis (meses)", min_value=6, max_value=24, value=12, step=1)
ranking = rank_companies(monthly, weights=weights, lookback_months=lookback)

top_n = st.slider("Top N empresas", min_value=10, max_value=50, value=20, step=5)
top = ranking.head(top_n).copy()

st.dataframe(
    top[
        [
            "empresa",
            "segmento",
            "volume_tons",
            "value_usd",
            "risk_congestion",
            "incidence_probability",
            "opportunity_score",
            "drivers_score",
            "ruta_principal",
        ]
    ].rename(
        columns={
            "volume_tons": "volumen_tons",
            "value_usd": "valor_usd",
            "risk_congestion": "riesgo_congestion",
        }
    ),
    use_container_width=True,
)

fig = px.scatter(
    ranking,
    x="value_usd",
    y="incidence_probability",
    size="volume_tons",
    color="opportunity_score",
    hover_name="empresa",
    title="Matriz de oportunidad: valor vs riesgo",
    color_continuous_scale="Turbo",
)
st.plotly_chart(fig, use_container_width=True)

