from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analytics.simulation import run_scenarios
from app.utils import load_config, load_data


st.title("4. Simulador de escenarios")
st.markdown("**Perdida Esperada = P(incidencia cadena de frio) * valor_carga * tiempo_transporte**")

cfg = load_config()
defaults = cfg.get("scenario_defaults", {})

data = load_data()
monthly = data["monthly"]
cutoff = monthly["month"].max() - pd.DateOffset(months=11)
base_df = monthly[monthly["month"] >= cutoff].copy()

c1, c2, c3 = st.columns(3)
adoption_rate = c1.slider(
    "Escenario 1: tasa adopcion",
    0.0,
    1.0,
    float(defaults.get("adoption_rate", 0.30)),
    0.01,
)
adoption_effectiveness = c2.slider(
    "Escenario 1: efectividad mitigacion",
    0.0,
    1.0,
    float(defaults.get("adoption_effectiveness", 0.35)),
    0.01,
)
risk_multiplier = c3.slider(
    "Escenario 2: multiplicador riesgo",
    1.0,
    2.0,
    float(defaults.get("risk_multiplier", 1.15)),
    0.01,
)
expansion_growth = st.slider(
    "Escenario 3: crecimiento comercial",
    0.0,
    1.0,
    float(defaults.get("expansion_growth", 0.20)),
    0.01,
)

summary, details = run_scenarios(
    base_df,
    adoption_rate=adoption_rate,
    adoption_effectiveness=adoption_effectiveness,
    risk_multiplier=risk_multiplier,
    expansion_growth=expansion_growth,
)

st.dataframe(summary, use_container_width=True)

fig = px.bar(
    summary,
    x="escenario",
    y="perdida_esperada_usd",
    color="loss_rate",
    title="Comparativo de perdida esperada por escenario",
    color_continuous_scale="YlOrRd",
)
st.plotly_chart(fig, use_container_width=True)

selected = st.selectbox("Detalle por escenario", summary["escenario"].tolist())
st.dataframe(
    details[selected][
        ["empresa", "segmento", "ruta_logistica", "value_usd", "incidence_probability_model", "expected_loss"]
    ].sort_values("expected_loss", ascending=False).head(30),
    use_container_width=True,
)
