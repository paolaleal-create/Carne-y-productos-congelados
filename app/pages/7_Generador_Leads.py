from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analytics.allocation import generate_leads
from analytics.clustering import build_feature_table, fit_clusters
from analytics.scoring import DEFAULT_WEIGHTS, rank_companies
from app.utils import load_data


st.title("7. Generador de leads")
st.caption("Exporta leads comerciales priorizados para CL Circular.")

data = load_data()
monthly = data["monthly"]

ranking = rank_companies(monthly, weights=DEFAULT_WEIGHTS, lookback_months=12)
features = build_feature_table(monthly, ranking)
clustered, _ = fit_clusters(features, k=3)

top_n = st.slider("Numero de leads a exportar", min_value=10, max_value=80, value=30, step=5)
leads = generate_leads(ranking, clustered, top_n=top_n)

st.dataframe(leads, use_container_width=True)

csv = leads.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV de leads",
    data=csv,
    file_name="leads_priorizados_cl_circular.csv",
    mime="text/csv",
)

st.subheader("Ejemplo de pitch sugerido")
if not leads.empty:
    row = leads.iloc[0]
    st.write(f"**Empresa:** {row['empresa']}")
    st.write(f"**Pitch:** {row['pitch_comercial_sugerido']}")

