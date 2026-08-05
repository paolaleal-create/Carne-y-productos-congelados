from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analytics.clustering import build_feature_table, clustering_diagnostics, fit_clusters
from analytics.scoring import DEFAULT_WEIGHTS, rank_companies
from app.utils import load_data


def cluster_label(row) -> str:
    if row["opportunity_score"] >= 70 and row["riesgo_logistico"] >= 0.55:
        return "Alto volumen / alto riesgo / alta oportunidad"
    if row["opportunity_score"] >= 55:
        return "Medio volumen / alto crecimiento"
    return "Bajo volumen / baja prioridad"


st.title("6. Clustering de clientes")

data = load_data()
monthly = data["monthly"]
ranking = rank_companies(monthly, weights=DEFAULT_WEIGHTS, lookback_months=12)
features = build_feature_table(monthly, ranking)

max_k = st.slider("Maximo K para metodo del codo", min_value=4, max_value=10, value=8, step=1)
diag = clustering_diagnostics(features, max_k=max_k)

fig_elbow = px.line(diag, x="k", y="inertia", markers=True, title="Metodo del codo")
fig_sil = px.line(diag, x="k", y="silhouette", markers=True, title="Silhouette score")
st.plotly_chart(fig_elbow, use_container_width=True)
st.plotly_chart(fig_sil, use_container_width=True)

best_k = int(diag.sort_values("silhouette", ascending=False).iloc[0]["k"])
k = st.slider("Numero de clusters (K-Means)", min_value=2, max_value=max_k, value=best_k, step=1)

clustered, profile = fit_clusters(features, k=k)
profile["perfil"] = profile.apply(cluster_label, axis=1)

fig_pca = px.scatter(
    clustered,
    x="pca_1",
    y="pca_2",
    color=clustered["cluster"].astype(str),
    size="opportunity_score",
    hover_name="empresa",
    title="PCA 2D de empresas clusterizadas",
)
st.plotly_chart(fig_pca, use_container_width=True)

st.subheader("Perfiles de clusters")
st.dataframe(profile, use_container_width=True)

