from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "volumen_anual",
    "valor_anual",
    "frecuencia_envios",
    "distancia_promedio",
    "riesgo_logistico",
    "congestion_promedio",
    "opportunity_score",
]


def build_feature_table(monthly_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    cutoff = monthly_df["month"].max() - pd.DateOffset(months=11)
    recent = monthly_df[monthly_df["month"] >= cutoff].copy()

    features = (
        recent.groupby(["company_id", "empresa"], as_index=False)
        .agg(
            volumen_anual=("volume_tons", "sum"),
            valor_anual=("value_usd", "sum"),
            frecuencia_envios=("shipment_count", "sum"),
            distancia_promedio=("distance_km", "mean"),
            riesgo_logistico=("incidence_probability", "mean"),
            congestion_promedio=("congestion_border", "mean"),
        )
        .merge(
            ranking_df[["company_id", "opportunity_score"]],
            on="company_id",
            how="left",
        )
        .fillna({"opportunity_score": 0.0})
    )
    return features


def clustering_diagnostics(feature_df: pd.DataFrame, max_k: int = 8) -> pd.DataFrame:
    max_k = max(3, min(max_k, len(feature_df) - 1))
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df[FEATURE_COLUMNS])

    rows = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_clusters(feature_df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df[FEATURE_COLUMNS])

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X)

    out = feature_df.copy()
    out["cluster"] = labels
    out["pca_1"] = components[:, 0]
    out["pca_2"] = components[:, 1]

    profile = (
        out.groupby("cluster", as_index=False)[FEATURE_COLUMNS]
        .mean()
        .merge(out.groupby("cluster", as_index=False).size(), on="cluster")
        .rename(columns={"size": "n_empresas"})
        .sort_values("opportunity_score", ascending=False)
    )

    return out, profile

