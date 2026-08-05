from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS: Dict[str, float] = {
    "volume_tons": 0.22,
    "value_usd": 0.26,
    "distance_km": 0.12,
    "sensitivity_temperature": 0.14,
    "risk_congestion": 0.14,
    "incidences_proxy": 0.12,
}


def _normalize(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - min_v) / (max_v - min_v)


def apply_opportunity_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    required = [
        "volume_tons",
        "value_usd",
        "distance_km",
        "sensitivity_temperature",
        "risk_congestion",
        "incidences_proxy",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for score: {missing}")

    out = df.copy()
    weight_sum = sum(max(v, 0) for v in weights.values())
    if weight_sum == 0:
        raise ValueError("Weights must sum to a positive value.")

    contribution_cols = []
    for col in required:
        w = max(weights.get(col, 0), 0)
        norm_col = f"norm_{col}"
        contrib_col = f"contrib_{col}"
        out[norm_col] = _normalize(out[col].astype(float))
        out[contrib_col] = out[norm_col] * w
        contribution_cols.append(contrib_col)

    out["opportunity_score"] = 100 * out[contribution_cols].sum(axis=1) / weight_sum
    return out


def _top_drivers(row: pd.Series, weights: Dict[str, float], top_n: int = 3) -> str:
    names = {
        "volume_tons": "volumen",
        "value_usd": "valor",
        "distance_km": "distancia",
        "sensitivity_temperature": "sensibilidad temp",
        "risk_congestion": "congestion",
        "incidences_proxy": "incidencias",
    }
    scored = []
    for feature in names:
        contrib = row.get(f"contrib_{feature}", 0.0)
        scored.append((feature, float(contrib) * max(weights.get(feature, 0), 0)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return ", ".join(names[f] for f, _ in scored[:top_n])


def rank_companies(
    monthly_df: pd.DataFrame,
    weights: Dict[str, float] | None = None,
    lookback_months: int = 12,
) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    cutoff = monthly_df["month"].max() - pd.DateOffset(months=lookback_months - 1)
    recent = monthly_df[monthly_df["month"] >= cutoff].copy()

    grouped = (
        recent.groupby(["company_id", "empresa", "segmento"], as_index=False)
        .agg(
            volume_tons=("volume_tons", "sum"),
            value_usd=("value_usd", "sum"),
            distance_km=("distance_km", "mean"),
            sensitivity_temperature=("sensitivity_temperature", "mean"),
            risk_congestion=("congestion_border", "mean"),
            incidences_proxy=("incidences_proxy", "mean"),
            incidence_probability=("incidence_probability", "mean"),
            shipment_count=("shipment_count", "sum"),
        )
    )

    top_route = (
        recent.groupby(["company_id", "ruta_logistica"], as_index=False)["value_usd"]
        .sum()
        .sort_values(["company_id", "value_usd"], ascending=[True, False])
        .drop_duplicates("company_id")
        .rename(columns={"ruta_logistica": "ruta_principal"})
        [["company_id", "ruta_principal"]]
    )

    ranked = apply_opportunity_score(grouped, weights).merge(top_route, on="company_id", how="left")
    ranked["drivers_score"] = ranked.apply(_top_drivers, axis=1, weights=weights)
    ranked = ranked.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    return ranked

