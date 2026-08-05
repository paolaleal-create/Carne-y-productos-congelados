from __future__ import annotations

import numpy as np
import pandas as pd


def _logistic(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1 / (1 + np.exp(-x))


def estimate_incidence_probability(df: pd.DataFrame) -> pd.Series:
    return _logistic(
        -2.25
        + 1.75 * df["congestion_border"]
        + 1.30 * df["climate_variability"]
        + 0.02 * df["transport_time_hrs"]
        + 0.85 * df["sensitivity_temperature"]
        + 0.65 * df["incidences_proxy"]
    )


def add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["incidence_probability_model"] = estimate_incidence_probability(out).clip(0.01, 0.99)
    out["risk_score"] = 100 * out["incidence_probability_model"]
    out["expected_loss_proxy"] = (
        out["incidence_probability_model"] * out["value_usd"] * out["transport_time_hrs"]
    )
    return out


def summarize_route_risk(df: pd.DataFrame) -> pd.DataFrame:
    risk_df = add_risk_columns(df)
    summary = (
        risk_df.groupby("ruta_logistica", as_index=False)
        .agg(
            volumen_total=("volume_tons", "sum"),
            valor_total=("value_usd", "sum"),
            riesgo_promedio=("risk_score", "mean"),
            perdida_esperada=("expected_loss_proxy", "sum"),
        )
        .sort_values("perdida_esperada", ascending=False)
    )
    return summary

