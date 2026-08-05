from __future__ import annotations

import pandas as pd


def _pitch(row: pd.Series) -> str:
    high_score = row["score"] >= 70
    high_risk = row.get("riesgo_logistico", 0) >= 0.55
    high_value = row.get("valor", 0) >= row.get("valor_p75", row.get("valor", 0))

    if high_score and high_risk and high_value:
        return "Oferta premium: monitoreo 24/7 con alertas y reduccion de merma en rutas criticas."
    if high_score and high_risk:
        return "Oferta de mitigacion: visibilidad de cadena de frio para reducir incidencias y reclamos."
    if high_score:
        return "Oferta de crecimiento: implementar monitoreo para escalar volumen sin elevar riesgo operativo."
    return "Oferta base: piloto de bajo costo para validar ROI en una ruta principal."


def generate_leads(ranking_df: pd.DataFrame, cluster_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    merged = ranking_df.merge(
        cluster_df[["company_id", "cluster", "riesgo_logistico"]],
        on="company_id",
        how="left",
    )
    merged = merged.rename(
        columns={
            "empresa": "empresa",
            "segmento": "segmento",
            "volume_tons": "volumen",
            "value_usd": "valor",
            "ruta_principal": "ruta_principal",
            "opportunity_score": "score",
            "drivers_score": "drivers_score",
        }
    )
    merged["valor_p75"] = merged["valor"].quantile(0.75)
    merged["pitch_comercial_sugerido"] = merged.apply(_pitch, axis=1)

    leads = merged.sort_values(["score", "riesgo_logistico"], ascending=[False, False]).head(top_n)
    return leads[
        [
            "empresa",
            "segmento",
            "volumen",
            "valor",
            "ruta_principal",
            "score",
            "cluster",
            "drivers_score",
            "pitch_comercial_sugerido",
        ]
    ].reset_index(drop=True)

