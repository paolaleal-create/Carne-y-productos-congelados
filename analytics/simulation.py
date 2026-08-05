from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.risk_model import add_risk_columns


def expected_loss(df: pd.DataFrame, prob_col: str = "incidence_probability_model") -> pd.Series:
    return df[prob_col] * df["value_usd"] * df["transport_time_hrs"]


def run_scenarios(
    base_df: pd.DataFrame,
    adoption_rate: float = 0.3,
    adoption_effectiveness: float = 0.35,
    risk_multiplier: float = 1.15,
    expansion_growth: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = add_risk_columns(base_df).copy()
    base["expected_loss"] = expected_loss(base)

    n_adopted = int(len(base) * adoption_rate)
    adopted_idx = base.sort_values("value_usd", ascending=False).head(n_adopted).index

    s1 = base.copy()
    s1["prob_adj"] = s1["incidence_probability_model"]
    s1.loc[adopted_idx, "prob_adj"] = (
        s1.loc[adopted_idx, "incidence_probability_model"] * (1 - adoption_effectiveness)
    ).clip(0.01, 0.99)
    s1["expected_loss"] = expected_loss(s1, prob_col="prob_adj")

    s2 = base.copy()
    s2["prob_adj"] = (s2["incidence_probability_model"] * risk_multiplier).clip(0.01, 0.99)
    s2["expected_loss"] = expected_loss(s2, prob_col="prob_adj")

    s3 = base.copy()
    s3["volume_tons"] = s3["volume_tons"] * (1 + expansion_growth)
    s3["value_usd"] = s3["value_usd"] * (1 + expansion_growth)
    s3["prob_adj"] = (s3["incidence_probability_model"] * 1.05).clip(0.01, 0.99)
    s3["expected_loss"] = expected_loss(s3, prob_col="prob_adj")

    scenarios = {
        "Base": base,
        "Escenario 1: Adopcion": s1,
        "Escenario 2: Riesgo": s2,
        "Escenario 3: Expansion": s3,
    }

    summary = []
    for name, df in scenarios.items():
        total_value = df["value_usd"].sum()
        total_loss = df["expected_loss"].sum()
        summary.append(
            {
                "escenario": name,
                "valor_total_usd": total_value,
                "perdida_esperada_usd": total_loss,
                "loss_rate": (total_loss / total_value) if total_value > 0 else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary)
    return summary_df, scenarios

