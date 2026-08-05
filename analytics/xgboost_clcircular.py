from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


MODEL_FEATURES = [
    "participacion_mercado",
    "ventas_anuales_musd",
    "frecuencia_exportacion_dias",
    "presencia_internacional",
    "reporte_esg",
    "movimiento_cadena_frio",
    "volumen_exportaciones_musd",
    "mas_40_viajes_mensuales",
    "cluster_label",
]


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        out[col] = out[col].fillna(out[col].median() if out[col].notna().any() else 0.0)
    return out


def _assign_cluster_if_missing(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    out = df.copy()
    if "cluster_label" in out.columns and out["cluster_label"].notna().any():
        out["cluster_label"] = pd.to_numeric(out["cluster_label"], errors="coerce").fillna(0).astype(int)
        return out

    k = min(4, max(2, len(out)))
    feat_cols = [
        "participacion_mercado",
        "ventas_anuales_musd",
        "frecuencia_exportacion_dias",
        "presencia_internacional",
        "reporte_esg",
        "movimiento_cadena_frio",
        "volumen_exportaciones_musd",
        "mas_40_viajes_mensuales",
    ]
    feat_cols = [c for c in feat_cols if c in out.columns]
    if not feat_cols:
        out["cluster_label"] = 0
        return out

    X = out[feat_cols].values
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    out["cluster_label"] = model.fit_predict(X).astype(int)
    return out


def _sample_truncated_normal(
    mean: float,
    std: float,
    lower: float,
    upper: float,
    size: int,
    random_state: np.random.Generator,
) -> np.ndarray:
    if std <= 0:
        return np.repeat(mean, size)
    # Aproximacion estable sin dependencia externa: normal + recorte a limites.
    samples = random_state.normal(loc=mean, scale=std, size=size)
    return np.clip(samples, lower, upper)


def _generate_simulated_dataset(
    df_real: pd.DataFrame,
    target_n: int,
    n_iter: int,
    seed: int = 42,
) -> pd.DataFrame:
    if len(df_real) >= target_n:
        return pd.DataFrame(columns=df_real.columns)

    rng = np.random.default_rng(seed)
    cluster_col = "cluster_label"
    share_col = "participacion_mercado"
    n_sim = target_n - len(df_real)

    numeric_cols = df_real.select_dtypes(include=[np.number]).columns.tolist()
    binary_cols = [
        c
        for c in numeric_cols
        if c != share_col and set(pd.Series(df_real[c]).dropna().unique().tolist()).issubset({0, 1})
    ]
    numeric_non_binary_cols = [c for c in numeric_cols if c not in binary_cols and c != share_col]

    cluster_stats: dict[int, dict[str, dict[str, float]]] = {}
    for cl, g in df_real.groupby(cluster_col):
        stats_cl: dict[str, dict[str, float]] = {}
        n_cl = len(g)
        for col in numeric_non_binary_cols + [share_col]:
            mean_ = float(g[col].mean())
            std_ = float(g[col].std(ddof=1))
            if np.isnan(std_) or std_ == 0:
                std_ = max(abs(mean_) * 0.05, 1e-6)
            sem_ = std_ / max(np.sqrt(n_cl), 1.0)
            ci_half_95 = 1.96 * sem_
            stats_cl[col] = {
                "mean": mean_,
                "std": std_,
                "allowed_half_range": 1.5 * ci_half_95,
            }
        for col in binary_cols:
            stats_cl[col] = {"p": float(np.clip(g[col].mean(), 0, 1))}
        cluster_stats[int(cl)] = stats_cl

    cluster_counts = df_real[cluster_col].value_counts().sort_index()
    cluster_probs = (cluster_counts / cluster_counts.sum()).values
    cluster_ids = cluster_counts.index.tolist()
    sim_counts = rng.multinomial(n_sim, cluster_probs)
    sim_count_map = dict(zip(cluster_ids, sim_counts))

    real_share_sum = float(df_real[share_col].sum())
    remaining_share = max(0.0, 100.0 - real_share_sum)

    def build_one(seed_i: int) -> pd.DataFrame:
        local_rng = np.random.default_rng(seed_i)
        sim_rows: list[dict] = []
        for cl, cnt in sim_count_map.items():
            if cnt <= 0:
                continue
            for i in range(cnt):
                row: dict[str, object] = {cluster_col: int(cl)}
                row["empresa"] = f"Empresa_sim_{int(cl)}_{i + 1:03d}"
                for col in numeric_non_binary_cols:
                    mean_ = cluster_stats[int(cl)][col]["mean"]
                    std_ = cluster_stats[int(cl)][col]["std"]
                    half_range = cluster_stats[int(cl)][col]["allowed_half_range"]
                    val = _sample_truncated_normal(
                        mean=mean_,
                        std=std_,
                        lower=mean_ - half_range,
                        upper=mean_ + half_range,
                        size=1,
                        random_state=local_rng,
                    )[0]
                    row[col] = float(max(val, 0.0))
                for col in binary_cols:
                    p = cluster_stats[int(cl)][col]["p"]
                    row[col] = int(local_rng.binomial(1, p))
                sim_rows.append(row)
        sim_df = pd.DataFrame(sim_rows)
        for col in df_real.columns:
            if col not in sim_df.columns:
                sim_df[col] = np.nan
        sim_df = sim_df[df_real.columns]
        if len(sim_df):
            weights = np.abs(local_rng.normal(1.0, 0.5, len(sim_df)))
            weights = np.where(weights <= 0, 1e-6, weights)
            sim_df[share_col] = remaining_share * (weights / weights.sum())
        return sim_df

    def score_scenario(sim_df: pd.DataFrame) -> float:
        if sim_df.empty:
            return np.inf
        full = pd.concat([df_real, sim_df], ignore_index=True)
        score = 0.0
        for cl, real_g in df_real.groupby(cluster_col):
            full_g = full[full[cluster_col] == cl]
            for col in numeric_non_binary_cols + [share_col]:
                m_real = float(real_g[col].mean())
                m_full = float(full_g[col].mean()) if len(full_g) else 0.0
                score += abs(m_full - m_real) / max(abs(m_real), 1e-6)
            for col in binary_cols:
                p_real = float(real_g[col].mean())
                p_full = float(full_g[col].mean()) if len(full_g) else 0.0
                score += abs(p_full - p_real)
        score += abs(float(full[share_col].sum()) - 100.0) * 1000.0
        return score

    best_score = np.inf
    best_df = pd.DataFrame(columns=df_real.columns)
    for i in range(max(1, n_iter)):
        sim_df = build_one(seed + 1000 + i)
        sc = score_scenario(sim_df)
        if sc < best_score:
            best_score = sc
            best_df = sim_df.copy()
    return best_df


def _simulate_closure_prob(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(random_state)

    z_cols = [
        "participacion_mercado",
        "ventas_anuales_musd",
        "frecuencia_exportacion_dias",
        "volumen_exportaciones_musd",
    ]
    for col in z_cols:
        mean_ = float(out[col].mean())
        std_ = float(out[col].std())
        out[f"{col}_z"] = 0.0 if std_ == 0 else (out[col] - mean_) / std_

    score = (
        0.4 * out["ventas_anuales_musd_z"]
        + 1.1 * out["volumen_exportaciones_musd_z"]
        + 0.4 * out["participacion_mercado_z"]
        - 0.7 * out["frecuencia_exportacion_dias_z"]
        + 0.9 * out["presencia_internacional"]
        + 0.7 * out["movimiento_cadena_frio"]
        + 0.4 * out["reporte_esg"]
        + 0.5 * out["mas_40_viajes_mensuales"]
    )

    cluster_effect = {0: -0.4, 1: 0.7, 2: 0.5, 3: 0.4}
    score = score + out["cluster_label"].map(cluster_effect).fillna(0.0) + rng.normal(0, 0.8, len(out))

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    target_rate = 0.17
    intercepts = np.linspace(-3, 3, 200)
    best_b = 0.0
    best_diff = np.inf
    for b in intercepts:
        p = sigmoid(score + b)
        diff = abs(float(p.mean()) - target_rate)
        if diff < best_diff:
            best_diff = diff
            best_b = float(b)
    p_final = sigmoid(score + best_b)

    out["prob_cierre_simulada"] = p_final
    out["Cierre"] = rng.binomial(1, p_final).astype(int)
    return out


def _train_probability_model(df: pd.DataFrame, random_state: int = 42) -> tuple[np.ndarray, str, dict, dict]:
    out = df.copy()
    X = out[MODEL_FEATURES]
    y = out["Cierre"].astype(int)

    if y.nunique() < 2:
        prob = out["prob_cierre_simulada"].to_numpy()
        empty_fi = {k: None for k in MODEL_FEATURES}
        return prob, "baseline_simulado", {"accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}, empty_fi

    stratify = y if y.value_counts().min() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
        model_name = "xgboost"
    else:
        model = GradientBoostingClassifier(random_state=random_state)
        model_name = "gradient_boosting_fallback"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_test.nunique() > 1 else None,
    }
    if hasattr(model, "feature_importances_"):
        fi = {
            feature: float(importance)
            for feature, importance in zip(MODEL_FEATURES, np.asarray(model.feature_importances_).tolist())
        }
    else:
        fi = {k: None for k in MODEL_FEATURES}
    full_prob = model.predict_proba(X)[:, 1]
    return full_prob, model_name, metrics, fi


def generate_close_probability_table(
    companies_csv: Path | str,
    target_n: int = 60,
    n_iter: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    path = Path(companies_csv)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_csv(path)
    if "empresa" not in df.columns:
        raise ValueError("El dataset debe incluir la columna 'empresa'.")

    numeric_needed = [
        "participacion_mercado",
        "ventas_anuales_musd",
        "frecuencia_exportacion_dias",
        "presencia_internacional",
        "reporte_esg",
        "movimiento_cadena_frio",
        "volumen_exportaciones_musd",
        "mas_40_viajes_mensuales",
    ]
    df = _safe_numeric(df, numeric_needed + ["cluster_label"])
    df = _assign_cluster_if_missing(df, random_state=random_state)
    df["origen_dato"] = "real"

    sim_df = _generate_simulated_dataset(df, target_n=target_n, n_iter=n_iter, seed=random_state)
    if len(sim_df):
        sim_df["origen_dato"] = "simulado"
        final = pd.concat([df, sim_df], ignore_index=True)
    else:
        final = df.copy()

    modeled = _simulate_closure_prob(final, random_state=random_state)
    full_prob, model_name, metrics, feature_importance = _train_probability_model(modeled, random_state=random_state)
    modeled["probabilidad_cierre_modelo"] = full_prob

    for col in MODEL_FEATURES:
        if col in modeled.columns:
            modeled[col] = pd.to_numeric(modeled[col], errors="coerce")

    table_cols = ["empresa"] + MODEL_FEATURES + ["origen_dato", "probabilidad_cierre_modelo"]
    result = modeled[table_cols].copy()
    # Asegurar etiquetas de cluster discretas para visualizacion/segmentacion.
    result["cluster_label"] = pd.to_numeric(result["cluster_label"], errors="coerce").fillna(0).round().astype(int)
    result["cluster_label"] = result["cluster_label"].clip(lower=0, upper=3)
    result = result.sort_values("probabilidad_cierre_modelo", ascending=False).reset_index(drop=True)

    meta = {
        "model_name": model_name,
        "target_n": int(target_n),
        "n_real": int((result["origen_dato"] == "real").sum()),
        "n_simulado": int((result["origen_dato"] == "simulado").sum()),
        "metrics": metrics,
        "feature_importance": feature_importance,
    }
    return result, meta
