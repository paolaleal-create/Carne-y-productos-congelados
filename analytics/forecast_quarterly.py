from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


SEGMENT_CODE_MAP = {
    "bovino": ["0201", "0202"],
    "porcino": ["0203"],
}


def _parse_quarterly_exports(exports_path: Path) -> pd.DataFrame:
    df = pd.read_csv(exports_path)
    df = df.rename(
        columns={
            "Código": "codigo",
            "Descripcion del producto": "producto",
            "Descripción del producto": "producto",
        }
    )
    df["codigo"] = df["codigo"].astype(str).str.replace("'", "", regex=False).str.strip().str.zfill(4)
    quarter_cols = [c for c in df.columns if str(c).startswith("Valor exportado en ")]
    long_df = df.melt(
        id_vars=["codigo"],
        value_vars=quarter_cols,
        var_name="periodo",
        value_name="valor_exportado",
    )
    q = long_df["periodo"].str.extract(r"(\d{4})-T([1-4])")
    long_df["year"] = pd.to_numeric(q[0], errors="coerce")
    long_df["q"] = pd.to_numeric(q[1], errors="coerce")
    long_df = long_df.dropna(subset=["year", "q"]).copy()
    long_df["fecha"] = pd.PeriodIndex.from_fields(
        year=long_df["year"].astype(int),
        quarter=long_df["q"].astype(int),
        freq="Q",
    ).to_timestamp(how="start")
    long_df["valor_exportado"] = pd.to_numeric(long_df["valor_exportado"], errors="coerce").fillna(0.0)
    return long_df[["codigo", "fecha", "valor_exportado"]]


def _load_fx_exog(fx_path: Path) -> pd.DataFrame:
    fx = pd.read_csv(fx_path)
    fx["fecha"] = pd.to_datetime(fx["fecha"], errors="coerce")
    fx = fx.dropna(subset=["fecha"]).set_index("fecha").sort_index()
    fx = fx.asfreq("QS")
    fx["tipo_cambio_lag1"] = pd.to_numeric(fx["tipo_cambio_promedio_trimestral"], errors="coerce").shift(1)
    return fx[["tipo_cambio_lag1"]]


def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    den = y_true.replace(0, np.nan)
    out = (np.abs((y_true - y_pred) / den)).dropna()
    if out.empty:
        return np.nan
    return float(out.mean() * 100.0)


def _evaluate_arimax(
    y: pd.Series, x: pd.DataFrame, order: tuple[int, int, int], eval_h: int
) -> tuple[float, float, float] | None:
    if len(y) <= eval_h + 4:
        return None
    y_train, y_test = y.iloc[:-eval_h], y.iloc[-eval_h:]
    x_train, x_test = x.iloc[:-eval_h], x.iloc[-eval_h:]
    try:
        fit = SARIMAX(
            y_train,
            exog=x_train,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        fc = fit.get_forecast(steps=eval_h, exog=x_test).predicted_mean.reindex(y_test.index)
        mae = float(mean_absolute_error(y_test, fc))
        rmse = float(np.sqrt(mean_squared_error(y_test, fc)))
        mape = _safe_mape(y_test, fc)
        return mape, rmse, mae
    except Exception:
        return None


def _pick_best_order(y: pd.Series, x: pd.DataFrame, eval_h: int, candidates: list[tuple[int, int, int]] | None = None) -> dict:
    # Incluye el orden usado en tu notebook y una malla corta alrededor.
    if candidates is None:
        candidates = [
            (3, 2, 1),
            (2, 1, 1),
            (1, 1, 1),
            (1, 1, 0),
            (2, 2, 1),
            (3, 1, 1),
            (2, 1, 0),
        ]
    rows: list[dict] = []
    for order in candidates:
        metrics = _evaluate_arimax(y, x, order, eval_h=eval_h)
        if metrics is None:
            continue
        mape, rmse, mae = metrics
        rows.append({"order": order, "mape": mape, "rmse": rmse, "mae": mae})

    if not rows:
        fallback = (3, 2, 1)
        return {"order": fallback, "mape": np.nan, "rmse": np.nan, "mae": np.nan}

    eval_df = pd.DataFrame(rows)
    eval_df["rank_mape"] = eval_df["mape"].rank(method="min", na_option="bottom")
    eval_df["rank_rmse"] = eval_df["rmse"].rank(method="min", na_option="bottom")
    eval_df["rank_mae"] = eval_df["mae"].rank(method="min", na_option="bottom")
    eval_df["rank_avg"] = eval_df[["rank_mape", "rank_rmse", "rank_mae"]].mean(axis=1)
    best = eval_df.sort_values(["rank_avg", "mape", "rmse", "mae"], ascending=True).iloc[0]
    return {
        "order": tuple(int(x) for x in best["order"]),
        "mape": float(best["mape"]) if pd.notna(best["mape"]) else np.nan,
        "rmse": float(best["rmse"]) if pd.notna(best["rmse"]) else np.nan,
        "mae": float(best["mae"]) if pd.notna(best["mae"]) else np.nan,
    }


def _forecast_segment(
    exports_long: pd.DataFrame,
    fx_lagged: pd.DataFrame,
    codes: list[str],
    horizon_q: int,
    forced_order: tuple[int, int, int] | None = None,
    custom_candidates: list[tuple[int, int, int]] | None = None,
) -> dict:
    seg = exports_long[exports_long["codigo"].isin(codes)].copy()
    y = seg.groupby("fecha", as_index=True)["valor_exportado"].sum().sort_index().asfreq("QS")
    x = fx_lagged.reindex(y.index)
    mask = x.notna().all(axis=1)
    y = y[mask].dropna()
    x = x[mask]

    eval_h = min(8, max(4, horizon_q))
    if forced_order is not None:
        order = forced_order
        forced_metrics = _evaluate_arimax(y, x, order, eval_h=eval_h)
        best = {
            "order": order,
            "mape": np.nan if forced_metrics is None else forced_metrics[0],
            "rmse": np.nan if forced_metrics is None else forced_metrics[1],
            "mae": np.nan if forced_metrics is None else forced_metrics[2],
        }
    else:
        best = _pick_best_order(y, x, eval_h=eval_h, candidates=custom_candidates)
        order = best["order"]

    fit = SARIMAX(
        y,
        exog=x,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    coef_rows = []
    for name in fit.params.index:
        coef_rows.append(
            {
                "term": str(name),
                "coef": float(fit.params.get(name, np.nan)),
                "std_err": float(fit.bse.get(name, np.nan)),
                "p_value": float(fit.pvalues.get(name, np.nan)),
            }
        )

    # Como en tu notebook: proxys de exógena futura con últimos trimestres observados.
    x_future = x.iloc[-horizon_q:].copy()
    if len(x_future) < horizon_q:
        last_val = float(x.iloc[-1, 0]) if not x.empty else 0.0
        x_future = pd.DataFrame({"tipo_cambio_lag1": [last_val] * horizon_q})
    fc_obj = fit.get_forecast(steps=horizon_q, exog=x_future)
    fc_mean = fc_obj.predicted_mean
    fc_ci = fc_obj.conf_int()

    last_date = y.index[-1]
    future_idx = pd.date_range(last_date + pd.offsets.QuarterBegin(startingMonth=1), periods=horizon_q, freq="QS")
    fc_mean.index = future_idx
    fc_ci.index = future_idx

    hist = pd.DataFrame({"date": y.index, "value": y.values})
    out_fc = pd.DataFrame(
        {
            "date": future_idx,
            "forecast": fc_mean.values,
            "lower": fc_ci.iloc[:, 0].values,
            "upper": fc_ci.iloc[:, 1].values,
        }
    )

    return {
        "history": hist.to_dict("records"),
        "forecast": out_fc.to_dict("records"),
        "metrics": {
            "mape": None if pd.isna(best["mape"]) else float(best["mape"]),
            "rmse": None if pd.isna(best["rmse"]) else float(best["rmse"]),
            "mae": None if pd.isna(best["mae"]) else float(best["mae"]),
            "order": [int(order[0]), int(order[1]), int(order[2])],
        },
        "model_detail": {
            "order": [int(order[0]), int(order[1]), int(order[2])],
            "aic": float(fit.aic) if pd.notna(fit.aic) else None,
            "bic": float(fit.bic) if pd.notna(fit.bic) else None,
            "hqic": float(fit.hqic) if pd.notna(fit.hqic) else None,
            "llf": float(fit.llf) if pd.notna(fit.llf) else None,
            "nobs": int(fit.nobs),
            "coefficients": coef_rows,
        },
    }


def forecast_arimax_quarterly(exports_path: Path, fx_path: Path, horizon_q: int = 8) -> dict:
    exports_long = _parse_quarterly_exports(exports_path)
    fx_lagged = _load_fx_exog(fx_path)

    bovino = _forecast_segment(
        exports_long,
        fx_lagged,
        SEGMENT_CODE_MAP["bovino"],
        horizon_q=horizon_q,
        forced_order=(3, 2, 1),
    )
    # Para porcino, excluir MA(q) para evitar coeficientes MA no significativos.
    porcino_candidates = [
        (3, 2, 0),
        (2, 2, 0),
        (2, 1, 0),
        (1, 1, 0),
        (3, 1, 0),
        (1, 0, 0),
    ]
    porcino = _forecast_segment(
        exports_long,
        fx_lagged,
        SEGMENT_CODE_MAP["porcino"],
        horizon_q=horizon_q,
        custom_candidates=porcino_candidates,
    )

    return {"bovino": bovino, "porcino": porcino}
