from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def prepare_monthly_series(
    monthly_df: pd.DataFrame,
    target_col: str,
    segment: str | None = None,
    company_id: int | None = None,
) -> pd.Series:
    df = monthly_df.copy()
    if segment and segment != "todos":
        df = df[df["segmento"] == segment]
    if company_id:
        df = df[df["company_id"] == company_id]

    series = df.groupby("month")[target_col].sum().sort_index()
    series.index = pd.DatetimeIndex(series.index, freq="MS")
    return series


def _metrics(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    valid = actual != 0
    mape = float((np.abs((actual[valid] - pred[valid]) / actual[valid]).mean() * 100)) if valid.any() else np.nan
    rmse = float(np.sqrt(((actual - pred) ** 2).mean()))
    return mape, rmse


def forecast_series(series: pd.Series, model_name: str, horizon: int) -> dict:
    if len(series) < 18:
        raise ValueError("Se requieren al menos 18 observaciones mensuales para pronosticar.")

    train = series.iloc[:-horizon] if len(series) > (horizon + 12) else series
    test = series.iloc[-horizon:] if len(series) > (horizon + 12) else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if model_name == "ARIMA":
            model = ARIMA(train, order=(1, 1, 1)).fit()
            fc = model.get_forecast(steps=horizon)
            pred = fc.predicted_mean
            ci = fc.conf_int(alpha=0.2)
            lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
        elif model_name == "SARIMA":
            model = SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            fc = model.get_forecast(steps=horizon)
            pred = fc.predicted_mean
            ci = fc.conf_int(alpha=0.2)
            lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
        else:
            try:
                model = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12,
                    initialization_method="estimated",
                ).fit()
            except Exception:
                model = ExponentialSmoothing(train, trend="add", initialization_method="estimated").fit()
            pred = model.forecast(horizon)
            residual_std = (train - model.fittedvalues).std()
            lower = pred - (1.28 * residual_std)
            upper = pred + (1.28 * residual_std)

    forecast_df = pd.DataFrame(
        {
            "month": pred.index,
            "forecast": pred.values,
            "lower": lower.values,
            "upper": upper.values,
        }
    )

    mape, rmse = (np.nan, np.nan)
    if test is not None:
        mape, rmse = _metrics(test, pred)

    return {
        "history": train,
        "test": test,
        "forecast_df": forecast_df,
        "mape": mape,
        "rmse": rmse,
    }

