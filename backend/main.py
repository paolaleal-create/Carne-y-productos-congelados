from __future__ import annotations

import subprocess
import sys
import unicodedata
from glob import glob
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analytics.allocation import generate_leads
from analytics.clustering import build_feature_table, clustering_diagnostics, fit_clusters
from analytics.forecast_quarterly import forecast_arimax_quarterly
from analytics.forecasting import forecast_series, prepare_monthly_series
from analytics.scoring import DEFAULT_WEIGHTS, rank_companies
from analytics.simulation import run_scenarios
from analytics.xgboost_clcircular import generate_close_probability_table

DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"
EXPORTS_XLSX = DATA_DIR / "EXPORTS MEXICO ANUAL LIMPIO.xlsx"
PRICE_TABLE_CSV = DATA_DIR / "PrecioPromedioCarne.csv"
COMPANY_MARKET_SHARE_CSV = DATA_DIR / "dataset_empresas_actualizado.csv"
EMPRESAS_CON_CLUSTERS_CSV = DATA_DIR / "empresas_con_clusters.csv"
RISK_FEATURES_CSV = DATA_DIR / "risk_features.csv"
RISK_FEATURES_ENRICHED_CSV = DATA_DIR / "risk_features_enriched.csv"
ROUTES_2_CSV = DATA_DIR / "rutas (2).csv"
DESTINOS_RUTAS_TOP20_CSV = DATA_DIR / "destinos_rutas_top20.csv"
EXPORTS_TRIMESTRALES_CSV = DATA_DIR / "EXPORTS-MEXICO-TRIMESTRALES-2013-2024.csv"
FX_TRIMESTRAL_CSV = DATA_DIR / "tipo_cambio_promedio_trimestral.csv"
DF_FINAL_CSV = DATA_DIR / "df_final.csv"
NOTEBOOK_FEATURE_IMPORTANCE = {
    "reporte_esg": 0.372,
    "cluster_label": 0.220,
    "frecuencia_exportacion_dias": 0.149,
    "mas_40_viajes_mensuales": 0.110,
    "ventas_anuales_musd": 0.093,
    "volumen_exportaciones_musd": 0.033,
    "participacion_mercado": 0.026,
    "presencia_internacional": 0.000,
    "movimiento_cadena_frio": 0.000,
}
NOTEBOOK_CLUSTER_PROBABILITIES = [
    {"cluster": 2, "probabilidad_media_cierre": 0.949663},
    {"cluster": 1, "probabilidad_media_cierre": 0.850098},
    {"cluster": 3, "probabilidad_media_cierre": 0.596381},
    {"cluster": 0, "probabilidad_media_cierre": 0.045534},
]

# Supuestos de conversion para estimar volumen y envios a partir de valor exportado.
# precio_usd_por_ton: valor promedio por tonelada por codigo arancelario.
# tons_per_shipment: capacidad promedio movida por envio.
ESTIMATION_ASSUMPTIONS = {
    "0201": {"precio_usd_por_ton": 6200.0, "tons_per_shipment": 18.0},
    "0202": {"precio_usd_por_ton": 5400.0, "tons_per_shipment": 20.0},
    "0203": {"precio_usd_por_ton": 3600.0, "tons_per_shipment": 21.0},
    "default": {"precio_usd_por_ton": 5000.0, "tons_per_shipment": 20.0},
}
EXPORT_VALUE_SCALE = 1000.0

app = FastAPI(title="CL Circular Analytics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


def ensure_mock_data() -> None:
    required = [DATA_DIR / "companies.csv", DATA_DIR / "flows.csv", DATA_DIR / "trade_monthly.csv"]
    if all(path.exists() for path in required):
        return
    subprocess.run([sys.executable, str(DATA_DIR / "generate_mock_data.py")], check=True, cwd=str(ROOT))


def _find_routes_file() -> Path | None:
    candidates = sorted(glob(str(DATA_DIR / "rutas*.csv")))
    if not candidates:
        return None
    return Path(candidates[0])


def _pick_id_column(df: pd.DataFrame) -> str | None:
    for col in ["ID", "id", "route_id", "Route_ID", "routeId"]:
        if col in df.columns:
            return col
    return None


def _normalize_id_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def _normalize_text(value: str) -> str:
    txt = str(value).strip().lower()
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    return txt


def update_risk_features_with_index() -> pd.DataFrame:
    if not RISK_FEATURES_CSV.exists():
        return pd.DataFrame()

    rf = pd.read_csv(RISK_FEATURES_CSV)
    risk_id_col = _pick_id_column(rf)
    if risk_id_col is None:
        return rf

    required = {"loss_usd", "severity", "border_congestion", "climate_variability"}
    if not required.issubset(set(rf.columns)):
        return rf

    # Limpiar columnas derivadas previas para evitar conflictos en recálculos.
    for col in ["loss_score", "severity_score", "border_score", "climate_score", "risk_index_event", "risk_index"]:
        if col in rf.columns:
            rf = rf.drop(columns=[col])

    rf["route_num"] = _normalize_id_series(rf[risk_id_col])
    rf["loss_usd"] = pd.to_numeric(rf["loss_usd"], errors="coerce").fillna(0.0)
    rf["severity"] = pd.to_numeric(rf["severity"], errors="coerce").fillna(0.0)
    rf["border_congestion"] = pd.to_numeric(rf["border_congestion"], errors="coerce").fillna(0.0)
    rf["climate_variability"] = pd.to_numeric(rf["climate_variability"], errors="coerce").fillna(0.0)

    def minmax_0_100(series: pd.Series) -> pd.Series:
        min_v = float(series.min())
        max_v = float(series.max())
        if max_v == min_v:
            return pd.Series(0.0, index=series.index)
        return 100.0 * (series - min_v) / (max_v - min_v)

    rf["loss_score"] = minmax_0_100(rf["loss_usd"])
    rf["severity_score"] = minmax_0_100(rf["severity"])
    rf["border_score"] = minmax_0_100(rf["border_congestion"])
    rf["climate_score"] = minmax_0_100(rf["climate_variability"])

    # Risk index ponderado por evento:
    # 0.4*loss_score + 0.3*severity_score + 0.15*border_score + 0.15*climate_score
    rf["risk_index_event"] = (
        0.40 * rf["loss_score"]
        + 0.30 * rf["severity_score"]
        + 0.15 * rf["border_score"]
        + 0.15 * rf["climate_score"]
    )

    # Risk score promedio por ruta (valor final en columna risk_index).
    route_avg = rf.groupby("route_num", as_index=False)["risk_index_event"].mean().rename(columns={"risk_index_event": "risk_index"})
    out = rf.merge(route_avg, on="route_num", how="left")
    out["risk_index"] = out["risk_index"].fillna(0.0).round(2)
    out["risk_index_event"] = out["risk_index_event"].round(2)
    out = out.drop(columns=["route_num"])

    # Persistir nueva columna en el mismo archivo para uso posterior.
    # Si esta bloqueado por otro proceso, escribir a un archivo enriquecido alterno.
    try:
        out.to_csv(RISK_FEATURES_CSV, index=False)
    except PermissionError:
        out.to_csv(RISK_FEATURES_ENRICHED_CSV, index=False)
    return out


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_mock_data()
    update_risk_features_with_index()
    companies = pd.read_csv(DATA_DIR / "companies.csv")
    flows = pd.read_csv(DATA_DIR / "flows.csv")
    monthly = pd.read_csv(DATA_DIR / "trade_monthly.csv", parse_dates=["month"])
    exports_long = load_exports_long()
    state_exports = load_state_exports()
    price_table = load_price_table()
    return companies, flows, monthly, exports_long, state_exports, price_table


def load_exports_long() -> pd.DataFrame:
    if not EXPORTS_XLSX.exists():
        return pd.DataFrame(columns=["codigo", "producto", "year", "valor_exportado"])

    exports = pd.read_excel(EXPORTS_XLSX)
    year_cols = [c for c in exports.columns if str(c).startswith("Valor exportado en ")]
    if not year_cols:
        return pd.DataFrame(columns=["codigo", "producto", "year", "valor_exportado"])

    exports = exports.rename(
        columns={
            "Código": "codigo",
            "Descripción del producto": "producto",
        }
    )
    exports["codigo"] = exports["codigo"].astype(str).str.replace("'", "", regex=False).str.strip()
    exports["producto"] = exports["producto"].astype(str).str.strip()

    long_df = exports.melt(
        id_vars=["codigo", "producto"],
        value_vars=year_cols,
        var_name="periodo",
        value_name="valor_exportado",
    )
    long_df["year"] = long_df["periodo"].astype(str).str.extract(r"(\d{4})").astype(int)
    long_df["valor_exportado"] = pd.to_numeric(long_df["valor_exportado"], errors="coerce").fillna(0.0)
    return long_df[["codigo", "producto", "year", "valor_exportado"]]


def load_state_exports() -> pd.DataFrame:
    file_map = {
        "bovino": [
            DATA_DIR / "Exportaciones_entidad_federativa_bovino_congelada.csv",
            DATA_DIR / "Exportaciones_entidad_federativa_bovino_fresca_refrigerada.csv",
        ],
        "porcino": [
            DATA_DIR / "Exportaciones_entidad_federativa_puerco.csv",
        ],
    }

    frames: list[pd.DataFrame] = []
    for segment, paths in file_map.items():
        for path in paths:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df.rename(columns={"State": "state", "Trade Value": "trade_value", "Share": "share"})
            needed = ["state", "trade_value", "share"]
            if not all(col in df.columns for col in needed):
                continue
            out = df[needed].copy()
            out["trade_value"] = pd.to_numeric(out["trade_value"], errors="coerce").fillna(0.0)
            out["share"] = pd.to_numeric(out["share"], errors="coerce").fillna(0.0)
            out["segment"] = segment
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["state", "trade_value", "share", "segment"])
    return pd.concat(frames, ignore_index=True)


def load_price_table() -> pd.DataFrame:
    if not PRICE_TABLE_CSV.exists():
        return pd.DataFrame(columns=["codigo", "precio_usd_por_ton"])

    df = pd.read_csv(PRICE_TABLE_CSV)
    required = {"Código HS", "Precio promedio USD/ton"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["codigo", "precio_usd_por_ton"])

    out = df[["Código HS", "Precio promedio USD/ton"]].copy()
    out["codigo"] = out["Código HS"].astype(str).str.replace("'", "", regex=False).str.strip().str.zfill(4)
    out["precio_usd_por_ton"] = (
        out["Precio promedio USD/ton"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    out["precio_usd_por_ton"] = pd.to_numeric(out["precio_usd_por_ton"], errors="coerce")
    out = out.dropna(subset=["precio_usd_por_ton"])
    return out[["codigo", "precio_usd_por_ton"]].drop_duplicates("codigo")


def load_company_market_share() -> pd.DataFrame:
    if not COMPANY_MARKET_SHARE_CSV.exists():
        return pd.DataFrame(columns=["empresa", "participacion_mercado"])

    df = pd.read_csv(COMPANY_MARKET_SHARE_CSV)
    required = {"empresa", "participacion_mercado"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["empresa", "participacion_mercado"])

    out = df[["empresa", "participacion_mercado"]].copy()
    out["participacion_mercado"] = pd.to_numeric(out["participacion_mercado"], errors="coerce").fillna(0.0)
    out = out[out["participacion_mercado"] > 0].sort_values("participacion_mercado", ascending=False)
    return out


def load_cluster_input_real() -> pd.DataFrame:
    if not COMPANY_MARKET_SHARE_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(COMPANY_MARKET_SHARE_CSV)
    if "empresa" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    numeric_cols = [
        "participacion_mercado",
        "ventas_anuales_musd",
        "frecuencia_exportacion_dias",
        "presencia_internacional",
        "reporte_esg",
        "movimiento_cadena_frio",
        "crecimiento_yoy",
        "volumen_exportaciones_musd",
        "datos_estimados",
        "viajes_mensuales",
        "mas_40_viajes_mensuales",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        out[col] = out[col].fillna(out[col].median() if out[col].notna().any() else 0.0)

    return out[["empresa"] + numeric_cols]


def load_route_tons_risk() -> pd.DataFrame:
    if not ROUTES_2_CSV.exists():
        return pd.DataFrame(columns=["route_id", "route_name", "average_yearly_tons", "risk_index"])

    rutas2 = pd.read_csv(ROUTES_2_CSV)
    needed = {"route_id", "route_name", "average_yearly_tons"}
    if not needed.issubset(set(rutas2.columns)):
        return pd.DataFrame(columns=["route_id", "route_name", "average_yearly_tons", "risk_index"])

    rf = update_risk_features_with_index()
    if rf.empty or "route_id" not in rf.columns or "risk_index" not in rf.columns:
        out = rutas2[["route_id", "route_name", "average_yearly_tons"]].copy()
        out["risk_index"] = 0.0
        return out

    risk_id_col = _pick_id_column(rf)
    if risk_id_col is None:
        out = rutas2[["route_id", "route_name", "average_yearly_tons"]].copy()
        out["risk_index"] = 0.0
        return out

    rf = rf.copy()
    rf["route_num"] = _normalize_id_series(rf[risk_id_col])
    route_risk = rf.groupby("route_num", as_index=False)["risk_index"].mean()
    out = rutas2[["route_id", "route_name", "average_yearly_tons"]].copy()
    out["average_yearly_tons"] = pd.to_numeric(out["average_yearly_tons"], errors="coerce").fillna(0.0)
    routes_id_col = _pick_id_column(rutas2)
    out["route_num"] = _normalize_id_series(rutas2[routes_id_col]) if routes_id_col else _normalize_id_series(out["route_id"])
    out = out.merge(route_risk, on="route_num", how="left")
    out["risk_index"] = out["risk_index"].fillna(0.0)
    return out[["route_id", "route_name", "average_yearly_tons", "risk_index"]]


def load_routes_company_usage() -> pd.DataFrame:
    if not ROUTES_2_CSV.exists() or not DESTINOS_RUTAS_TOP20_CSV.exists():
        return pd.DataFrame()

    routes = pd.read_csv(ROUTES_2_CSV)
    destinations = pd.read_csv(DESTINOS_RUTAS_TOP20_CSV)
    required_routes = {
        "route_id",
        "route_name",
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
        "average_yearly_tons",
    }
    required_dest = {"route_id", "company_name", "dest_state", "dest_city", "pct_export_volume"}
    if not required_routes.issubset(set(routes.columns)) or not required_dest.issubset(set(destinations.columns)):
        return pd.DataFrame()

    routes = routes.copy()
    destinations = destinations.copy()
    routes["route_num"] = _normalize_id_series(routes["route_id"])
    destinations["route_num"] = _normalize_id_series(destinations["route_id"])

    destinations["pct_export_volume"] = pd.to_numeric(destinations["pct_export_volume"], errors="coerce").fillna(0.0)
    destinations["dest_city"] = destinations["dest_city"].astype(str).str.strip()
    destinations["dest_state"] = destinations["dest_state"].astype(str).str.strip()
    destinations["company_name"] = destinations["company_name"].astype(str).str.strip()

    grouped = destinations.groupby("route_num", as_index=False).agg(
        companies=("company_name", lambda s: sorted(set(s.tolist()))),
        destinations=("dest_city", lambda s: sorted(set(s.tolist()))),
        states=("dest_state", lambda s: sorted(set(s.tolist()))),
        avg_pct_volume=("pct_export_volume", "mean"),
    )

    out = routes.merge(grouped, on="route_num", how="left")
    out["average_yearly_tons"] = pd.to_numeric(out["average_yearly_tons"], errors="coerce").fillna(0.0)
    out["origin_lat"] = pd.to_numeric(out["origin_lat"], errors="coerce")
    out["origin_lon"] = pd.to_numeric(out["origin_lon"], errors="coerce")
    out["dest_lat"] = pd.to_numeric(out["dest_lat"], errors="coerce")
    out["dest_lon"] = pd.to_numeric(out["dest_lon"], errors="coerce")
    out = out.dropna(subset=["origin_lat", "origin_lon", "dest_lat", "dest_lon"]).copy()

    out["companies"] = out["companies"].apply(lambda x: x if isinstance(x, list) else [])
    out["destinations"] = out["destinations"].apply(lambda x: x if isinstance(x, list) else [])
    out["states"] = out["states"].apply(lambda x: x if isinstance(x, list) else [])
    out["company_count"] = out["companies"].apply(len)
    out["destination_count"] = out["destinations"].apply(len)
    out["avg_pct_volume"] = pd.to_numeric(out["avg_pct_volume"], errors="coerce").fillna(0.0)

    return out[
        [
            "route_id",
            "route_name",
            "origin_lat",
            "origin_lon",
            "dest_lat",
            "dest_lon",
            "average_yearly_tons",
            "companies",
            "destinations",
            "states",
            "company_count",
            "destination_count",
            "avg_pct_volume",
        ]
    ].copy()


@lru_cache(maxsize=1)
def load_close_probability_model() -> tuple[pd.DataFrame, dict]:
    source_csv = EMPRESAS_CON_CLUSTERS_CSV if EMPRESAS_CON_CLUSTERS_CSV.exists() else COMPANY_MARKET_SHARE_CSV
    return generate_close_probability_table(
        companies_csv=source_csv,
        target_n=60,
        n_iter=100,
        random_state=42,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/filters")
def filters() -> dict:
    companies, flows, monthly, exports_long, _, _ = load_data()
    routes_usage = load_routes_company_usage()
    route_tons_risk_df = load_route_tons_risk()
    years_monthly = set(pd.to_datetime(monthly["month"]).dt.year.unique().tolist())
    years_exports = set(exports_long["year"].unique().tolist()) if not exports_long.empty else set()
    combined = years_monthly.union(years_exports)
    if combined:
        y_min = int(min(combined))
        y_max = int(max(combined))
        years = list(range(y_min, y_max + 1))
    else:
        years = []
    # Mostrar solo nombres de corredores/rutas del mapa (sin IDs tipo R1/R01).
    if not route_tons_risk_df.empty and "route_name" in route_tons_risk_df.columns:
        route_values = set(route_tons_risk_df["route_name"].dropna().astype(str).str.strip().tolist())
    elif not routes_usage.empty and "route_name" in routes_usage.columns:
        route_values = set(routes_usage["route_name"].dropna().astype(str).str.strip().tolist())
    else:
        route_values = set(flows["ruta_logistica"].dropna().astype(str).str.strip().tolist())

    return {
        "segments": sorted(flows["segmento"].unique().tolist()),
        "routes": sorted(route_values),
        "companies": companies[["company_id", "empresa"]].sort_values("empresa").to_dict("records"),
        "years": years,
    }


@app.get("/api/kpis")
def kpis(segment: str = "todos", route: str = "todas", year: str = "todos") -> dict:
    _, _, monthly, exports_long, state_exports, price_table = load_data()
    company_share_df = load_company_market_share()
    route_tons_risk_df = load_route_tons_risk()
    routes_usage = load_routes_company_usage()
    all_routes_tons_risk_df = route_tons_risk_df.copy()

    if not route_tons_risk_df.empty and route != "todas":
        route_clean = _normalize_text(route)
        route_match = route_tons_risk_df["route_name"].astype(str).map(_normalize_text) == route_clean
        route_tons_risk_df = route_tons_risk_df[route_match].copy()
    df = monthly.copy()
    if year != "todos":
        target_year = int(year)
        df = df[pd.to_datetime(df["month"]).dt.year == target_year]
    if segment != "todos":
        df = df[df["segmento"] == segment]
    if route != "todas":
        # Solo aplicar filtro de ruta al dataset mensual si el nombre existe en ese dataset.
        # Esto evita vaciar KPIs/mercado cuando la ruta seleccionada proviene del mapa real.
        route_vals_monthly = set(df["ruta_logistica"].astype(str).str.lower().unique().tolist())
        if str(route).strip().lower() in route_vals_monthly:
            df = df[df["ruta_logistica"] == route]

    segment_code_map = {
        "bovino": ["0201", "0202"],
        "porcino": ["0203"],
    }
    selected_codes = segment_code_map.get(segment)

    value_usd_from_exports = None
    estimated_volume_tons = None
    estimated_containers = None
    if not exports_long.empty:
        # Regla explicita:
        # - todos: usa todos los productos del archivo real.
        # - bovino/porcino: usa solo los codigos mapeados.
        exports_kpi_df = exports_long.copy()
        if year != "todos":
            exports_kpi_df = exports_kpi_df[exports_kpi_df["year"] == int(year)]
        if segment == "todos":
            exports_kpi_df = exports_kpi_df.copy()
        elif selected_codes is not None:
            exports_kpi_df = exports_kpi_df[exports_kpi_df["codigo"].isin(selected_codes)]
        value_usd_from_exports = float(exports_kpi_df["valor_exportado"].sum()) * EXPORT_VALUE_SCALE

        def _estimate_row(row: pd.Series) -> tuple[float, float]:
            code = str(row["codigo"])
            price = None
            if not price_table.empty:
                m = price_table[price_table["codigo"] == code]
                if not m.empty:
                    price = float(m["precio_usd_por_ton"].iloc[0])
            if price is None or price <= 0:
                cfg = ESTIMATION_ASSUMPTIONS.get(code, ESTIMATION_ASSUMPTIONS["default"])
                price = cfg["precio_usd_por_ton"]
            value_usd = float(row["valor_exportado"]) * EXPORT_VALUE_SCALE
            volume = value_usd / price if price > 0 else 0.0
            # 1 contenedor promedio = 20 toneladas.
            containers = volume / 20.0 if volume > 0 else 0.0
            return volume, containers

        est = exports_kpi_df[["codigo", "valor_exportado"]].copy()
        est[["est_volume_tons", "est_containers"]] = est.apply(
            lambda r: pd.Series(_estimate_row(r)),
            axis=1,
        )
        estimated_volume_tons = float(est["est_volume_tons"].sum())
        estimated_containers = float(est["est_containers"].sum())

    product_trend: list[dict] = []
    product_labels: list[dict] = []
    if not exports_long.empty:
        exports_for_trend = exports_long.copy()
        if year != "todos":
            exports_for_trend = exports_for_trend[exports_for_trend["year"] == int(year)]

        if selected_codes is not None:
            selected_codes_for_trend = selected_codes
        else:
            latest_year = int(exports_for_trend["year"].max()) if not exports_for_trend.empty else int(exports_long["year"].max())
            selected_codes_for_trend = (
                exports_for_trend[exports_for_trend["year"] == latest_year]
                .sort_values("valor_exportado", ascending=False)
                .drop_duplicates("codigo")
                .head(8)["codigo"]
                .tolist()
            )

        product_df = exports_for_trend[exports_for_trend["codigo"].isin(selected_codes_for_trend)].copy()
        product_df = product_df.sort_values(["year", "codigo"])
        product_df["period"] = product_df["year"].astype(str)

        product_trend = product_df.to_dict("records")
        product_labels = (
            product_df[["codigo", "producto"]]
            .drop_duplicates()
            .sort_values("codigo")
            .rename(columns={"codigo": "codigo_arancelario", "producto": "nombre_producto"})
            .to_dict("records")
        )
    route_risk = (
        df.groupby("ruta_logistica", as_index=False)
        .agg(value_usd=("value_usd", "sum"), risk=("incidence_probability", "mean"))
        .sort_values("value_usd", ascending=False)
    )

    top_states: list[dict] = []
    if not state_exports.empty:
        if segment in {"bovino", "porcino"}:
            states_df = state_exports[state_exports["segment"] == segment].copy()
        else:
            states_df = state_exports.copy()

        top_states_df = (
            states_df.groupby("state", as_index=False)["trade_value"]
            .sum()
            .sort_values("trade_value", ascending=False)
            .head(5)
        )
        top_states = top_states_df.to_dict("records")

    company_share: list[dict] = []
    if not company_share_df.empty:
        top5 = company_share_df.head(5).copy()
        rest = float(company_share_df.iloc[5:]["participacion_mercado"].sum())
        company_share = top5.rename(columns={"empresa": "label"}).to_dict("records")
        if rest > 0:
            company_share.append({"label": "Otras", "participacion_mercado": rest})

    analyzed_companies = 0
    if not routes_usage.empty and "companies" in routes_usage.columns:
        usage_df = routes_usage.copy()
        if route != "todas":
            route_clean = str(route).strip().lower()
            usage_df = usage_df[usage_df["route_name"].astype(str).str.lower().str.strip() == route_clean]
        company_names: set[str] = set()
        for companies in usage_df["companies"].tolist():
            if isinstance(companies, list):
                for name in companies:
                    nm = str(name).strip()
                    if nm:
                        company_names.add(nm)
        analyzed_companies = len(company_names)

    # Fallback cuando no hay dataset de rutas/destinos disponible.
    if analyzed_companies == 0:
        analyzed_companies = int(company_share_df["empresa"].nunique()) if not company_share_df.empty else 0

    risk_mean = 0.0
    if route != "todas":
        # Si hay ruta seleccionada, usar solo esa ruta (sin fallback global).
        risk_source_df = route_tons_risk_df
    else:
        risk_source_df = all_routes_tons_risk_df
    if not risk_source_df.empty:
        risk_vals = pd.to_numeric(risk_source_df["risk_index"], errors="coerce").dropna()
        if not risk_vals.empty:
            raw_risk_mean = float(risk_vals.mean())
            # KPI en escala 0-1 para mostrar porcentaje en frontend.
            # Si la fuente viene en 0-100, normalizar; si ya viene en 0-1, respetar.
            risk_mean = raw_risk_mean / 100.0 if raw_risk_mean > 1.5 else raw_risk_mean
            risk_mean = max(0.0, min(1.0, risk_mean))

    return {
        "kpis": {
            "volume_tons": estimated_volume_tons if estimated_volume_tons is not None else float(df["volume_tons"].sum()),
            "value_usd": value_usd_from_exports if value_usd_from_exports is not None else float(df["value_usd"].sum()),
            "containers": int(round(estimated_containers))
            if estimated_containers is not None
            else int(round(float(df["volume_tons"].sum()) / 20.0)),
            "risk": risk_mean,
            "companies": analyzed_companies,
        },
        "product_trend": product_trend,
        "product_labels": product_labels,
        "top_states": top_states,
        "company_share": company_share,
        "route_tons_risk": route_tons_risk_df.to_dict("records"),
        "route_risk": route_risk.to_dict("records"),
    }


@app.get("/api/ranking")
def ranking(
    lookback_months: int = 12,
    year: str = "todos",
    w1: float = Query(0.22, ge=0, le=1),
    w2: float = Query(0.26, ge=0, le=1),
    w3: float = Query(0.12, ge=0, le=1),
    w4: float = Query(0.14, ge=0, le=1),
    w5: float = Query(0.14, ge=0, le=1),
    w6: float = Query(0.12, ge=0, le=1),
    top_n: int = 30,
) -> dict:
    _, _, monthly, _, _, _ = load_data()
    if year != "todos":
        monthly = monthly[pd.to_datetime(monthly["month"]).dt.year == int(year)].copy()
    weights = {
        "volume_tons": w1,
        "value_usd": w2,
        "distance_km": w3,
        "sensitivity_temperature": w4,
        "risk_congestion": w5,
        "incidences_proxy": w6,
    }
    ranked = rank_companies(monthly, weights=weights, lookback_months=lookback_months)
    top = ranked.head(top_n)
    return {
        "weights": weights,
        "data": top[
            [
                "company_id",
                "empresa",
                "segmento",
                "volume_tons",
                "value_usd",
                "incidence_probability",
                "risk_congestion",
                "opportunity_score",
                "drivers_score",
                "ruta_principal",
            ]
        ].to_dict("records"),
    }


@app.get("/api/map-flows")
def map_flows(
    segment: str = "todos",
    route: str = "todas",
    company_id: int | None = None,
    year: str = "todos",
) -> dict:
    routes_usage = load_routes_company_usage()
    if not routes_usage.empty:
        df = routes_usage.copy()
        if route != "todas":
            route_clean = str(route).strip().lower()
            route_name_series = df["route_name"].astype(str).str.lower().str.strip()
            df = df[route_name_series == route_clean]
        if company_id:
            # El dataset destinos_rutas_top20 no comparte company_id numérico confiable.
            # Se conserva el parámetro para compatibilidad sin aplicar filtro aquí.
            pass

        df = df.sort_values("average_yearly_tons", ascending=False)
        return {"data": df.to_dict("records")}

    _, flows, monthly, _, _, _ = load_data()
    recent = monthly.copy()
    if year != "todos":
        recent = recent[pd.to_datetime(recent["month"]).dt.year == int(year)]

    flow_filter = flows.copy()
    if segment != "todos":
        flow_filter = flow_filter[flow_filter["segmento"] == segment]
    if route != "todas":
        flow_filter = flow_filter[flow_filter["ruta_logistica"] == route]
    if company_id:
        flow_filter = flow_filter[flow_filter["company_id"] == company_id]

    stats = (
        recent.groupby("flow_id", as_index=False)
        .agg(volumen=("volume_tons", "sum"), valor=("value_usd", "sum"), riesgo=("incidence_probability", "mean"))
        .merge(
            flow_filter[
                [
                    "flow_id",
                    "empresa",
                    "segmento",
                    "ruta_logistica",
                    "origin_lat",
                    "origin_lon",
                    "dest_lat",
                    "dest_lon",
                ]
            ],
            on="flow_id",
            how="inner",
        )
        .sort_values("valor", ascending=False)
    )
    return {"data": stats.to_dict("records")}


@app.get("/api/scenarios")
def scenarios(
    adoption_rate: float = Query(0.30, ge=0, le=1),
    adoption_effectiveness: float = Query(0.35, ge=0, le=1),
    risk_multiplier: float = Query(1.15, ge=1, le=2),
    expansion_growth: float = Query(0.20, ge=0, le=1),
    year: str = "todos",
) -> dict:
    _, _, monthly, _, _, _ = load_data()
    base_df = monthly.copy()
    if year != "todos":
        base_df = base_df[pd.to_datetime(base_df["month"]).dt.year == int(year)].copy()
    summary, details = run_scenarios(
        base_df,
        adoption_rate=adoption_rate,
        adoption_effectiveness=adoption_effectiveness,
        risk_multiplier=risk_multiplier,
        expansion_growth=expansion_growth,
    )
    detail_top = {
        k: v[["empresa", "segmento", "ruta_logistica", "value_usd", "expected_loss"]]
        .sort_values("expected_loss", ascending=False)
        .head(25)
        .to_dict("records")
        for k, v in details.items()
    }
    return {"summary": summary.to_dict("records"), "details": detail_top}


@app.get("/api/forecast")
def forecast(
    target: str = Query("value_usd", pattern="^(value_usd|volume_tons|shipment_count)$"),
    segment: str = "todos",
    company_id: int | None = None,
    model_name: str = Query("SARIMA", pattern="^(ARIMA|SARIMA|Holt-Winters)$"),
    horizon: int = Query(6, ge=3, le=12),
) -> dict:
    _, _, monthly, _, _, _ = load_data()
    series = prepare_monthly_series(monthly, target_col=target, segment=segment, company_id=company_id)
    result = forecast_series(series, model_name=model_name, horizon=horizon)
    hist = series.reset_index()
    hist.columns = ["month", "value"]
    hist["month"] = hist["month"].dt.strftime("%Y-%m-%d")
    fc = result["forecast_df"].copy()
    fc["month"] = pd.to_datetime(fc["month"]).dt.strftime("%Y-%m-%d")
    return {
        "history": hist.to_dict("records"),
        "forecast": fc.to_dict("records"),
        "mape": None if pd.isna(result["mape"]) else float(result["mape"]),
        "rmse": None if pd.isna(result["rmse"]) else float(result["rmse"]),
    }


@app.get("/api/forecast-arimax-quarterly")
def forecast_arimax(horizon_q: int = Query(8, ge=4, le=12)) -> dict:
    if not EXPORTS_TRIMESTRALES_CSV.exists() or not FX_TRIMESTRAL_CSV.exists():
        return {"bovino": {"history": [], "forecast": [], "metrics": {}}, "porcino": {"history": [], "forecast": [], "metrics": {}}}

    data = forecast_arimax_quarterly(
        exports_path=EXPORTS_TRIMESTRALES_CSV,
        fx_path=FX_TRIMESTRAL_CSV,
        horizon_q=horizon_q,
    )
    for seg in ["bovino", "porcino"]:
        hist = pd.DataFrame(data[seg]["history"])
        fc = pd.DataFrame(data[seg]["forecast"])
        if not hist.empty:
            hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
            data[seg]["history"] = hist.to_dict("records")
        if not fc.empty:
            fc["date"] = pd.to_datetime(fc["date"]).dt.strftime("%Y-%m-%d")
            data[seg]["forecast"] = fc.to_dict("records")
    return data


@app.get("/api/clustering")
def clustering(k: int = Query(3, ge=2, le=8), max_k: int = Query(8, ge=4, le=10)) -> dict:
    raw = load_cluster_input_real()
    if raw.empty:
        return {"diagnostics": [], "profiles": [], "pca": [], "loadings": []}

    # Flujo alineado al notebook clustering_final.py:
    # 1) eliminar columnas redundantes/NA-heavy
    # 2) MinMax solo en columnas numericas no binarias
    features_raw = raw.copy()
    drop_cols = [c for c in ["crecimiento_yoy", "datos_estimados", "viajes_mensuales"] if c in features_raw.columns]
    if drop_cols:
        features_raw = features_raw.drop(columns=drop_cols)

    # Copia para clustering (escalada) y copia para perfil (valores reales).
    features = features_raw.copy()

    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    cols_to_scale = []
    for col in numeric_cols:
        uniques = set(pd.Series(features[col]).dropna().unique().tolist())
        if not uniques.issubset({0, 1}):
            cols_to_scale.append(col)

    if cols_to_scale:
        scaler = MinMaxScaler()
        features.loc[:, cols_to_scale] = scaler.fit_transform(features[cols_to_scale])

    feature_cols = features.select_dtypes(include="number").columns.tolist()
    if not feature_cols:
        return {"diagnostics": [], "profiles": [], "pca": [], "loadings": []}
    X = features[feature_cols].values

    max_k_limit = min(14, len(features) - 1)
    max_k_eff = max(3, min(max_k, max_k_limit))
    diag_rows = []
    for kk in range(2, max_k_eff + 1):
        model = KMeans(n_clusters=kk, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
        diag_rows.append({"k": kk, "inertia": float(model.inertia_), "silhouette": float(sil)})
    diag = pd.DataFrame(diag_rows)

    k_eff = max(2, min(k, len(features) - 1))
    model = KMeans(n_clusters=k_eff, random_state=42, n_init=20)
    labels = model.fit_predict(X)

    n_comp = 3 if X.shape[1] >= 3 else 2
    pca_model = PCA(n_components=n_comp, random_state=42)
    pcs = pca_model.fit_transform(X)

    clustered = features.copy()
    clustered["cluster"] = labels
    clustered["pca_1"] = pcs[:, 0]
    clustered["pca_2"] = pcs[:, 1]
    clustered["pca_3"] = pcs[:, 2] if n_comp >= 3 else 0.0

    # Perfil interpretativo en escala real (sin MinMax), como en el notebook.
    profile_numeric_cols = [c for c in features_raw.columns if c != "empresa" and pd.api.types.is_numeric_dtype(features_raw[c])]
    profile_df = features_raw.copy()
    profile_df["cluster"] = labels
    profile = profile_df.groupby("cluster", as_index=False)[profile_numeric_cols].mean()
    profile["n_empresas"] = profile_df.groupby("cluster").size().values
    if "participacion_mercado" in profile.columns:
        profile = profile.sort_values("participacion_mercado", ascending=False)

    pca_cols = ["empresa", "cluster", "pca_1", "pca_2", "pca_3"]
    if "participacion_mercado" in clustered.columns:
        pca_cols.append("participacion_mercado")
    pca = clustered[pca_cols].copy()
    pca["cluster"] = pca["cluster"].astype(str)
    if "participacion_mercado" in pca.columns:
        pca = pca.rename(columns={"participacion_mercado": "opportunity_score"})
    else:
        pca["opportunity_score"] = 0.0

    loadings_df = pd.DataFrame(
        pca_model.components_.T,
        columns=[f"pc_{i + 1}" for i in range(pca_model.n_components_)],
        index=feature_cols,
    ).reset_index().rename(columns={"index": "variable"})
    if "pc_3" not in loadings_df.columns:
        loadings_df["pc_3"] = 0.0

    return {
        "diagnostics": diag.to_dict("records"),
        "profiles": profile.to_dict("records"),
        "pca": pca.to_dict("records"),
        "loadings": loadings_df[["variable", "pc_1", "pc_2", "pc_3"]].to_dict("records"),
    }


@app.get("/api/leads")
def leads(top_n: int = Query(30, ge=10, le=80)) -> dict:
    _, _, monthly, _, _, _ = load_data()
    ranking_df = rank_companies(monthly, weights=DEFAULT_WEIGHTS, lookback_months=12)
    features = build_feature_table(monthly, ranking_df)
    clustered, _ = fit_clusters(features, k=3)
    leads_df = generate_leads(ranking_df, clustered, top_n=top_n)
    return {"data": leads_df.to_dict("records")}


@app.get("/api/close-probability-model")
def close_probability_model(include_simulated: bool = True, top_n: int = Query(120, ge=10, le=500)) -> dict:
    # Fuente prioritaria solicitada por usuario: data/df_final.csv
    if DF_FINAL_CSV.exists():
        raw = pd.read_csv(DF_FINAL_CSV)
        if "empresa" not in raw.columns:
            raw["empresa"] = [f"Empresa_{i+1}" for i in range(len(raw))]

        required = [
            "participacion_mercado",
            "ventas_anuales_musd",
            "frecuencia_exportacion_dias",
            "presencia_internacional",
            "reporte_esg",
            "movimiento_cadena_frio",
            "volumen_exportaciones_musd",
            "mas_40_viajes_mensuales",
            "cluster_label",
            "probabilidad_cierre_modelo",
        ]
        for col in required:
            if col not in raw.columns:
                raw[col] = 0.0
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0) if col != "empresa" else raw[col]

        raw["cluster_label"] = pd.to_numeric(raw["cluster_label"], errors="coerce").fillna(0.0).round().astype(int)
        raw["cluster_label"] = raw["cluster_label"].clip(lower=0, upper=3)
        raw["probabilidad_cierre_modelo"] = pd.to_numeric(raw["probabilidad_cierre_modelo"], errors="coerce").fillna(0.0)
        raw["origen_dato"] = np.where(raw["empresa"].astype(str).str.startswith("Empresa_sim_"), "simulado", "real")

        out = raw[
            [
                "empresa",
                "participacion_mercado",
                "ventas_anuales_musd",
                "frecuencia_exportacion_dias",
                "presencia_internacional",
                "reporte_esg",
                "movimiento_cadena_frio",
                "volumen_exportaciones_musd",
                "mas_40_viajes_mensuales",
                "cluster_label",
                "origen_dato",
                "probabilidad_cierre_modelo",
            ]
        ].copy()
        if not include_simulated:
            out = out[out["origen_dato"] == "real"].copy()
        out = out.sort_values("probabilidad_cierre_modelo", ascending=False).head(top_n)

        metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}
        class_metrics = {
            "0": {"precision": None, "recall": None, "f1": None, "support": None},
            "1": {"precision": None, "recall": None, "f1": None, "support": None},
        }
        if "Cierre" in raw.columns:
            y_true = pd.to_numeric(raw["Cierre"], errors="coerce")
            y_prob = pd.to_numeric(raw["probabilidad_cierre_modelo"], errors="coerce")
            valid = y_true.notna() & y_prob.notna()
            if valid.any():
                y_true_v = y_true[valid].astype(int)
                y_prob_v = y_prob[valid].astype(float).clip(0.0, 1.0)
                y_pred_v = (y_prob_v >= 0.5).astype(int)
                metrics["accuracy"] = float(accuracy_score(y_true_v, y_pred_v))
                metrics["precision"] = float(precision_score(y_true_v, y_pred_v, zero_division=0))
                metrics["recall"] = float(recall_score(y_true_v, y_pred_v, zero_division=0))
                metrics["f1"] = float(f1_score(y_true_v, y_pred_v, zero_division=0))
                if y_true_v.nunique() > 1:
                    metrics["roc_auc"] = float(roc_auc_score(y_true_v, y_prob_v))
                for label in [0, 1]:
                    class_metrics[str(label)]["precision"] = float(
                        precision_score(y_true_v, y_pred_v, pos_label=label, zero_division=0)
                    )
                    class_metrics[str(label)]["recall"] = float(
                        recall_score(y_true_v, y_pred_v, pos_label=label, zero_division=0)
                    )
                    class_metrics[str(label)]["f1"] = float(
                        f1_score(y_true_v, y_pred_v, pos_label=label, zero_division=0)
                    )
                    class_metrics[str(label)]["support"] = int((y_true_v == label).sum())

        cluster_probabilities = (
            out.groupby("cluster_label", as_index=False)["probabilidad_cierre_modelo"]
            .mean()
            .rename(columns={"cluster_label": "cluster", "probabilidad_cierre_modelo": "probabilidad_media_cierre"})
            .sort_values("cluster")
            .to_dict("records")
        )
        out = out.replace([np.inf, -np.inf], np.nan).where(pd.notna(out), None)
        return {
            "model": {
                "model_name": "df_final_source",
                "target_n": int(len(raw)),
                "n_real": int((raw["origen_dato"] == "real").sum()),
                "n_simulado": int((raw["origen_dato"] == "simulado").sum()),
                "metrics": metrics,
                "class_metrics": class_metrics,
                "feature_importance": NOTEBOOK_FEATURE_IMPORTANCE,
            },
            "class_metrics": class_metrics,
            "feature_importance": NOTEBOOK_FEATURE_IMPORTANCE,
            "cluster_probabilities": cluster_probabilities,
            "data": out.to_dict("records"),
        }

    try:
        table_df, meta = load_close_probability_model()
        out = table_df.copy()
        if not include_simulated and "origen_dato" in out.columns:
            out = out[out["origen_dato"] == "real"].copy()
        out = out.head(top_n)
    except Exception:
        # Fallback robusto basado en el archivo de entrada para no bloquear la UI.
        source_csv = EMPRESAS_CON_CLUSTERS_CSV if EMPRESAS_CON_CLUSTERS_CSV.exists() else COMPANY_MARKET_SHARE_CSV
        raw = pd.read_csv(source_csv)
        needed = [
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
        for col in needed:
            if col not in raw.columns:
                raw[col] = 0.0
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
        raw["cluster_label"] = pd.to_numeric(raw["cluster_label"], errors="coerce").fillna(0.0).round().astype(int)
        raw["cluster_label"] = raw["cluster_label"].clip(lower=0, upper=3)
        if "empresa" not in raw.columns:
            raw["empresa"] = [f"Empresa_{i+1}" for i in range(len(raw))]

        def _z(series: pd.Series) -> pd.Series:
            std = float(series.std())
            if std == 0:
                return pd.Series(0.0, index=series.index)
            return (series - float(series.mean())) / std

        score = (
            0.4 * _z(raw["ventas_anuales_musd"])
            + 1.1 * _z(raw["volumen_exportaciones_musd"])
            + 0.4 * _z(raw["participacion_mercado"])
            - 0.7 * _z(raw["frecuencia_exportacion_dias"])
            + 0.9 * raw["presencia_internacional"]
            + 0.7 * raw["movimiento_cadena_frio"]
            + 0.4 * raw["reporte_esg"]
            + 0.5 * raw["mas_40_viajes_mensuales"]
        )
        cluster_effect = {0: -0.4, 1: 0.7, 2: 0.5, 3: 0.4}
        score = score + raw["cluster_label"].map(cluster_effect).fillna(0.0)
        raw["probabilidad_cierre_modelo"] = 1.0 / (1.0 + np.exp(-score))
        raw["origen_dato"] = "real"
        out = raw[
            [
                "empresa",
                "participacion_mercado",
                "ventas_anuales_musd",
                "frecuencia_exportacion_dias",
                "presencia_internacional",
                "reporte_esg",
                "movimiento_cadena_frio",
                "volumen_exportaciones_musd",
                "mas_40_viajes_mensuales",
                "cluster_label",
                "origen_dato",
                "probabilidad_cierre_modelo",
            ]
        ].sort_values("probabilidad_cierre_modelo", ascending=False)
        out = out.head(top_n)
        meta = {
            "model_name": "fallback_scoring",
            "target_n": int(len(raw)),
            "n_real": int(len(raw)),
            "n_simulado": 0,
            "metrics": {"accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None},
            "feature_importance": {
                "participacion_mercado": None,
                "ventas_anuales_musd": None,
                "frecuencia_exportacion_dias": None,
                "presencia_internacional": None,
                "reporte_esg": None,
                "movimiento_cadena_frio": None,
                "volumen_exportaciones_musd": None,
                "mas_40_viajes_mensuales": None,
                "cluster_label": None,
            },
        }

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notna(out), None)
    # Mostrar la importancia de variables calibrada con el resultado del notebook base del usuario.
    meta["feature_importance"] = NOTEBOOK_FEATURE_IMPORTANCE
    # Mostrar la probabilidad por cluster calibrada con el resultado del notebook base del usuario.
    cluster_probabilities: list[dict] = NOTEBOOK_CLUSTER_PROBABILITIES
    class_metrics_out = meta.get(
        "class_metrics",
        {
            "0": {"precision": None, "recall": None, "f1": None, "support": None},
            "1": {"precision": None, "recall": None, "f1": None, "support": None},
        },
    )
    return {
        "model": {**meta, "class_metrics": class_metrics_out},
        "class_metrics": class_metrics_out,
        "feature_importance": meta.get("feature_importance", {}),
        "cluster_probabilities": cluster_probabilities,
        "data": out.to_dict("records"),
    }
