from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.utils import load_data


st.title("3. Mapa de flujos Mexico -> Estados Unidos")

data = load_data()
flows = data["flows"]
monthly = data["monthly"]

companies = ["todas"] + sorted(flows["empresa"].unique().tolist())
routes = ["todas"] + sorted(flows["ruta_logistica"].unique().tolist())
segments = ["todos"] + sorted(flows["segmento"].unique().tolist())

c1, c2, c3 = st.columns(3)
empresa = c1.selectbox("Empresa", companies, index=0)
ruta = c2.selectbox("Ruta", routes, index=0)
segmento = c3.selectbox("Segmento", segments, index=0)

flow_filter = flows.copy()
if empresa != "todas":
    flow_filter = flow_filter[flow_filter["empresa"] == empresa]
if ruta != "todas":
    flow_filter = flow_filter[flow_filter["ruta_logistica"] == ruta]
if segmento != "todos":
    flow_filter = flow_filter[flow_filter["segmento"] == segmento]

latest_cutoff = monthly["month"].max() - pd.DateOffset(months=11)
recent = monthly[monthly["month"] >= latest_cutoff]

flow_stats = (
    recent.groupby("flow_id", as_index=False)
    .agg(volumen=("volume_tons", "sum"), riesgo=("incidence_probability", "mean"), valor=("value_usd", "sum"))
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
)

if flow_stats.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

flow_stats["color_r"] = (255 * flow_stats["riesgo"]).astype(int).clip(0, 255)
flow_stats["color_g"] = (255 * (1 - flow_stats["riesgo"])).astype(int).clip(0, 255)
flow_stats["color_b"] = 80
flow_stats["line_width"] = (flow_stats["volumen"] / flow_stats["volumen"].max() * 12 + 2).clip(2, 15)

layer = pdk.Layer(
    "ArcLayer",
    data=flow_stats,
    get_source_position="[origin_lon, origin_lat]",
    get_target_position="[dest_lon, dest_lat]",
    get_source_color="[color_r, color_g, color_b, 180]",
    get_target_color="[color_r, color_g, color_b, 180]",
    get_width="line_width",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(latitude=30.0, longitude=-104.0, zoom=4.0, pitch=35)
tooltip = {
    "html": "<b>{empresa}</b><br/>{ruta_logistica}<br/>Segmento: {segmento}<br/>Volumen: {volumen}<br/>Riesgo: {riesgo}",
    "style": {"backgroundColor": "steelblue", "color": "white"},
}

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

st.dataframe(
    flow_stats[["empresa", "segmento", "ruta_logistica", "volumen", "valor", "riesgo"]].sort_values(
        "valor", ascending=False
    ),
    use_container_width=True,
)

