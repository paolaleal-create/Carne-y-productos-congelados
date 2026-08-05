from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


def build_companies(rng: np.random.Generator, n_companies: int = 100) -> pd.DataFrame:
    prefixes = [
        "Frio",
        "Norte",
        "Carnes",
        "Proteina",
        "Export",
        "Ganadera",
        "LogiMeat",
        "ColdTrade",
        "Delta",
        "Atlas",
    ]
    suffixes = [
        "Foods",
        "MX",
        "Prime",
        "Global",
        "Link",
        "Group",
        "Solutions",
        "Pack",
        "Chain",
        "Express",
    ]
    segments = ["bovino", "porcino", "mixto"]
    weights = [0.4, 0.35, 0.25]

    companies = []
    for cid in range(1, n_companies + 1):
        name = f"{rng.choice(prefixes)} {rng.choice(suffixes)} {cid:03d}"
        segment_focus = rng.choice(segments, p=weights)
        size = rng.choice(["grande", "mediana", "pequena"], p=[0.25, 0.5, 0.25])
        companies.append(
            {
                "company_id": cid,
                "empresa": name,
                "segmento_foco": segment_focus,
                "tamano_empresa": size,
            }
        )
    return pd.DataFrame(companies)


def build_flows(
    rng: np.random.Generator, companies_df: pd.DataFrame, n_flows: int = 200
) -> pd.DataFrame:
    routes = [
        {
            "ruta_logistica": "Nogales - Arizona",
            "origen": "Nogales, Sonora",
            "destino": "Nogales, Arizona",
            "origin_lat": 31.3086,
            "origin_lon": -110.9458,
            "dest_lat": 31.3404,
            "dest_lon": -110.9343,
            "distance_base_km": 350,
            "congestion_base": 0.52,
        },
        {
            "ruta_logistica": "Nuevo Laredo - Texas",
            "origen": "Nuevo Laredo, Tamaulipas",
            "destino": "Laredo, Texas",
            "origin_lat": 27.4864,
            "origin_lon": -99.5075,
            "dest_lat": 27.5306,
            "dest_lon": -99.4803,
            "distance_base_km": 900,
            "congestion_base": 0.67,
        },
        {
            "ruta_logistica": "Reynosa - Pharr",
            "origen": "Reynosa, Tamaulipas",
            "destino": "Pharr, Texas",
            "origin_lat": 26.0922,
            "origin_lon": -98.2779,
            "dest_lat": 26.1948,
            "dest_lon": -98.1836,
            "distance_base_km": 780,
            "congestion_base": 0.64,
        },
        {
            "ruta_logistica": "Tijuana - San Diego",
            "origen": "Tijuana, Baja California",
            "destino": "San Diego, California",
            "origin_lat": 32.5149,
            "origin_lon": -117.0382,
            "dest_lat": 32.7157,
            "dest_lon": -117.1611,
            "distance_base_km": 420,
            "congestion_base": 0.58,
        },
        {
            "ruta_logistica": "Ciudad Juarez - El Paso",
            "origen": "Ciudad Juarez, Chihuahua",
            "destino": "El Paso, Texas",
            "origin_lat": 31.6904,
            "origin_lon": -106.4245,
            "dest_lat": 31.7619,
            "dest_lon": -106.4850,
            "distance_base_km": 740,
            "congestion_base": 0.61,
        },
    ]

    flows = []
    for fid in range(1, n_flows + 1):
        company = companies_df.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        route = routes[int(rng.integers(0, len(routes)))]
        segment_focus = company["segmento_foco"]
        segment = rng.choice(["bovino", "porcino"]) if segment_focus == "mixto" else segment_focus
        if segment == "mixto":
            segment = rng.choice(["bovino", "porcino"])

        sensitivity = 0.86 if segment == "bovino" else 0.78
        sensitivity += float(rng.normal(0, 0.06))
        sensitivity = float(np.clip(sensitivity, 0.4, 1.0))

        distance = float(np.clip(route["distance_base_km"] + rng.normal(0, 90), 180, 1500))
        transit_hrs = float(np.clip(distance / 58 + rng.normal(1.5, 1.3), 4, 48))
        climate_base = float(np.clip(rng.normal(0.45, 0.15), 0.1, 0.95))
        incidences_proxy = float(np.clip(rng.normal(0.32, 0.12), 0.05, 0.95))
        congestion = float(np.clip(route["congestion_base"] + rng.normal(0, 0.09), 0.1, 1.0))

        flows.append(
            {
                "flow_id": fid,
                "company_id": int(company["company_id"]),
                "empresa": company["empresa"],
                "segmento": segment,
                "ruta_logistica": route["ruta_logistica"],
                "origen": route["origen"],
                "destino": route["destino"],
                "origin_lat": route["origin_lat"],
                "origin_lon": route["origin_lon"],
                "dest_lat": route["dest_lat"],
                "dest_lon": route["dest_lon"],
                "distance_km": distance,
                "transport_time_hrs": transit_hrs,
                "congestion_base": congestion,
                "climate_variability_base": climate_base,
                "sensitivity_temperature": sensitivity,
                "incidences_proxy": incidences_proxy,
            }
        )
    return pd.DataFrame(flows)


def build_monthly_trade(rng: np.random.Generator, flows_df: pd.DataFrame) -> pd.DataFrame:
    months = pd.date_range(end="2026-02-01", periods=60, freq="MS")
    records: list[dict] = []

    for flow in flows_df.itertuples(index=False):
        size_factor = rng.choice([0.9, 1.1, 1.5], p=[0.45, 0.35, 0.20])
        segment_base = 130 if flow.segmento == "bovino" else 115
        base_volume = max(25, rng.normal(segment_base, 35) * size_factor)
        growth = rng.uniform(-0.03, 0.10)
        season_phase = rng.uniform(0, 2 * np.pi)

        for i, month in enumerate(months):
            season = 1 + 0.12 * np.sin(2 * np.pi * month.month / 12 + season_phase)
            trend = 1 + growth * (i / len(months))
            volume_tons = float(max(18, base_volume * season * trend * rng.lognormal(0, 0.13)))
            shipment_count = int(max(1, rng.poisson(volume_tons / 24)))

            base_price = 5200 if flow.segmento == "bovino" else 3550
            price_per_ton = float(max(2500, rng.normal(base_price, 230)))
            value_usd = float(volume_tons * price_per_ton * rng.uniform(0.96, 1.07))

            festive_bump = 0.08 if month.month in {11, 12} else 0.0
            congestion = float(
                np.clip(flow.congestion_base + festive_bump + rng.normal(0, 0.07), 0.05, 1.0)
            )
            climate_var = float(
                np.clip(
                    flow.climate_variability_base
                    + 0.10 * np.sin(2 * np.pi * month.month / 12)
                    + rng.normal(0, 0.06),
                    0.05,
                    1.0,
                )
            )

            incidence_prob = float(
                np.clip(
                    logistic(
                        -2.25
                        + 1.75 * congestion
                        + 1.30 * climate_var
                        + 0.02 * flow.transport_time_hrs
                        + 0.85 * flow.sensitivity_temperature
                        + 0.65 * flow.incidences_proxy
                    ),
                    0.02,
                    0.95,
                )
            )
            incidence_flag = int(rng.binomial(1, incidence_prob))

            records.append(
                {
                    "month": month,
                    "flow_id": int(flow.flow_id),
                    "company_id": int(flow.company_id),
                    "empresa": flow.empresa,
                    "segmento": flow.segmento,
                    "ruta_logistica": flow.ruta_logistica,
                    "volume_tons": volume_tons,
                    "value_usd": value_usd,
                    "shipment_count": shipment_count,
                    "distance_km": float(flow.distance_km),
                    "transport_time_hrs": float(flow.transport_time_hrs),
                    "congestion_border": congestion,
                    "climate_variability": climate_var,
                    "sensitivity_temperature": float(flow.sensitivity_temperature),
                    "incidences_proxy": float(flow.incidences_proxy),
                    "incidence_probability": incidence_prob,
                    "incidence_flag": incidence_flag,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    companies_df = build_companies(rng, n_companies=100)
    flows_df = build_flows(rng, companies_df, n_flows=200)
    monthly_df = build_monthly_trade(rng, flows_df)

    companies_df.to_csv(output_dir / "companies.csv", index=False)
    flows_df.to_csv(output_dir / "flows.csv", index=False)
    monthly_df.to_csv(output_dir / "trade_monthly.csv", index=False)

    print("Mock data generated:")
    print(f"- companies: {len(companies_df)}")
    print(f"- flows: {len(flows_df)}")
    print(f"- monthly records: {len(monthly_df)}")


if __name__ == "__main__":
    main()

