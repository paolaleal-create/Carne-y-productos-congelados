from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_FILE = ROOT / "config" / "config.yaml"


def ensure_mock_data() -> None:
    required = [DATA_DIR / "companies.csv", DATA_DIR / "flows.csv", DATA_DIR / "trade_monthly.csv"]
    if all(path.exists() for path in required):
        return
    subprocess.run([sys.executable, str(DATA_DIR / "generate_mock_data.py")], check=True, cwd=str(ROOT))


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    ensure_mock_data()
    companies = pd.read_csv(DATA_DIR / "companies.csv")
    flows = pd.read_csv(DATA_DIR / "flows.csv")
    monthly = pd.read_csv(DATA_DIR / "trade_monthly.csv", parse_dates=["month"])
    return {"companies": companies, "flows": flows, "monthly": monthly}


def query_with_duckdb(monthly_df: pd.DataFrame, sql: str) -> pd.DataFrame:
    conn = duckdb.connect(database=":memory:")
    conn.register("trade", monthly_df)
    out = conn.execute(sql).fetchdf()
    conn.close()
    return out

