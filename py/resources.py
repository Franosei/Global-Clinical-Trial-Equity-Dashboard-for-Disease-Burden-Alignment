"""World Bank country resource layer for cost-aware site planning.

Every value is a published measurement; nothing is imputed. Where a country
lacks a measurement the field stays empty and the planner reports it.

Sources:
  • World Bank country metadata: income group, region, capital coordinates
  • ICP 2021 (World Bank source 90): price level index for "Actual health"
    expenditure (World = 100). This is the cost-scaling basis. The GDP price
    level index from the same programme is kept as a labelled fallback.
  • World Development Indicators: research and health-system infrastructure

Output:
  data/clean/country_resources.parquet

Run:
  python -m py.resources
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pycountry
import requests
from requests.adapters import HTTPAdapter, Retry
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO_ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
OUT_PARQUET = REPO_ROOT / "data" / "clean" / "country_resources.parquet"

WB = CFG.get("world_bank", {})
BASE = str(WB.get("base_url", "https://api.worldbank.org/v2/")).rstrip("/") + "/"
ICP_SOURCE = str(WB.get("icp_source", "90"))
ICP_YEAR = str(WB.get("icp_year", "2021"))
ICP_HEALTH_SERIES = "9080000"  # ACTUAL HEALTH
ICP_GDP_SERIES = "1000000"     # GROSS DOMESTIC PRODUCT
ICP_PRICE_LEVEL = "PX.WL"      # Price level index (World = 100)

# column name -> (WDI code, label shown in the dashboard)
INDICATORS: dict[str, tuple[str, str]] = {
    "population": ("SP.POP.TOTL", "Population"),
    "physicians_per_1000": ("SH.MED.PHYS.ZS", "Physicians per 1,000 people"),
    "hospital_beds_per_1000": ("SH.MED.BEDS.ZS", "Hospital beds per 1,000 people"),
    "health_spend_per_capita_usd": ("SH.XPD.CHEX.PC.CD", "Health expenditure per capita (current US$)"),
    "electricity_access_pct": ("EG.ELC.ACCS.ZS", "Access to electricity (% of population)"),
    "urban_population_pct": ("SP.URB.TOTL.IN.ZS", "Urban population (% of total)"),
    "logistics_performance_index": ("LP.LPI.OVRL.XQ", "Logistics performance index (1 to 5)"),
    "researchers_per_million": ("SP.POP.SCIE.RD.P6", "Researchers in R&D per million people"),
}

VALID_ISO3 = frozenset(country.alpha_3 for country in pycountry.countries)


def _session() -> requests.Session:
    retries = CFG.get("retries", {})
    session = requests.Session()
    retry = Retry(
        total=int(retries.get("total", 5)),
        backoff_factor=float(retries.get("backoff_factor", 0.5)),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.headers.update({"User-Agent": "Bayezian-EquityDashboard/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = _session()


def _get_pages(url: str, params: dict) -> list:
    rows, page, pages = [], 1, 1
    while page <= pages:
        response = SESSION.get(url, params={**params, "format": "json", "page": page}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"World Bank API returned no data for {url}: {payload}")
        pages = int(payload[0].get("pages", 1))
        rows.extend(payload[1] or [])
        page += 1
    return rows


def fetch_countries() -> pd.DataFrame:
    rows = _get_pages(BASE + "country", {"per_page": 400})
    frame = pd.DataFrame([{
        "country_iso3": row["id"],
        "country_name": row["name"],
        "region": row["region"]["value"],
        "income_group": row["incomeLevel"]["value"],
        "capital_city": row.get("capitalCity") or None,
        "capital_lat": pd.to_numeric(row.get("latitude"), errors="coerce"),
        "capital_lon": pd.to_numeric(row.get("longitude"), errors="coerce"),
    } for row in rows if row["region"]["id"] != "NA"])  # "NA" marks aggregates
    return frame[frame["country_iso3"].isin(VALID_ISO3)]


def fetch_wdi(code: str) -> pd.DataFrame:
    """Most recent non-empty value per country, with its year."""
    rows = _get_pages(BASE + f"country/all/indicator/{code}", {"per_page": 20000, "mrnev": 1})
    frame = pd.DataFrame([{
        "country_iso3": row["countryiso3code"],
        "value": row["value"],
        "year": pd.to_numeric(row["date"], errors="coerce"),
    } for row in rows if row.get("value") is not None and row.get("countryiso3code")])
    return frame[frame["country_iso3"].isin(VALID_ISO3)]


def fetch_icp_price_level(series: str) -> pd.DataFrame:
    url = BASE + (
        f"sources/{ICP_SOURCE}/country/all/series/{series}"
        f"/classification/{ICP_PRICE_LEVEL}/time/YR{ICP_YEAR}/data"
    )
    response = SESSION.get(url, params={"format": "json", "per_page": 1000}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    records = []
    for pages_seen in range(int(payload.get("pages", 1))):
        if pages_seen:
            response = SESSION.get(url, params={"format": "json", "per_page": 1000, "page": pages_seen + 1}, timeout=120)
            response.raise_for_status()
            payload = response.json()
        for item in payload["source"]["data"]:
            country = next(v["id"] for v in item["variable"] if v["concept"] == "Country")
            if item.get("value") is not None:
                records.append({"country_iso3": country, "value": float(item["value"])})
    frame = pd.DataFrame(records, columns=["country_iso3", "value"])
    return frame[frame["country_iso3"].isin(VALID_ISO3)]


def run_resources() -> pd.DataFrame:
    out = fetch_countries()

    health = fetch_icp_price_level(ICP_HEALTH_SERIES).rename(columns={"value": "health_price_level"})
    gdp = fetch_icp_price_level(ICP_GDP_SERIES).rename(columns={"value": "gdp_price_level"})
    out = out.merge(health, on="country_iso3", how="left").merge(gdp, on="country_iso3", how="left")
    out["price_level_year"] = int(ICP_YEAR)

    for column, (code, _label) in INDICATORS.items():
        values = fetch_wdi(code).rename(columns={"value": column, "year": f"{column}_year"})
        out = out.merge(values, on="country_iso3", how="left")
        print(f"[WB] {code:<20} {values[column].notna().sum():>4} countries")

    print(f"[ICP {ICP_YEAR}] health price level: {out['health_price_level'].notna().sum()} countries; "
          f"GDP price level: {out['gdp_price_level'].notna().sum()} countries")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PARQUET.with_suffix(".parquet.tmp")
    out.sort_values("country_iso3").to_parquet(temporary, index=False)
    temporary.replace(OUT_PARQUET)
    print(f"[OK] Wrote {OUT_PARQUET} (rows={len(out)})")
    return out


if __name__ == "__main__":
    try:
        run_resources()
    except Exception as exc:
        print(f"\n[ERROR] resources.py failed:\n{exc}")
        sys.exit(1)
