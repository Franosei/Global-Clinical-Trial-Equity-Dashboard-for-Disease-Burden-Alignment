# py/global_health_sources.py
"""
WHO Global Health Estimates (GHE) burden for the 13 tracked diseases.

Source: WHO Global Health Estimates 2023 summary tables, "DALY by country,
age and sex", one workbook per estimate year (2000, 2010, 2015, 2019 to 2023).
https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates

Each disease maps to exactly one GHE cause code, so no indicator is chosen by
keyword. The GHO OData API is not used because it publishes cause-specific
DALYs only for regions, not countries.

Output (Parquet): data/clean/who_daly_selected.parquet
  country_iso3, cause_code, cause_name, year,
  dalys                absolute DALYs, both sexes, all ages
  population           WHO GHE population for that year
  daly_rate_per_100k   dalys / population x 100,000
  who_data_quality     WHO GHE country data-quality colour code (see WHO notes)
  ghe_cause_code, ghe_cause_name, source

Run:
  python -m py.global_health_sources
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import unicodedata
from typing import Dict, List, Tuple

import openpyxl
import pandas as pd
import pycountry
import requests
from requests.adapters import HTTPAdapter, Retry
import yaml

# ------------------------ Paths & Config ------------------------

REPO_ROOT   = Path(__file__).resolve().parents[1]
CFG_PATH    = REPO_ROOT / "config" / "sources.yaml"
DATA_DIR    = REPO_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw" / "who_ghe"
CLEAN_DIR   = DATA_DIR / "clean"
OUT_PARQUET = CLEAN_DIR / "who_daly_selected.parquet"

CFG = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
GHE = CFG["who_ghe"]
GHE_YEARS: List[int] = [int(year) for year in GHE["years"]]
GHE_URL_TEMPLATE: str = GHE["daly_by_country_url"]
GHE_RELEASE: str = str(GHE["release"])

VALID_ISO3 = frozenset(country.alpha_3 for country in pycountry.countries)

# ------------------------ Disease dictionary ------------------------

# cause_code -> (cause name, ClinicalTrials.gov condition keywords)
DISEASE_PATTERNS: Dict[str, Tuple[str, List[str]]] = {
    "IHD":      ("Ischaemic heart disease",                ["ischaemic heart", "ischemic heart", "coronary artery", "myocardial infarction", "ihd", "ischaem", "ischem"]),
    "STROKE":   ("Stroke",                                 ["stroke", "cerebrovascular", "cbvd"]),
    "COPD":     ("Chronic obstructive pulmonary disease",  ["copd", "chronic obstructive pulmonary", "chronic obstructive"]),
    "LRI":      ("Lower respiratory infections",           ["lower respiratory infection", "lri", "pneumonia", "acute lower respiratory"]),
    "DIAB":     ("Diabetes mellitus",                      ["diabetes", "dm", "diabetes mellitus"]),
    "NEON":     ("Neonatal conditions",                    ["neonatal", "preterm", "birth asphyxia", "neonatal sepsis", "newborn"]),
    "LUNG_CA":  ("Trachea, bronchus and lung cancers",     ["lung cancer", "trachea", "bronchus", "trachea bronchus and lung"]),
    "ALZH":     ("Alzheimer's disease and dementias",      ["alzheimer", "alzheim", "dementia", "dementias"]),
    "DIARR":    ("Diarrhoeal diseases",                    ["diarrhoe", "diarrhoea", "diarrhea", "gastroenteritis"]),
    "KIDNEY":   ("Chronic kidney disease",                 ["chronic kidney", "ckd", "renal", "kidney disease"]),
    "TB":       ("Tuberculosis",                           ["tuberculosis", "tb"]),
    "HIV":      ("HIV/AIDS",                               ["hiv", "aids", "acquired immunodeficiency"]),
    "RTI":      ("Road traffic injuries",                  ["road traffic", "transport injury", "motor vehicle", "rti"]),
}

# cause_code -> (GHE cause code, GHE cause name as published by WHO)
GHE_CAUSES: Dict[str, Tuple[int, str]] = {
    "IHD":     (1130, "Ischaemic heart disease"),
    "STROKE":  (1140, "Stroke"),
    "COPD":    (1180, "Chronic obstructive pulmonary disease"),
    "LRI":     (390,  "Lower respiratory infections"),
    "DIAB":    (800,  "Diabetes mellitus"),
    "NEON":    (490,  "Neonatal conditions"),
    "LUNG_CA": (680,  "Trachea, bronchus, lung cancers"),
    "ALZH":    (950,  "Alzheimer disease and other dementias"),
    "DIARR":   (110,  "Diarrhoeal diseases"),
    "KIDNEY":  (1270, "Kidney diseases"),
    "TB":      (30,   "Tuberculosis"),
    "HIV":     (100,  "HIV/AIDS"),
    "RTI":     (1530, "Road injury"),
}

# ------------------------ HTTP ------------------------

def session_with_retries() -> requests.Session:
    retries = CFG.get("retries", {})
    s = requests.Session()
    retry = Retry(
        total=int(retries.get("total", 5)),
        backoff_factor=float(retries.get("backoff_factor", 0.5)),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.headers.update({"User-Agent": "Bayezian-EquityDashboard/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = session_with_retries()

# ------------------------ Helpers ------------------------

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def download_workbook(year: int, refresh: bool = False) -> Path:
    """Cache the WHO workbook locally; WHO republishes only with a new release."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"daly_bycountry_{year}_{GHE_RELEASE}.xlsx"
    if path.exists() and not refresh:
        return path
    response = SESSION.get(GHE_URL_TEMPLATE.format(year=year), timeout=300)
    response.raise_for_status()
    temporary = path.with_suffix(".xlsx.tmp")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    return path


def parse_workbook(path: Path, year: int) -> pd.DataFrame:
    """Read the 'All ages' sheet: both sexes, all ages, values in thousands."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["All ages"]
    sheet.reset_dimensions()  # the file under-reports its used range
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()

    iso_row = next(i for i, row in enumerate(rows) if row[6] == "ISO-3 Code")
    first_col = 7
    iso3 = [str(code).strip() if code else None for code in rows[iso_row][first_col:]]
    quality = rows[iso_row + 1][first_col:]
    population_row = next(row for row in rows if row[0] == "Persons" and str(row[3] or "").startswith("Population"))
    wanted = {code: cause for cause, (code, _) in GHE_CAUSES.items()}
    found = {}
    for row in rows:
        if row[0] == "Persons" and isinstance(row[1], (int, float)) and int(row[1]) in wanted:
            found[int(row[1])] = row
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"GHE {year} workbook lacks cause codes {sorted(missing)}")

    records = []
    for ghe_code, row in found.items():
        cause_code = wanted[ghe_code]
        for index, country in enumerate(iso3):
            value, population = row[first_col + index], population_row[first_col + index]
            if country is None or value is None:
                continue
            records.append({
                "country_iso3": country,
                "cause_code": cause_code,
                "cause_name": DISEASE_PATTERNS[cause_code][0],
                "year": year,
                "dalys": float(value) * 1_000,
                "population": float(population) * 1_000 if population is not None else None,
                "who_data_quality": int(quality[index]) if isinstance(quality[index], (int, float)) else None,
                "ghe_cause_code": ghe_code,
                "ghe_cause_name": GHE_CAUSES[cause_code][1],
            })
    return pd.DataFrame(records)

# ------------------------ Main ------------------------

def run_pipeline(refresh: bool = False) -> pd.DataFrame:
    frames = []
    for year in GHE_YEARS:
        path = download_workbook(year, refresh=refresh)
        frame = parse_workbook(path, year)
        print(f"[GHE {year}] {frame['country_iso3'].nunique()} countries, {len(frame)} rows")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["country_iso3"].isin(VALID_ISO3)].copy()
    out["daly_rate_per_100k"] = out["dalys"] / out["population"] * 100_000
    out["source"] = f"WHO Global Health Estimates {GHE_RELEASE}"
    out = out[[
        "country_iso3", "cause_code", "cause_name", "year", "dalys", "population",
        "daly_rate_per_100k", "who_data_quality", "ghe_cause_code", "ghe_cause_name", "source",
    ]].sort_values(["cause_code", "year", "country_iso3"]).reset_index(drop=True)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PARQUET.with_suffix(".parquet.tmp")
    out.to_parquet(temporary, index=False)
    temporary.replace(OUT_PARQUET)
    print(f"[OK] Wrote {OUT_PARQUET} (rows={len(out)})")
    return out


if __name__ == "__main__":
    try:
        run_pipeline(refresh="--refresh" in sys.argv)
    except Exception as exc:
        print(f"\n[ERROR] global_health_sources.py failed:\n{exc}")
        sys.exit(1)
