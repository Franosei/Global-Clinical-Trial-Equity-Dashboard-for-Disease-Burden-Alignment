"""
WHO GHO DALY pull for 13 fixed diseases with robust discovery/matching.

Output (Parquet): data/clean/who_daly_selected.parquet
Columns:
  country_iso3, year, cause_code, cause_name, value, value_low, value_high, metric
Run:
  python -m py.global_health_sources
"""

from __future__ import annotations
import os
import re
import unicodedata
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

# ------------------------ Paths & Config ------------------------

# repo_root = clinical-trial-equity/
REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "config" / "sources.yaml"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
OUT_PARQUET = CLEAN_DIR / "who_daly_selected.parquet"

def load_cfg() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_cfg()

# Years from config (inclusive range)
YEARS = list(range(int(CFG["years"]["start"]), int(CFG["years"]["end"]) + 1))

# WHO base url from config
BASE_GHO = str(CFG["who_gho"]["base_url"]).rstrip("/") + "/"

# Retry/backoff from config (with sensible defaults)
RETRY_TOTAL = int(CFG.get("retries", {}).get("total", 5))
RETRY_BACKOFF = float(CFG.get("retries", {}).get("backoff_factor", 0.5))

# Optional small raw dumps for provenance (kept minimal)
SAVE_RAW = bool(CFG.get("who_gho", {}).get("save_raw", False))

BOTH_SEXES_CODES = {"BTSX", "BTSX_AGEALL", "ALLSEX", "BTSX_AGESTANDARDIZED"}

# ------------------------ Disease dictionary ------------------------

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

# indicator-code hints (expanded). These help catch terse SA_ codes.
CODE_HINTS = {
    "IHD":     ["IHD", "SA_0000001425"],
    "STROKE":  ["STROKE", "CBVD", "SA_0000001689"],
    "COPD":    ["COPD", "SA_0000001426"],
    "LRI":     ["LRI", "LRTI", "SA_0000001431"],
    "DIAB":    ["DIAB", "SA_0000001421"],
    "NEON":    ["NEON", "NEONATAL", "SA_0000001419"],
    "LUNG_CA": ["LUNG", "TRACHEA", "BRONCHUS", "SA_0000001420"],
    "ALZH":    ["ALZH", "DEMENT", "SA_0000001418"],
    "DIARR":   ["DIARR", "WSH_30", "WSH_30_WAT", "WSH_30_SAN", "WSH_30_HYG", "SA_0000001833"],
    "KIDNEY":  ["CKD", "KIDNEY", "RENAL", "SA_0000001430"],
    "TB":      ["TB", "TUBERC", "SA_0000001422"],
    "HIV":     ["HIV", "AIDS", "SA_0000001423"],
    "RTI":     ["RTI", "ROAD", "TRAFFIC", "SA_0000001433", "OCC_6", "OCC_8"],
}

# ------------------------ HTTP ------------------------

def session_with_retries(total: int = RETRY_TOTAL, backoff: float = RETRY_BACKOFF) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total, connect=total, read=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.headers.update({"User-Agent": "Bayezian-EquityDashboard/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

SESSION = session_with_retries()

def gho_get_json(path: str, params: Optional[Dict[str, str]] = None) -> dict:
    url = BASE_GHO + path.lstrip("/")
    r = SESSION.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()

# ------------------------ Normalisation helpers ------------------------

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def contains_any(series: pd.Series, needles: List[str], normalised: bool = False) -> pd.Series:
    if series.empty:
        return pd.Series(False, index=series.index)
    s = series.fillna("")
    if not normalised:
        s = s.astype(str).apply(norm_text)
        needles = [norm_text(n) for n in needles]
    mask = pd.Series(False, index=s.index)
    for n in needles:
        if not n:
            continue
        mask = mask | s.str.contains(re.escape(n), na=False)
    return mask

# ------------------------ WHO DALY catalogue & matching ------------------------

def pull_indicator_catalogue() -> pd.DataFrame:
    rows, skip = [], 0
    while True:
        data = gho_get_json("Indicator", params={"$skip": str(skip)})
        vals = data.get("value", [])
        if not vals:
            break
        rows.extend(vals)
        skip += len(vals)
    cat = pd.DataFrame(rows)
    if cat.empty:
        return cat
    # Defensive typing
    cat["IndicatorName"] = cat["IndicatorName"].astype(str)
    cat["IndicatorCode"] = cat["IndicatorCode"].astype(str)
    cat["IndicatorName_n"] = cat["IndicatorName"].apply(norm_text)
    cat["IndicatorCode_u"] = cat["IndicatorCode"].str.upper()
    return cat[["IndicatorCode","IndicatorName","IndicatorName_n","IndicatorCode_u"]]

def match_indicators(cat: pd.DataFrame) -> pd.DataFrame:
    """
    Return mapping rows: indicator_code, indicator_name, cause_code, cause_name.

    Rules:
      • accept indicators whose NAME mentions 'daly' OR whose CODE starts with 'SA_' (GHE cause lists often do)
      • AND name/code matches any disease keyword OR code-hint list for that cause
    """
    if cat.empty:
        return pd.DataFrame(columns=["indicator_code","indicator_name","cause_code","cause_name"])

    is_dalyish = contains_any(cat["IndicatorName_n"], ["daly"]) | cat["IndicatorCode_u"].str.startswith("SA_")

    mappings = []
    for cause_code, (cause_name, keywords) in DISEASE_PATTERNS.items():
        name_hit = contains_any(cat["IndicatorName_n"], keywords)
        code_hit = contains_any(cat["IndicatorCode_u"], CODE_HINTS.get(cause_code, [cause_code]), normalised=True)
        match = cat[is_dalyish & (name_hit | code_hit)].drop_duplicates(subset=["IndicatorCode"])
        for _, row in match.iterrows():
            mappings.append({
                "indicator_code": row["IndicatorCode"],
                "indicator_name": row["IndicatorName"],
                "cause_code": cause_code,
                "cause_name": cause_name,
            })

    map_df = pd.DataFrame(mappings).drop_duplicates(subset=["indicator_code","cause_code"]).reset_index(drop=True)

    # Debug table: what matched per cause
    print("\n[MATCH DEBUG] Indicators matched per disease:")
    if map_df.empty:
        print("  (none)")
    else:
        dbg = (map_df.groupby(["cause_code","cause_name"])["indicator_code"]
                     .apply(lambda x: ", ".join(sorted(set(map(str, x))))).reset_index())
        for _, r in dbg.iterrows():
            print(f"  {r['cause_code']:6s}  {r['cause_name']}: {r['indicator_code']}")

    # Missing causes warning
    want = set(DISEASE_PATTERNS.keys())
    got = set(map_df["cause_code"].unique())
    miss = sorted(want - got)
    if miss:
        print("\n[WARN] No DALY indicators found for causes:", ", ".join(miss))
    return map_df

# ------------------------ Fetch & tidy ------------------------

def fetch_indicator_all(code: str) -> pd.DataFrame:
    rows, skip = [], 0
    while True:
        data = gho_get_json(code, params={"$skip": str(skip)})
        vals = data.get("value", [])
        if not vals:
            break
        rows.extend(vals)
        skip += len(vals)
    df = pd.DataFrame(rows)
    if SAVE_RAW and not df.empty:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_json(RAW_DIR / f"who_{code}.json", orient="records", lines=False)
    return df

def parse_year(row: pd.Series) -> Optional[int]:
    # Typical fields: "TimeDim", sometimes "TimeDimensionBegin" or "TimeDimensionValue"
    for fld in ("TimeDim", "TimeDimensionBegin", "TimeDimensionValue"):
        if fld in row and pd.notna(row[fld]):
            s = str(row[fld])
            try:
                return int(s[:4])
            except Exception:
                continue
    return None

def prefer_total_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per (SpatialDim, year, IndicatorCode), preferring:
      1) Both sexes + no extra dims
      2) All-null Dim* (implicit totals)
      3) Otherwise the least stratified row
    """
    if df.empty:
        return df.copy()

    # Year
    if "TimeDim" in df.columns:
        df["year"] = pd.to_numeric(df["TimeDim"], errors="coerce")
    else:
        df["year"] = df.apply(parse_year, axis=1)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    dim_cols = [c for c in df.columns if c.startswith("Dim")]
    df["_allnull"] = df[dim_cols].isna().all(axis=1) if dim_cols else True

    def is_both(row):
        if "Dim1Type" in row and str(row["Dim1Type"]).upper() == "SEX":
            d1 = str(row.get("Dim1","")).upper()
            if d1 in BOTH_SEXES_CODES:
                # must not have other dims populated
                for c in ["Dim2","Dim3","Dim4","Dim5"]:
                    if c in row and pd.notna(row[c]):
                        return False
                return True
        return False

    df["_both"] = df.apply(is_both, axis=1)
    df["_nn"] = df[dim_cols].notna().sum(axis=1) if dim_cols else 0
    # Lower priority value wins in sort; so set "best" to 0
    df["_prio"] = 2 + df["_nn"]
    df.loc[df["_allnull"], "_prio"] = 1
    df.loc[df["_both"], "_prio"] = 0

    df = df.sort_values(["_prio"])
    keep_cols = [c for c in df.columns if c not in ["_allnull","_both","_nn","_prio"]]
    collapsed = (df.groupby(["SpatialDim","year","IndicatorCode"], as_index=False)
                   .first()[keep_cols])
    return collapsed

def tidy_who(df: pd.DataFrame, cause_code: str, cause_name: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    # Numeric coercion
    for c in ("NumericValue","Low","High"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df.rename(columns={
        "SpatialDim": "country_iso3",
        "NumericValue": "value",
        "Low": "value_low",
        "High": "value_high",
    })

    # Filter to configured years and non-null essentials
    out = out.dropna(subset=["country_iso3","year","value"])
    out["year"] = out["year"].astype(int)
    out = out[out["year"].isin(YEARS)]

    # Country: force ISO3 upper & drop non-ISO3 codes (e.g., regions)
    out["country_iso3"] = out["country_iso3"].astype(str).str.upper().str[:3]
    out = out[out["country_iso3"].str.fullmatch(r"[A-Z]{3}")]

    # Metric classification from indicator name
    name_series = out.get("IndicatorName", pd.Series(index=out.index, dtype="object")).astype(str).apply(norm_text)
    metric = pd.Series("DALYs (count)", index=out.index, dtype="object")
    is_rate = name_series.str.contains("per 100", na=False) | name_series.str.contains("age standard", na=False)
    metric.loc[is_rate] = "DALYs per 100k population"

    out["cause_code"] = cause_code
    out["cause_name"] = cause_name
    out["metric"] = metric

    cols = ["country_iso3","year","cause_code","cause_name","value","value_low","value_high","metric"]
    return out[cols]

# ------------------------ Main pipeline ------------------------

def run_pipeline() -> pd.DataFrame:
    # Ensure output folder exists
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_RAW:
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Pull & match DALY indicators
    cat = pull_indicator_catalogue()
    if cat.empty:
        print("[ERROR] WHO indicator catalogue is empty.")
        return pd.DataFrame()

    mapping = match_indicators(cat)
    if mapping.empty:
        print("[ERROR] No indicators matched disease patterns.")
        return pd.DataFrame()

    # 2) Fetch all indicators and tidy
    frames: List[pd.DataFrame] = []
    for _, row in mapping.iterrows():
        code = str(row["indicator_code"])
        try:
            raw = fetch_indicator_all(code)
        except requests.HTTPError as e:
            print(f"[WARN] fetch failed for {code}: {e}")
            continue
        if raw.empty:
            continue
        reduced = prefer_total_stratum(raw)
        if reduced.empty:
            continue
        tidy = tidy_who(reduced, str(row["cause_code"]), str(row["cause_name"]))
        if not tidy.empty:
            frames.append(tidy)

    if not frames:
        print("[WARN] No rows produced after fetching.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # 3) If multiple indicators map to the same cause, keep both metrics if different,
    #    else prefer *counts* (more universal for ER denominator)
    df["_rank"] = (df["metric"] == "DALYs per 100k population").astype(int)  # counts=0, rate=1
    df = (df.sort_values(["country_iso3","year","cause_code","_rank"])
            .drop_duplicates(subset=["country_iso3","year","cause_code","metric"], keep="first")
            .drop(columns=["_rank"]))

    # 4) Final sanity: drop impossible values and years outside config (defensive)
    df = df[df["value"].notna() & np.isfinite(df["value"])]
    df = df[df["year"].isin(YEARS)]

    # 5) Coverage log
    have = sorted(df["cause_code"].unique())
    missing = sorted(set(DISEASE_PATTERNS.keys()) - set(have))
    print("\n[COVERAGE] causes captured:", ", ".join(have) if have else "(none)")
    if missing:
        print("[WARN] missing causes:", ", ".join(missing))

    # 6) Write exactly one Parquet
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[OK] Wrote {OUT_PARQUET} (rows={len(df)})")

    # Small preview (safe)
    with pd.option_context("display.max_columns", None, "display.width", 1200):
        print(df.head(12).to_string(index=False))
    return df

if __name__ == "__main__":
    run_pipeline()
