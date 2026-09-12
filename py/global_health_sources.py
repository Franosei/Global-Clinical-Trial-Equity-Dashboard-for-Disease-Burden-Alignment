# py/global_health_sources.py
"""
WHO GHO DALY pull for 13 fixed diseases with robust discovery/matching.

Output (Parquet): data/clean/who_daly_selected.parquet

Columns (final aggregated across years):
  country_iso3, cause_code, cause_name, value, value_low, value_high, metric

Where:
  • value = SUM over years of DALYs per 100k (cumulative per-100k across configured YEARS)

Run:
  python -m py.global_health_sources
"""

from __future__ import annotations
import re
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

# ------------------------ Paths & Config ------------------------

REPO_ROOT   = Path(__file__).resolve().parents[1]
CFG_PATH    = REPO_ROOT / "config" / "sources.yaml"
DATA_DIR    = REPO_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
CLEAN_DIR   = DATA_DIR / "clean"
OUT_PARQUET = CLEAN_DIR / "who_daly_selected.parquet"

def load_cfg() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_cfg()

_year_end     = CFG["years"]["end"]
YEAR_END      = datetime.now(timezone.utc).year if str(_year_end).lower() == "current" else int(_year_end)
YEARS         = list(range(int(CFG["years"]["start"]), YEAR_END + 1))
BASE_GHO      = str(CFG["who_gho"]["base_url"]).rstrip("/") + "/"
RETRY_TOTAL   = int(CFG.get("retries", {}).get("total", 5))
RETRY_BACKOFF = float(CFG.get("retries", {}).get("backoff_factor", 0.5))
SAVE_RAW      = bool(CFG.get("who_gho", {}).get("save_raw", False))

BOTH_SEXES_CODES = {"BTSX", "ALLSEX", "BTSX_AGEALL", "BTSX_AGESTANDARDIZED"}

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

# Broad code hints
CODE_HINTS = {
    "IHD":     ["IHD", "SA_0000001425"],
    "STROKE":  ["STROKE", "CBVD", "SA_0000001689"],
    "COPD":    ["COPD", "SA_0000001426"],
    "LRI":     ["LRI", "LRTI", "SA_0000001431"],
    "DIAB":    ["DIAB", "SA_0000001421"],
    "NEON":    ["NEON", "NEONATAL", "SA_0000001419"],
    "LUNG_CA": ["LUNG", "TRACHEA", "BRONCHUS", "SA_0000001420"],
    "ALZH":    ["ALZH", "DEMENT", "SA_0000001418"],
    "DIARR":   ["DIARR", "SA_0000001833"],
    "KIDNEY":  ["CKD", "KIDNEY", "RENAL", "SA_0000001430"],
    "TB":      ["TB", "TUBERC", "SA_0000001422"],
    "HIV":     ["HIV", "AIDS", "SA_0000001423"],
    "RTI":     ["RTI", "ROAD", "TRAFFIC", "SA_0000001433"],
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

def parse_year(row: pd.Series) -> Optional[int]:
    for fld in ("TimeDim", "TimeDimensionBegin", "TimeDimensionValue"):
        if fld in row and pd.notna(row[fld]):
            s = str(row[fld])
            try:
                return int(s[:4])
            except Exception:
                continue
    return None

# ---- relaxed totals logic ----
TOTAL_CODES = {
    "SEX": {"BTSX", "ALLSEX", "BTSX_AGEALL", "BTSX_AGESTANDARDIZED"},
    "AGEGROUP": {"AGEALL", "AGESTANDARDIZED"},
    "INCOMELEVEL": {"ALL"},
    "URBANRURAL": {"ALL"},
    "AREAS": {"ALL"},
    "RESIDENCE": {"ALL"},
    "EDUCATION": {"ALL"},
}

def is_total_like_row(row) -> bool:
    for i in range(1, 6):
        t = str(row.get(f"Dim{i}Type", "") or "").upper()
        v = str(row.get(f"Dim{i}", "") or "").upper()
        if (t, v) == ("", ""):
            continue
        if t in TOTAL_CODES and v in TOTAL_CODES[t]:
            continue
        if v in {"ALL", "TOTAL"}:
            continue
        return False
    return True

def prefer_total_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per (SpatialDim, year, IndicatorCode), preferring:
      - "total-like" rows (both sexes; AGEALL over AGESTANDARDIZED; other dims ALL/TOTAL)
      - then both-sexs
      - then AGEALL > AGESTANDARDIZED
      - then fewer non-null dims
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

    df["_total_like"] = df.apply(is_total_like_row, axis=1)
    df["_both_sex"] = (df.get("Dim1Type", "").astype(str).str.upper() == "SEX") & \
                      (df.get("Dim1", "").astype(str).str.upper().isin(BOTH_SEXES_CODES))
    df["_age_all"] = (df.get("Dim2Type", "").astype(str).str.upper() == "AGEGROUP") & \
                     (df.get("Dim2", "").astype(str).str.upper().eq("AGEALL"))
    df["_age_std"] = (df.get("Dim2Type", "").astype(str).str.upper() == "AGEGROUP") & \
                     (df.get("Dim2", "").astype(str).str.upper().eq("AGESTANDARDIZED"))
    df["_nn"] = df[dim_cols].notna().sum(axis=1) if dim_cols else 0

    df["_prio"] = np.where(df["_total_like"], 0, 10)
    df = df.sort_values(
        by=["_prio", "_both_sex", "_age_all", "_age_std", "_nn"],
        ascending=[True, False, True, True, True],
    )

    keep_cols = [c for c in df.columns if c not in ["_total_like","_both_sex","_age_all","_age_std","_nn","_prio"]]
    return (df.groupby(["SpatialDim","year","IndicatorCode"], as_index=False)
              .first()[keep_cols])

# ------------------------ Catalogue & matching ------------------------

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
    cat["IndicatorName"] = cat["IndicatorName"].astype(str)
    cat["IndicatorCode"] = cat["IndicatorCode"].astype(str)
    cat["IndicatorName_n"] = cat["IndicatorName"].apply(norm_text)
    cat["IndicatorCode_u"] = cat["IndicatorCode"].str.upper()
    return cat[["IndicatorCode","IndicatorName","IndicatorName_n","IndicatorCode_u"]]

def _infer_component(row) -> Optional[str]:
    text = (str(row["IndicatorName"]) + " " + str(row["IndicatorCode_u"])).upper()
    if "DALY" in text or row["IndicatorCode_u"].startswith("SA_"):
        return "DALY"
    if re.search(r"\bYLL\b", text):
        return "YLL"
    if re.search(r"\bYLD\b", text):
        return "YLD"
    return None

def match_indicators(cat: pd.DataFrame) -> pd.DataFrame:
    """
    Map indicators to our causes and tag component ∈ {DALY, YLL, YLD}.
    Stricter: require explicit DALY/YLL/YLD hints; SA_ treated as DALY.
    """
    if cat.empty:
        return pd.DataFrame(columns=["indicator_code","indicator_name","cause_code","cause_name","component"])

    mappings = []
    for cause_code, (cause_name, keywords) in DISEASE_PATTERNS.items():
        name_hit = contains_any(cat["IndicatorName_n"], keywords)
        code_hit = contains_any(cat["IndicatorCode_u"], CODE_HINTS.get(cause_code, [cause_code]), normalised=True)
        subset = cat[name_hit | code_hit].drop_duplicates(subset=["IndicatorCode"])
        if subset.empty:
            continue
        subset = subset.copy()
        subset["component"] = subset.apply(_infer_component, axis=1)
        subset = subset[subset["component"].notna()]
        for _, r in subset.iterrows():
            mappings.append({
                "indicator_code": r["IndicatorCode"],
                "indicator_name": r["IndicatorName"],
                "cause_code": cause_code,
                "cause_name": cause_name,
                "component": str(r["component"]),
            })

    map_df = pd.DataFrame(mappings).drop_duplicates(subset=["indicator_code","cause_code"]).reset_index(drop=True)

    print("\n[MATCH CHECK] Indicators matched per disease/component:")
    if map_df.empty:
        print("  (none)")
    else:
        dbg = (map_df.groupby(["cause_code","cause_name","component"])["indicator_code"]
                     .apply(lambda x: ", ".join(sorted(set(map(str, x))))).reset_index())
        for _, r in dbg.iterrows():
            print(f"  {r['cause_code']:6s}  {r['cause_name']:<45s} [{r['component']}]: {r['indicator_code']}")
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

def tidy_who(df: pd.DataFrame, cause_code: str, cause_name: str, component: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    # Coerce numerics
    for c in ("NumericValue","Low","High"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df.rename(columns={
        "SpatialDim": "country_iso3",
        "NumericValue": "value",
        "Low": "value_low",
        "High": "value_high",
    })

    # Year
    if "year" not in out.columns:
        out["year"] = out.apply(parse_year, axis=1)
    out = out.dropna(subset=["country_iso3","year","value"])
    out["year"] = out["year"].astype(int)
    out = out[out["year"].isin(YEARS)]

    # ISO3
    out["country_iso3"] = out["country_iso3"].astype(str).str.upper().str[:3]
    out = out[out["country_iso3"].str.fullmatch(r"[A-Z]{3}")]

    # -------- Robust unit detection (NO double-normalization) --------
    name_n = out.get("IndicatorName", pd.Series(index=out.index, dtype="object")).astype(str).str.lower()
    units  = out.get("Units",         pd.Series(index=out.index, dtype="object")).astype(str).str.lower()
    vtype  = out.get("ValueType",     pd.Series(index=out.index, dtype="object")).astype(str).str.lower()
    code_u = out.get("IndicatorCode", pd.Series(index=out.index, dtype="object")).astype(str).str.upper()

    # WHO convention: SA_* series for DALY/YLL/YLD are rates per 100k
    is_sa = code_u.str.startswith("SA_")

    # Rate if: marked rate OR mentions per-100k OR age-standardized wording OR SA_ code
    is_rate = (
        is_sa |
        vtype.str.contains("rate", na=False) |
        units.str.contains(r"per 100 ?000|per 100000|/100 ?000", regex=True, na=False) |
        name_n.str.contains("per 100", na=False) |
        name_n.str.contains("age[- ]?standard", na=False)
    )

    # Rare per-capita cases (rate-like but not per-100k and not SA_)
    is_per_capita = is_rate & ~(
        units.str.contains(r"per 100 ?000|per 100000|/100 ?000", regex=True, na=False) |
        name_n.str.contains("per 100", na=False) |
        is_sa
    )

    out["cause_code"] = cause_code
    out["cause_name"] = cause_name
    out["component"]  = component
    out["unit_tag"]   = np.where(is_per_capita, "PER_CAPITA", np.where(is_rate, "PER_100K", "COUNT"))
    out["metric"]     = np.where(is_rate, "RATE", "COUNT")  # temp tag

    keep = ["country_iso3","year","cause_code","cause_name","value","value_low","value_high",
            "metric","IndicatorName","component","unit_tag","IndicatorCode"]
    keep = [c for c in keep if c in out.columns]
    return out[keep]

# ------------------------ Population & unit unification ------------------------

WB_POP_URL = "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"

ISO3_REMAP = {
    # add special cases if needed
}

def fetch_population(countries_iso3: List[str], years: List[int]) -> pd.DataFrame:
    rows, page, pages = [], 1, 1
    while page <= pages:
        r = SESSION.get(WB_POP_URL, params={"format": "json", "per_page": 20000, "page": page}, timeout=120)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) == 2:
            meta, vals = data[0], data[1]
            pages = int(meta.get("pages", 1))
            rows.extend(vals or [])
        page += 1
    if not rows:
        return pd.DataFrame(columns=["country_iso3","year","population"])
    df = pd.DataFrame(rows)
    df = df.rename(columns={"countryiso3code":"country_iso3", "date":"year", "value":"population"})
    df["country_iso3"] = df["country_iso3"].astype(str).str.upper().replace(ISO3_REMAP)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["country_iso3","year","population"])
    df["year"] = df["year"].astype(int)
    df = df[df["country_iso3"].isin(set(countries_iso3))]
    df = df[df["year"].isin(set(years))]
    return df[["country_iso3","year","population"]]

def to_rate_per100k(df: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize rows to DALYs per 100k ONLY when needed:
      - PER_100K: keep original 'value'
      - PER_CAPITA: value * 100000
      - COUNT: (value / population) * 100000
    """
    if df.empty:
        df["daly_per_100k"] = pd.Series(dtype="float64")
        return df

    out = df.merge(pop, on=["country_iso3","year"], how="left")

    out["daly_per_100k"] = np.nan

    # Already per 100k (includes SA_* series) -> do NOT re-normalize
    mask_100k = out["unit_tag"].eq("PER_100K")
    out.loc[mask_100k, "daly_per_100k"] = out.loc[mask_100k, "value"]

    # Per capita -> scale up
    mask_pc = out["unit_tag"].eq("PER_CAPITA")
    out.loc[mask_pc, "daly_per_100k"] = out.loc[mask_pc, "value"] * 100_000

    # Counts -> divide by pop
    mask_cnt = out["unit_tag"].eq("COUNT")
    with np.errstate(divide="ignore", invalid="ignore"):
        out.loc[mask_cnt, "daly_per_100k"] = (out.loc[mask_cnt, "value"] / out.loc[mask_cnt, "population"]) * 100_000

    return out

def build_daly_total_per100k(df: pd.DataFrame) -> pd.DataFrame:
    key = ["country_iso3","year","cause_code","cause_name"]
    daly = (df[df["component"].eq("DALY")]
              .groupby(key, as_index=False)
              .agg({"daly_per_100k":"first"}))
    comp = (df[df["component"].isin(["YLL","YLD"])]
              .groupby(key, as_index=False)
              .agg({"daly_per_100k":"sum"}))
    out = daly.merge(comp, on=key, how="outer", suffixes=("_daly","_yll_yld"))
    out["daly_per_100k"] = out["daly_per_100k_daly"].fillna(out["daly_per_100k_yll_yld"])
    return out[key + ["daly_per_100k"]]

# ------------------------ Main ------------------------

def run_pipeline() -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_RAW:
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1) catalogue & match (DALY + YLL + YLD)
    cat = pull_indicator_catalogue()
    if cat.empty:
        print("[ERROR] WHO indicator catalogue is empty.")
        return pd.DataFrame()
    mapping = match_indicators(cat)
    if mapping.empty:
        print("[ERROR] No indicators matched disease patterns.")
        return pd.DataFrame()

    # 2) fetch -> collapse (relaxed) -> tidy
    frames: List[pd.DataFrame] = []
    for _, row in mapping.iterrows():
        code = str(row["indicator_code"])
        component = str(row["component"])
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
        tidy = tidy_who(reduced, str(row["cause_code"]), str(row["cause_name"]), component)
        if not tidy.empty:
            frames.append(tidy)

    if not frames:
        print("[WARN] No rows produced after fetching.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # 3) De-dup within (country,year,cause,component): prefer RATE over COUNT
    df["_rank_rate"] = (df["metric"].astype(str).str.upper() == "RATE").astype(int)
    df = (df.sort_values(["country_iso3","year","cause_code","_rank_rate"], ascending=[True, True, True, False])
            .drop_duplicates(subset=["country_iso3","year","cause_code","component"], keep="first")
            .drop(columns=["_rank_rate"]))

    # 4) Normalize to per-100k ONLY when needed (COUNT/PER_CAPITA)
    countries = sorted(df["country_iso3"].unique().tolist())
    pop = fetch_population(countries, YEARS)

    # Diagnostics: population coverage & unit tags
    print(f"\n[POP] Countries in WHO rows: {len(set(df['country_iso3']))} | with WB pop: {len(set(pop['country_iso3']))}")
    tmp_cov = df.merge(pop, on=["country_iso3","year"], how="left", indicator=True)
    need_pop = tmp_cov["unit_tag"].eq("COUNT")
    missing_pop_rows = int(((tmp_cov["_merge"] == "left_only") & need_pop).sum())
    print(f"[POP] COUNT rows missing population for normalization: {missing_pop_rows}")

    df = to_rate_per100k(df, pop)

    # 5) Build DALY totals (prefer DALY; else YLL+YLD) in per-100k space
    daly = build_daly_total_per100k(df)

    # QA: inspect yearly magnitudes (should look like hundreds to thousands per 100k)
    if not daly.empty:
        print("\n[QA] sample yearly DALY per 100k (top 10):")
        try:
            print(daly.sort_values("daly_per_100k", ascending=False).head(10).to_string(index=False))
        except Exception:
            pass

    by_unit = df.groupby("unit_tag", dropna=False)["value"].count()
    print("\n[QA] rows by unit_tag before/after normalization (count of original 'value'):\n", by_unit)

    # 6) Aggregate across years -> one row per country & cause (sum of per-year rates)
    agg = (daly.rename(columns={"daly_per_100k": "value"})
               .groupby(["country_iso3", "cause_code", "cause_name"], as_index=False)
               .agg({"value": "sum"}))

    # Final schema
    agg["metric"] = "DALYs per 100k population (sum across years)"
    agg["value_low"] = np.nan
    agg["value_high"] = np.nan
    out = agg[["country_iso3","cause_code","cause_name","value","value_low","value_high","metric"]]

    out.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[OK] Wrote {OUT_PARQUET} (rows={len(out)})")
    with pd.option_context("display.max_columns", None, "display.width", 1200):
        print(out.head(12).to_string(index=False))
    return out

if __name__ == "__main__":
    run_pipeline()
