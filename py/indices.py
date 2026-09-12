# py/indices.py
"""
Interval indices (2000 to 2005, 2006 to 2010, 2011 to 2015, 2016 to 2020, 2021 to 2025):
Equity Ratio, Burden-Adjusted Trial Density, Gini & Theil

Inputs:
  data/clean/who_daly_selected.parquet
     • country_iso3, cause_code, cause_name, value, value_low, value_high, metric
       (WHO burden already unified to DALYs per 100k and SUMMED across all years in config)

  data/clean/ctgov_sites.parquet
     • country_iso3, year, cause_code, cause_name, phase, trial_sites

Output (gold, dashboard-ready):
  data/gold/country_interval_disease.parquet

Columns:
  country_iso3, interval_start, interval_end, interval_label,
  cause_code, cause_name,
  trial_sites, burden_dalys,
  trial_share, burden_share, equity_ratio,
  trial_density_per_k_dalys,
  gini_er_weighted, theil_T,
  metric_kind, metric_label

Method notes:
  • WHO burden is country×cause (no year). We use it as a stable severity weight.
  • CT trial sites are aggregated into fixed intervals:
        2000 to 2005, 2006 to 2010, 2011 to 2015, 2016 to 2020, 2021 to 2025
  • Universe is ALL (country×cause) that appear in WHO or CT.gov, crossed with all intervals.
    Missing trials/burden remain NaN in their columns; calculations treat missing trials/burden as 0 only for math.
"""

from __future__ import annotations
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pycountry

VALID_COUNTRY_ISO3 = frozenset(country.alpha_3 for country in pycountry.countries)

# ------------------------ Paths ------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
GOLD_DIR  = DATA_DIR / "gold"

IN_WHO    = CLEAN_DIR / "who_daly_selected.parquet"
IN_CT     = CLEAN_DIR / "ctgov_sites.parquet"
OUT_GOLD  = GOLD_DIR  / "country_interval_disease.parquet"

# ------------------------ Utils ------------------------

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def _info(msg: str) -> None:
    print(f"[INFO] {msg}")

def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Gini of values weighted by weights (both non-negative)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v, w = v[m], w[m]
    if v.size == 0 or w.sum() <= 0:
        return np.nan
    mu = np.average(v, weights=w)
    if not np.isfinite(mu) or mu <= 0:
        return np.nan
    order = np.argsort(v)
    v, w = v[order], w[order]
    vw   = v * w
    c_vw = np.cumsum(vw)
    cw   = np.cumsum(w)
    W    = cw[-1]
    s    = (w * (np.concatenate(([0.0], c_vw[:-1])) + c_vw)).sum()
    return float(1.0 - s / (mu * W * W))

def theil_T(trial_share: np.ndarray, burden_share: np.ndarray) -> float:
    """Theil T of allocation (trial_share vs burden_share) for a given (interval, cause)."""
    p = np.asarray(trial_share, dtype=float)
    b = np.asarray(burden_share, dtype=float)
    # Equity is identifiable only where measured burden is positive.
    m = np.isfinite(p) & np.isfinite(b) & (p >= 0) & (b > 0)
    p, b = p[m], b[m]
    if p.size == 0:
        return np.nan
    ps, bs = p.sum(), b.sum()
    if ps <= 0 or bs <= 0:
        return np.nan
    # Normalize explicitly to probability simplex
    p = p / ps
    b = b / bs
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 0, p / b, np.nan)  # r undefined where b==0
        term = np.where((p > 0) & np.isfinite(r), p * np.log(r), 0.0)
    # If all b==0, earlier guard returned NaN
    return float(np.nansum(term))

# Fixed interval mapping: 2000 to 2005, 2006 to 2010, 2011 to 2015, 2016 to 2020, 2021 to 2025
_INTERVALS = [
    (2000, 2005),
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2025),
]
while _INTERVALS[-1][1] < datetime.now(timezone.utc).year:
    _start = _INTERVALS[-1][1] + 1
    _INTERVALS.append((_start, _start + 4))

def assign_interval(year: int) -> tuple[int, int, str] | None:
    """Return (start, end, label) for year in fixed intervals; None if outside all."""
    for s, e in _INTERVALS:
        if s <= year <= e:
            return s, e, f"{s} to {e}"
    return None

# ------------------------ Pipeline ------------------------

def run_indices() -> pd.DataFrame:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_WHO.exists():
        raise FileNotFoundError(f"Missing WHO input: {IN_WHO}")
    if not IN_CT.exists():
        raise FileNotFoundError(f"Missing CT.gov input: {IN_CT}")

    who = pd.read_parquet(IN_WHO)
    ct  = pd.read_parquet(IN_CT)

    # WHO extracts can contain three-letter regional aggregates (AFR, GLO,
    # etc.). They are not countries and must not enter country-level shares.
    who = who[who["country_iso3"].astype(str).isin(VALID_COUNTRY_ISO3)].copy()
    ct = ct[ct["country_iso3"].astype(str).isin(VALID_COUNTRY_ISO3)].copy()

    # ---- WHO: country×cause (no year). Keep original burden column; compute standardized.
    for c in ["country_iso3", "cause_code", "cause_name"]:
        who[c] = who[c].astype(str)
    who["burden_dalys"] = pd.to_numeric(who["value"], errors="coerce")
    who["metric_kind"]  = "rate_per_100k"
    who["metric_label"] = who.get("metric", "DALYs per 100k population")
    who = who[["country_iso3","cause_code","cause_name","burden_dalys","metric_kind","metric_label"]]

    # ---- CT.gov: clean, map to intervals, aggregate across phases
    for c in ["country_iso3", "cause_code", "cause_name", "phase"]:
        ct[c] = ct[c].astype(str)
    ct["year"]        = pd.to_numeric(ct["year"], errors="coerce").astype("Int64")
    ct["trial_sites"] = pd.to_numeric(ct["trial_sites"], errors="coerce")
    ct = ct.dropna(subset=["year"]).copy()
    ct["year"] = ct["year"].astype(int)

    ints = ct["year"].apply(assign_interval)
    ct["interval_tuple"] = ints
    ct = ct[ct["interval_tuple"].notna()].copy()

    if ct.empty and who.empty:
        _warn("Both WHO and CT.gov inputs are empty after cleaning.")
        out = pd.DataFrame(columns=[
            "country_iso3","interval_start","interval_end","interval_label",
            "cause_code","cause_name",
            "trial_sites","burden_dalys",
            "trial_share","burden_share","equity_ratio",
            "trial_density_per_k_dalys",
            "gini_er_weighted","theil_T",
            "metric_kind","metric_label"
        ])
        out.to_parquet(OUT_GOLD, index=False)
        _info(f"Wrote {OUT_GOLD} (rows=0)")
        return out

    if not ct.empty:
        ct["interval_start"] = ct["interval_tuple"].apply(lambda x: x[0])
        ct["interval_end"]   = ct["interval_tuple"].apply(lambda x: x[1])
        ct["interval_label"] = ct["interval_tuple"].apply(lambda x: x[2])

        ct_int = (ct.groupby(
                    ["country_iso3", "interval_start", "interval_end", "interval_label",
                     "cause_code", "cause_name"], as_index=False)
                  .agg(trial_sites=("trial_sites", "sum")))
    else:
        # Create an empty ct_int with expected columns to allow merging
        ct_int = pd.DataFrame(columns=[
            "country_iso3","interval_start","interval_end","interval_label",
            "cause_code","cause_name","trial_sites"
        ])

    # ---- Build the FULL universe: (countries ∪) × (causes ∪) × intervals
    countries = pd.Series(pd.unique(pd.concat([who["country_iso3"], ct_int["country_iso3"]], ignore_index=True))).dropna()
    causes    = pd.DataFrame(pd.unique(pd.concat([who[["cause_code","cause_name"]].apply(tuple, axis=1),
                                                  ct_int[["cause_code","cause_name"]].apply(tuple, axis=1)],
                                                 ignore_index=True)),
                             columns=["pair"]).dropna()
    if causes.empty:
        _warn("No causes found in WHO or CT.gov; output will be empty.")
        out = pd.DataFrame(columns=[
            "country_iso3","interval_start","interval_end","interval_label",
            "cause_code","cause_name",
            "trial_sites","burden_dalys",
            "trial_share","burden_share","equity_ratio",
            "trial_density_per_k_dalys",
            "gini_er_weighted","theil_T",
            "metric_kind","metric_label"
        ])
        out.to_parquet(OUT_GOLD, index=False)
        _info(f"Wrote {OUT_GOLD} (rows=0)")
        return out

    causes[["cause_code","cause_name"]] = pd.DataFrame(causes["pair"].tolist(), index=causes.index)
    causes = causes.drop(columns=["pair"])

    intervals_df = pd.DataFrame(
        [(s, e, f"{s} to {e}") for (s, e) in _INTERVALS],
        columns=["interval_start","interval_end","interval_label"]
    )

    universe = (
        pd.DataFrame({"country_iso3": countries})
        .assign(key=1)
        .merge(causes.assign(key=1), on="key", how="outer")
        .merge(intervals_df.assign(key=1), on="key", how="outer")
        .drop(columns=["key"])
        .dropna(subset=["country_iso3","cause_code","cause_name"])
        .reset_index(drop=True)
    )

    # ---- Merge WHO & CT aggregates onto the universe (keep originals as-is)
    df = (universe
          .merge(who, on=["country_iso3","cause_code","cause_name"], how="left")
          .merge(ct_int, on=["country_iso3","cause_code","cause_name","interval_start","interval_end","interval_label"], how="left")
          .copy())

    # IMPORTANT: Do NOT overwrite original columns with zeros.
    # For math, use filled helper columns (NaN -> 0 for totals/shares only).
    df["_trial_sites_filled"]   = df["trial_sites"].fillna(0.0)
    df["_burden_dalys_filled"]  = df["burden_dalys"].fillna(0.0)

    # ---- Totals per interval × cause (using the filled helpers)
    totals = (
        df.groupby(["interval_start","interval_end","interval_label","cause_code"], as_index=False)
          .agg(trial_sites_total=("_trial_sites_filled","sum"),
               burden_total=("_burden_dalys_filled","sum"))
    )

    df = df.merge(totals, on=["interval_start","interval_end","interval_label","cause_code"], how="left")

    # ---- Shares (well-defined even when some countries have NaN originals)
    # If total == 0, share is 0 by convention; otherwise standard fraction.
    # Keep shares as float in [0,1], and allow NaN only when denominator is NaN (shouldn't happen due to fill=0).
    ts_den = df["trial_sites_total"].replace({0: np.nan})
    df["trial_share"] = np.where(ts_den.isna(), 0.0, df["_trial_sites_filled"] / ts_den)

    bd_den = df["burden_total"].replace({0: np.nan})
    df["burden_share"] = np.where(bd_den.isna(), 0.0, df["_burden_dalys_filled"] / bd_den)

    # ---- Equity Ratio: ER = trial_share / burden_share; undefined where burden_share==0
    with np.errstate(divide="ignore", invalid="ignore"):
        df["equity_ratio"] = np.where(df["burden_share"] > 0, df["trial_share"] / df["burden_share"], np.nan)

    # ---- Trial density per 1k DALYs (use original burden; undefined if burden_dalys <= 0 or NaN)
    denom_density = (df["burden_dalys"] / 1_000.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["trial_density_per_k_dalys"] = np.where(denom_density > 0, df["_trial_sites_filled"] / denom_density, np.nan)

    # ---- Inequality metrics per interval×cause
    metrics = []
    for (s, e, lab, cc), sub in df.groupby(["interval_start","interval_end","interval_label","cause_code"], dropna=False):
        p = sub["trial_share"].to_numpy(float)   # already normalized per interval×cause when summed
        b = sub["burden_share"].to_numpy(float)
        er = sub["equity_ratio"].to_numpy(float)

        # If both totals zero, metrics undefined
        ps, bs = np.nansum(p), np.nansum(b)
        if ps <= 0 and bs <= 0:
            g, T = np.nan, np.nan
        else:
            g = weighted_gini(er, b)
            T = theil_T(p, b)

        metrics.append({
            "interval_start": s, "interval_end": e, "interval_label": lab, "cause_code": cc,
            "gini_er_weighted": g, "theil_T": T
        })
    mdf = pd.DataFrame(metrics)
    df  = df.merge(mdf, on=["interval_start","interval_end","interval_label","cause_code"], how="left")

    # ---- Final select (original columns preserved; helpers dropped)
    out_cols = [
        "country_iso3","interval_start","interval_end","interval_label",
        "cause_code","cause_name",
        "trial_sites","burden_dalys",
        "trial_share","burden_share","equity_ratio",
        "trial_density_per_k_dalys",
        "gini_er_weighted","theil_T",
        "metric_kind","metric_label"
    ]
    out = (df[out_cols]
           .sort_values(["interval_start","cause_code","country_iso3"])
           .reset_index(drop=True))

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_GOLD, index=False)
    _info(f"Wrote {OUT_GOLD} (rows={len(out)})")

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(out.head(12).to_string(index=False))
    return out

# ------------------------ Entry ------------------------

if __name__ == "__main__":
    try:
        run_indices()
    except Exception as e:
        print("\n[ERROR] indices.py failed:")
        print(str(e))
        sys.exit(1)
