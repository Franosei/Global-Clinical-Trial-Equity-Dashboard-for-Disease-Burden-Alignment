# py/indices.py
"""
Interval indices (2000–2005, 2006–2010, 2011–2015, 2016–2020, 2021–2025):
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
        2000–2005, 2006–2010, 2011–2015, 2016–2020, 2021–2025
  • For each (interval, cause): trial_share varies with CT allocation; burden_share derives from WHO.
    Equity Ratio (ER) = trial_share / burden_share per country.
  • Trial density per 1k DALYs uses WHO burden as denominator.
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

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
    m = np.isfinite(p) & np.isfinite(b) & (p >= 0) & (b > 0)
    p, b = p[m], b[m]
    if p.size == 0:
        return np.nan
    ps, bs = p.sum(), b.sum()
    if ps <= 0 or bs <= 0:
        return np.nan
    p, b = p / ps, b / bs
    with np.errstate(divide="ignore", invalid="ignore"):
        r = p / b
        term = np.where(p > 0, p * np.log(r), 0.0)
    return float(np.nansum(term))

# Fixed interval mapping: 2000–2005, 2006–2010, 2011–2015, 2016–2020, 2021–2025
_INTERVALS = [
    (2000, 2005),
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2025),
]

def assign_interval(year: int) -> tuple[int, int, str] | None:
    """Return (start, end, label) for year in fixed intervals; None if outside all."""
    for s, e in _INTERVALS:
        if s <= year <= e:
            return s, e, f"{s}–{e}"
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

    # ---- WHO: country×cause only (no year). Use as stable severity weight.
    for c in ["country_iso3", "cause_code", "cause_name"]:
        who[c] = who[c].astype(str)
    who["burden_dalys"] = pd.to_numeric(who["value"], errors="coerce")
    # Metric kind/label (from WHO pipeline this is per-100k, summed across years)
    who["metric_kind"]  = "rate_per_100k"
    who["metric_label"] = who.get("metric", "DALYs per 100k population")

    who = who[["country_iso3","cause_code","cause_name","burden_dalys","metric_kind","metric_label"]]

    # ---- CT.gov: aggregate into fixed intervals; sum across phases
    for c in ["country_iso3", "cause_code", "cause_name", "phase"]:
        ct[c] = ct[c].astype(str)
    ct["year"]        = pd.to_numeric(ct["year"], errors="coerce").astype("Int64")
    ct["trial_sites"] = pd.to_numeric(ct["trial_sites"], errors="coerce").fillna(0).astype(int)
    ct = ct.dropna(subset=["year"]).copy()
    ct["year"] = ct["year"].astype(int)

    # Map to intervals and drop years outside defined bins
    ints = ct["year"].apply(assign_interval)
    ct["interval_tuple"] = ints
    ct = ct[ct["interval_tuple"].notna()].copy()
    if ct.empty:
        _warn("All CT.gov rows fall outside the defined intervals.")
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

    ct["interval_start"] = ct["interval_tuple"].apply(lambda x: x[0])
    ct["interval_end"]   = ct["interval_tuple"].apply(lambda x: x[1])
    ct["interval_label"] = ct["interval_tuple"].apply(lambda x: x[2])

    # Total trial sites per country×interval×cause (phases aggregated)
    ct_int = (ct.groupby(
                ["country_iso3", "interval_start", "interval_end", "interval_label",
                 "cause_code", "cause_name"], as_index=False)
                .agg(trial_sites=("trial_sites", "sum")))

    if ct_int.empty:
        _warn("CT.gov interval aggregation produced no rows.")
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

    # ---- Merge WHO burden onto CT intervals (no year key needed)
    df = ct_int.merge(who, on=["country_iso3","cause_code","cause_name"], how="left")

    # If burden is missing for some country×cause, drop (ER cannot be computed)
    df = df.dropna(subset=["burden_dalys"]).copy()

    # ---- Compute shares & ER per (interval, cause)
    totals = (df.groupby(["interval_start","interval_end","interval_label","cause_code"], as_index=False)
                .agg(trial_sites_total=("trial_sites","sum"),
                     burden_total=("burden_dalys","sum")))
    df = df.merge(totals, on=["interval_start","interval_end","interval_label","cause_code"], how="left")

    df["trial_share"]  = df["trial_sites"]  / df["trial_sites_total"].replace({0: np.nan})
    df["burden_share"] = df["burden_dalys"] / df["burden_total"].replace({0: np.nan})
    df["equity_ratio"] = df["trial_share"] / df["burden_share"]

    # ---- Trial density per 1k DALYs (burden_dalys is per-100k; density is sites per 1k DALYs)
    df["trial_density_per_k_dalys"] = df["trial_sites"] / (df["burden_dalys"] / 1_000.0).replace({0: np.nan})

    # ---- Inequality metrics (weighted by burden_share) per interval×cause
    metrics = []
    for (s, e, lab, cc), sub in df.groupby(["interval_start","interval_end","interval_label","cause_code"], dropna=False):
        p  = sub["trial_share"].to_numpy(float)
        b  = sub["burden_share"].to_numpy(float)
        er = sub["equity_ratio"].to_numpy(float)
        g  = weighted_gini(er, b)
        T  = theil_T(p, b)
        metrics.append({
            "interval_start": s, "interval_end": e, "interval_label": lab, "cause_code": cc,
            "gini_er_weighted": g, "theil_T": T
        })
    mdf = pd.DataFrame(metrics)
    df  = df.merge(mdf, on=["interval_start","interval_end","interval_label","cause_code"], how="left")

    # ---- Final select & write
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
