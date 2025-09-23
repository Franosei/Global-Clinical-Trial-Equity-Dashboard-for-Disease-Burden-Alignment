"""
Indices: Equity Ratio, Burden-Adjusted Trial Density, Gini & Theil

Inputs (clean):
  data/clean/who_daly_selected.parquet       # columns: country_iso3, year, cause_code, cause_name, value, value_low, value_high, metric
  data/clean/ctgov_sites.parquet             # columns: country_iso3, year, cause_code, cause_name, phase, trial_sites

Output (gold, single file for dashboard):
  data/gold/country_year_disease.parquet

Columns:
  country_iso3, year, cause_code, cause_name,
  trial_sites, burden_dalys,
  trial_share, burden_share, equity_ratio,
  trial_density_per_k_dalys,
  gini_er_weighted, theil_T

Run:
  python -m py.indices
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ------------------------ Paths ------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
GOLD_DIR = DATA_DIR / "gold"

IN_WHO = CLEAN_DIR / "who_daly_selected.parquet"
IN_CT  = CLEAN_DIR / "ctgov_sites.parquet"

OUT_GOLD = GOLD_DIR / "country_year_disease.parquet"

# ------------------------ Helpers ------------------------

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def _info(msg: str) -> None:
    print(f"[INFO] {msg}")

def _validate_group_shares(df: pd.DataFrame) -> None:
    """
    Soft validation: within each (year, cause), sums of trial_share and burden_share should be ~1.
    Prints warnings when off by >1e-6.
    """
    if df.empty:
        return
    grouped = df.groupby(["year", "cause_code"], dropna=False)
    for (yr, cc), sub in grouped:
        ts = sub["trial_share"].dropna().sum()
        bs = sub["burden_share"].dropna().sum()
        if not np.isclose(ts, 1.0, atol=1e-6):
            _warn(f"trial_share does not sum to 1.0 → {ts:.6f} for cause={cc}, year={yr}")
        if not np.isclose(bs, 1.0, atol=1e-6):
            _warn(f"burden_share does not sum to 1.0 → {bs:.6f} for cause={cc}, year={yr}")

def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted Gini coefficient.
    Here it is applied to ER (values) with burden_share (weights).
    Returns NaN if inputs invalid or mean <= 0.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v, w = v[mask], w[mask]
    if v.size == 0 or w.sum() <= 0:
        return np.nan
    # Weighted mean
    mu = np.average(v, weights=w)
    if not np.isfinite(mu) or mu <= 0:
        return np.nan
    # Sort by value
    order = np.argsort(v)
    v, w = v[order], w[order]
    # Cumulative weights & values
    cw = np.cumsum(w)
    # Gini via pairwise formulation expressed with cumulative sums:
    # see e.g. Ogwang (2000) / common weighted implementation
    # G = 1 - (1/mu) * sum_i w_i * (S_{i-1} + S_i) / (W)  where S_i is cumweighted value
    vw = v * w
    c_vw = np.cumsum(vw)
    W = cw[-1]
    # sum over i of w_i * (S_{i-1} + S_i)
    s = (w * (np.concatenate(([0.0], c_vw[:-1])) + c_vw)).sum()
    G = 1.0 - s / (mu * W * W)
    return float(G)

def theil_T(trial_share: np.ndarray, burden_share: np.ndarray) -> float:
    """
    Theil T index for divergence of trial allocation from burden.
    T = sum_i p_i * ln(p_i / b_i), where p_i = trial_share, b_i = burden_share.
    Returns NaN if inputs invalid; terms with p_i == 0 contribute 0.
    """
    p = np.asarray(trial_share, dtype=float)
    b = np.asarray(burden_share, dtype=float)
    mask = np.isfinite(p) & np.isfinite(b) & (p >= 0) & (b > 0)  # require b > 0
    p, b = p[mask], b[mask]
    if p.size == 0:
        return np.nan
    # Normalise defensively to sum to 1 (numeric stability)
    p_sum = p.sum()
    b_sum = b.sum()
    if p_sum <= 0 or b_sum <= 0:
        return np.nan
    p = p / p_sum
    b = b / b_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p / b
        term = np.where(p > 0, p * np.log(ratio), 0.0)
    return float(np.nansum(term))

# ------------------------ Pipeline ------------------------

def run_indices() -> pd.DataFrame:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load inputs
    if not IN_WHO.exists():
        raise FileNotFoundError(f"Missing WHO input: {IN_WHO}")
    if not IN_CT.exists():
        raise FileNotFoundError(f"Missing CT.gov input: {IN_CT}")

    who = pd.read_parquet(IN_WHO)
    ct  = pd.read_parquet(IN_CT)

    # ---- Keep DALYs (count) only
    who = who[(who["metric"] == "DALYs (count)")].copy()

    # ---- Ensure types
    for col in ["country_iso3", "cause_code", "cause_name"]:
        who[col] = who[col].astype(str)
        ct[col]  = ct[col].astype(str)
    who["year"] = who["year"].astype(int)
    ct["year"]  = ct["year"].astype(int)
    who["value"] = pd.to_numeric(who["value"], errors="coerce")
    ct["trial_sites"] = pd.to_numeric(ct["trial_sites"], errors="coerce").fillna(0).astype(int)

    # ---- Restrict to overlapping (year, cause) sets to avoid zero denominators
    who_keys = who[["year","cause_code"]].drop_duplicates()
    ct_keys  = ct[["year","cause_code"]].drop_duplicates()
    overlap  = who_keys.merge(ct_keys, on=["year","cause_code"], how="inner")
    if overlap.empty:
        _warn("No overlapping (year, cause) between WHO and CT.gov. Output will be empty.")
        out_empty = pd.DataFrame(columns=[
            "country_iso3","year","cause_code","cause_name","trial_sites","burden_dalys",
            "trial_share","burden_share","equity_ratio","trial_density_per_k_dalys",
            "gini_er_weighted","theil_T"
        ])
        out_empty.to_parquet(OUT_GOLD, index=False)
        _info(f"Wrote {OUT_GOLD} (rows=0)")
        return out_empty

    who = who.merge(overlap, on=["year","cause_code"], how="inner")
    ct  = ct.merge(overlap,  on=["year","cause_code"], how="inner")

    # ---- Aggregate CT across phases for ER (phase slices can be derived later if needed)
    ct_tot = (ct.groupby(["country_iso3","year","cause_code","cause_name"], as_index=False)
                .agg(trial_sites=("trial_sites","sum")))

    # ---- Aggregate WHO DALYs counts per country/year/cause
    who_tot = (who.groupby(["country_iso3","year","cause_code","cause_name"], as_index=False)
                 .agg(burden_dalys=("value","sum")))

    # ---- Inner-merge on the shared keys
    df = ct_tot.merge(who_tot, on=["country_iso3","year","cause_code","cause_name"], how="inner")

    if df.empty:
        _warn("Merge produced no rows after aligning on (country, year, cause).")
        out_empty = pd.DataFrame(columns=[
            "country_iso3","year","cause_code","cause_name","trial_sites","burden_dalys",
            "trial_share","burden_share","equity_ratio","trial_density_per_k_dalys",
            "gini_er_weighted","theil_T"
        ])
        out_empty.to_parquet(OUT_GOLD, index=False)
        _info(f"Wrote {OUT_GOLD} (rows=0)")
        return out_empty

    # ---- Compute group totals per (year, cause)
    grp = df.groupby(["year","cause_code"], as_index=False)
    totals = grp.agg(trial_sites_total=("trial_sites","sum"),
                     burden_total=("burden_dalys","sum"))

    df = df.merge(totals, on=["year","cause_code"], how="left")

    # ---- Shares & ER
    df["trial_share"]  = df["trial_sites"] / df["trial_sites_total"].replace({0: np.nan})
    df["burden_share"] = df["burden_dalys"] / df["burden_total"].replace({0: np.nan})
    df["equity_ratio"] = df["trial_share"] / df["burden_share"]

    # ---- Burden-adjusted trial density (sites per 1k DALYs)
    df["trial_density_per_k_dalys"] = df["trial_sites"] / (df["burden_dalys"] / 1_000.0).replace({0: np.nan})

    # ---- Inequality metrics per (year, cause), then broadcast to rows
    metrics = []
    for (yr, cc), sub in df.groupby(["year","cause_code"]):
        # Filter out NaNs and zeros safely
        p = sub["trial_share"].to_numpy(dtype=float)
        b = sub["burden_share"].to_numpy(dtype=float)
        er = sub["equity_ratio"].to_numpy(dtype=float)

        # Weighted Gini of ER using burden weights (b)
        gini = weighted_gini(er, b)

        # Theil T divergence between trial and burden shares
        T = theil_T(p, b)

        metrics.append({"year": yr, "cause_code": cc,
                        "gini_er_weighted": gini, "theil_T": T})

    mdf = pd.DataFrame(metrics)
    df = df.merge(mdf, on=["year","cause_code"], how="left")

    # ---- Validation: shares should sum ~1 within each (year, cause)
    _validate_group_shares(df)

    # ---- Select & sort columns for gold
    out_cols = [
        "country_iso3","year","cause_code","cause_name",
        "trial_sites","burden_dalys",
        "trial_share","burden_share","equity_ratio",
        "trial_density_per_k_dalys",
        "gini_er_weighted","theil_T",
    ]
    out = (df[out_cols]
            .sort_values(["year","cause_code","country_iso3"])
            .reset_index(drop=True))

    # ---- Write single gold output
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_GOLD, index=False)
    _info(f"Wrote {OUT_GOLD} (rows={len(out)})")

    # ---- Preview
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
