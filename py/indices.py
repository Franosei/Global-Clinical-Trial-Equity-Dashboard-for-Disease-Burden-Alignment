# py/indices.py
"""
Equity indices:
- Equity Ratio (ER) = trial_share / burden_share
- Trial density per million DALYs
- Weighted Gini and Theil indices on ER (weights = burden_share)
- Optional Dirichlet bootstrap confidence intervals for Gini/Theil

Inputs (from data/clean/):
  - who_daly_selected_harmonised.parquet
  - ctgov_sites_harmonised.parquet
  - country_reference.parquet

Outputs:
  - data/gold/country_year_disease.parquet
  - data/clean/equity_indices.parquet

Run:
  python -m py.indices
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------ Paths & switches ------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
GOLD_DIR = DATA_DIR / "gold"

IN_WHO = CLEAN_DIR / "who_daly_selected_harmonised.parquet"
IN_CT  = CLEAN_DIR / "ctgov_sites_harmonised.parquet"
IN_REF = CLEAN_DIR / "country_reference.parquet"

OUT_COUNTRY = GOLD_DIR / "country_year_disease.parquet"
OUT_INDICES = CLEAN_DIR / "equity_indices.parquet"

# Use only country codes (exclude WHO regions like AFR/AMR/etc)
ONLY_COUNTRIES = True

# Include phase dimension in an auxiliary artifact? (ER uses trials summed across phases)
INCLUDE_PHASE_BREAKOUT = False

# Restrict CT.gov to (year, cause) groups that exist in WHO DALY counts (>0)
RESTRICT_TO_WHO_AVAILABILITY = True

# Bootstrap settings (off by default to stay light)
BOOTSTRAP_DRAWS = 0            # e.g., 500 for CIs
BOOTSTRAP_SEED = 42

# ------------------------ Inequality helpers ------------------------

def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Proper weighted Gini (Lerman–Yitzhaki form):
    Sort by x; let p_i = w_i/sum w, s_i = (w_i x_i)/sum(w x), L_i = cumsum(s_i)
    G = 1 - Σ p_i * (L_{i-1} + L_i)
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v, w = v[m], w[m]
    W = w.sum()
    if v.size == 0 or W <= 0:
        return np.nan

    order = np.argsort(v)
    v, w = v[order], w[order]

    income = w * v
    mu = income.sum()
    if mu <= 0:
        return np.nan

    p = w / W                           # population shares
    s = income / mu                     # income shares
    L = np.cumsum(s)                    # cumulative income shares
    L_prev = np.r_[0.0, L[:-1]]
    G = 1.0 - np.sum(p * (L_prev + L))
    # numerical guard
    return float(np.clip(G, 0.0, 1.0))

def theil_index(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted Theil T:
    T = Σ w_i * (x_i/μ) * ln(x_i/μ) / Σ w_i
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v, w = v[m], w[m]
    W = w.sum()
    if v.size == 0 or W <= 0:
        return np.nan
    mu = np.sum(w * v) / W
    if mu <= 0:
        return np.nan
    r = v / mu
    ok = (r > 0) & (w > 0)
    return float(np.sum(w[ok] * r[ok] * np.log(r[ok])) / W)

# ------------------------ Core computation ------------------------

def _prepare_burden(who: pd.DataFrame) -> pd.DataFrame:
    """
    Keep WHO 'DALYs (count)' only. Aggregate to country-year-cause. Compute shares.
    """
    df = who.copy()
    if "metric" not in df.columns:
        raise ValueError("WHO table is missing 'metric'. Re-run harmonise/global_health_sources.")
    df = df[df["metric"] == "DALYs (count)"].copy()
    if df.empty:
        return df

    grp = (df.groupby(["country_iso3","country_name","year","cause_code","cause_name"], as_index=False)
             .agg(burden=("value","sum")))
    # totals per (year, cause)
    totals = (grp.groupby(["year","cause_code"], as_index=False)["burden"]
                .sum().rename(columns={"burden":"burden_total"}))
    out = grp.merge(totals, on=["year","cause_code"], how="left")
    out["burden_share"] = np.where(out["burden_total"] > 0, out["burden"] / out["burden_total"], np.nan)
    return out

def _prepare_trials(ct: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Sum trial sites across phases (for ER); optionally keep phase breakout.
    Returns (trials_summed, trials_by_phase_or_None)
    """
    base = ["country_iso3","country_name","year","cause_code","cause_name"]
    by_phase = (ct.groupby(base + ["phase"], as_index=False)
                  .agg(trial_sites=("trial_sites","sum")))
    summed = (by_phase.groupby(base, as_index=False)
                      .agg(trial_sites=("trial_sites","sum")))
    # pre-merge totals (used for diagnostics; we will recompute after merge)
    totals = (summed.groupby(["year","cause_code"], as_index=False)["trial_sites"]
                    .sum().rename(columns={"trial_sites":"trial_total"}))
    out = summed.merge(totals, on=["year","cause_code"], how="left")
    out["trial_share"] = np.where(out["trial_total"] > 0, out["trial_sites"] / out["trial_total"], np.nan)
    return out, (by_phase if INCLUDE_PHASE_BREAKOUT else None)

def _merge_and_metrics(burden: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-merge burden and trials; then recompute group totals and shares so
    every row in the same (year, cause) has consistent totals (fixes NaN shares).
    """
    keys = ["country_iso3","country_name","year","cause_code","cause_name"]
    df = pd.merge(burden, trials, on=keys, how="outer", suffixes=("_burden","_trial"))

    # Fill base numerics
    for col in ["burden","trial_sites"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Recompute group totals AFTER the merge
    df["burden_total"] = df.groupby(["year","cause_code"])["burden"].transform("sum")
    df["trial_total"]  = df.groupby(["year","cause_code"])["trial_sites"].transform("sum")

    # Shares (countries with zero trials get 0 share if group has trials)
    df["burden_share"] = np.where(df["burden_total"] > 0, df["burden"] / df["burden_total"], np.nan)
    df["trial_share"]  = np.where(df["trial_total"]  > 0, df["trial_sites"] / df["trial_total"], np.nan)

    # Equity Ratio
    df["equity_ratio"] = np.where(df["burden_share"] > 0, df["trial_share"] / df["burden_share"], np.nan)

    # Trial density per million DALYs
    df["trial_density_per_million_dalys"] = np.where(df["burden"] > 0, df["trial_sites"] / (df["burden"] / 1e6), np.nan)

    return df

def _inequality_table(country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted Gini and Theil on ER per (year, cause_code) with weights = burden_share.
    """
    rows = []
    for (yr, cause), sub in country_df.groupby(["year","cause_code"]):
        x = sub["equity_ratio"].to_numpy()
        w = sub["burden_share"].to_numpy()
        g = weighted_gini(x, w)
        t = theil_index(x, w)
        rows.append({
            "year": yr,
            "cause_code": cause,
            "gini_er": g,
            "theil_er": t,
            "countries_n": int(np.isfinite(x).sum()),
        })

    out = pd.DataFrame(rows)
    # Attach canonical cause names
    cause_name_map = (country_df[["cause_code","cause_name"]]
                      .drop_duplicates()
                      .groupby("cause_code")["cause_name"].first())
    out["cause_name"] = out["cause_code"].map(cause_name_map)
    cols = ["year","cause_code","cause_name","gini_er","theil_er","countries_n"]
    return out[cols].sort_values(["cause_code","year"]).reset_index(drop=True)

def _bootstrap_indices(country_df: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    """
    Dirichlet bootstrap on trial shares per (year, cause_code).
    """
    rng = np.random.default_rng(seed)
    out_rows = []

    for (yr, cause), sub in country_df.groupby(["year","cause_code"]):
        sub = sub[np.isfinite(sub["burden_share"].to_numpy())].copy()
        if sub.empty:
            continue
        alpha = (sub["trial_sites"].fillna(0).to_numpy(dtype=float) + 1.0)
        bshare = sub["burden_share"].to_numpy(dtype=float)
        if not np.isfinite(bshare).all() or bshare.sum() <= 0:
            continue

        g_samples, t_samples = [], []
        for _ in range(draws):
            p_trial = rng.dirichlet(alpha)
            er = np.where(bshare > 0, p_trial / bshare, np.nan)
            g_samples.append(weighted_gini(er, bshare))
            t_samples.append(theil_index(er, bshare))

        if g_samples:
            qg = np.nanpercentile(g_samples, [2.5, 50, 97.5]).tolist()
            qt = np.nanpercentile(t_samples, [2.5, 50, 97.5]).tolist()
            out_rows.append({
                "year": yr, "cause_code": cause,
                "gini_p2_5": qg[0], "gini_p50": qg[1], "gini_p97_5": qg[2],
                "theil_p2_5": qt[0], "theil_p50": qt[1], "theil_p97_5": qt[2],
                "draws": draws,
            })

    return pd.DataFrame(out_rows)

# ------------------------ Main ------------------------

def run() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    if not IN_WHO.exists() or not IN_CT.exists():
        missing = [p.name for p in [IN_WHO, IN_CT] if not p.exists()]
        raise FileNotFoundError(f"Missing inputs: {', '.join(missing)}. Run harmonise first.")

    who = pd.read_parquet(IN_WHO)
    ct  = pd.read_parquet(IN_CT)

    # Optionally restrict to actual countries
    if ONLY_COUNTRIES:
        if "is_country_code" in who.columns:
            who = who[who["is_country_code"].fillna(False)]
        if "is_country_code" in ct.columns:
            ct = ct[ct["is_country_code"].fillna(False)]

    # Prep datasets
    burden = _prepare_burden(who)
    trials, by_phase = _prepare_trials(ct)

    # Optionally restrict CT.gov to (year, cause) groups that exist in WHO with positive burden
    if RESTRICT_TO_WHO_AVAILABILITY and not burden.empty and not trials.empty:
        who_groups = (burden.groupby(["year","cause_code"], as_index=False)
                             .agg(burden_total_sum=("burden","sum")))
        who_groups = who_groups[who_groups["burden_total_sum"] > 0][["year","cause_code"]]
        before = len(trials)
        trials = trials.merge(who_groups, on=["year","cause_code"], how="inner")
        after = len(trials)
        if before != after:
            print(f"[INFO] Dropped {before - after} CT.gov rows for (year,cause) with no WHO DALY counts.")

    # Merge + metrics (recompute group totals here)
    country = _merge_and_metrics(burden, trials)

    # Optional phase breakout artifact
    if INCLUDE_PHASE_BREAKOUT and by_phase is not None:
        by_phase.to_parquet(CLEAN_DIR / "ctgov_sites_by_phase.parquet", index=False)

    # Save detailed country-year-disease table
    cols = ["country_iso3","country_name","year","cause_code","cause_name",
            "trial_sites","trial_total","trial_share",
            "burden","burden_total","burden_share",
            "equity_ratio","trial_density_per_million_dalys"]
    country_out = country[cols].sort_values(["cause_code","year","country_iso3"]).reset_index(drop=True)
    country_out.to_parquet(OUT_COUNTRY, index=False)
    print(f"[OK] wrote {OUT_COUNTRY} (rows={len(country_out)})")

    # Inequality indices
    indices = _inequality_table(country_out)

    # Optional bootstrap CIs
    if BOOTSTRAP_DRAWS and BOOTSTRAP_DRAWS > 0:
        boot = _bootstrap_indices(country_out, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        indices = indices.merge(boot, on=["year","cause_code"], how="left")

    indices.to_parquet(OUT_INDICES, index=False)
    print(f"[OK] wrote {OUT_INDICES} (rows={len(indices)})")

    # Quick previews
    with pd.option_context("display.max_columns", None, "display.width", 1200):
        print("\n[Country-level preview]")
        print(country_out.head(10).to_string(index=False))
        print("\n[Inequality preview]")
        print(indices.head(10).to_string(index=False))

if __name__ == "__main__":
    run()
