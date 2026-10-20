"""Decision analytics for forecasts and trial-site allocation scenarios.

The functions in this module are framework-independent so they can be tested,
audited and reused outside the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd


Strategy = Literal["Equity gap", "Disease burden", "Hybrid"]


def theil_index(trial_share: np.ndarray, burden_share: np.ndarray) -> float:
    p = np.asarray(trial_share, dtype=float)
    b = np.asarray(burden_share, dtype=float)
    valid = np.isfinite(p) & np.isfinite(b) & (p >= 0) & (b > 0)
    p, b = p[valid], b[valid]
    if p.size == 0 or p.sum() <= 0 or b.sum() <= 0:
        return np.nan
    p, b = p / p.sum(), b / b.sum()
    positive = p > 0
    return float(np.sum(p[positive] * np.log(p[positive] / b[positive])))


def alignment_score(trial_share: np.ndarray, burden_share: np.ndarray) -> float:
    """Return 0-100 alignment based on one minus total-variation distance."""
    p = np.nan_to_num(np.asarray(trial_share, dtype=float), nan=0.0)
    b = np.nan_to_num(np.asarray(burden_share, dtype=float), nan=0.0)
    if p.sum() <= 0 or b.sum() <= 0:
        return np.nan
    p, b = p / p.sum(), b / b.sum()
    return float(100 * (1 - 0.5 * np.abs(p - b).sum()))


def sites_to_parity(sites: np.ndarray, burden_share: np.ndarray) -> np.ndarray:
    """Exact additions needed for each country to reach ER=1 on its own.

    Solves ``(sites_i + x) / (total_sites + x) = burden_share_i``.
    """
    s = np.nan_to_num(np.asarray(sites, dtype=float), nan=0.0)
    b = np.asarray(burden_share, dtype=float)
    total = s.sum()
    denominator = 1 - b
    with np.errstate(divide="ignore", invalid="ignore"):
        needed = np.where(
            (b > 0) & (b < 1), np.maximum((b * total - s) / denominator, 0), 0
        )
    return needed


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    if total <= 0:
        return np.zeros(len(weights), dtype=int)
    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0)
    weights = np.maximum(weights, 0)
    if weights.sum() <= 0:
        weights = np.ones(len(weights), dtype=float)
    raw = total * weights / weights.sum()
    result = np.floor(raw).astype(int)
    remainder = total - int(result.sum())
    if remainder:
        order = np.argsort(-(raw - result), kind="stable")
        result[order[:remainder]] += 1
    return result


def allocate_sites(data: pd.DataFrame, extra_sites: int, strategy: Strategy) -> pd.DataFrame:
    """Allocate an integer portfolio and recompute all post-scenario shares."""
    out = data.copy()
    sites = np.nan_to_num(out["trial_sites"].to_numpy(float), nan=0.0)
    burden = np.nan_to_num(out["burden_share"].to_numpy(float), nan=0.0)
    burden = burden / burden.sum() if burden.sum() > 0 else burden
    total_after = sites.sum() + max(int(extra_sites), 0)
    target_gap = np.maximum(burden * total_after - sites, 0)

    if strategy == "Disease burden":
        weights = burden
    elif strategy == "Hybrid":
        gap_w = target_gap / target_gap.sum() if target_gap.sum() else target_gap
        weights = 0.7 * gap_w + 0.3 * burden
    else:
        weights = target_gap

    allocated = _largest_remainder(weights, max(int(extra_sites), 0))
    after = sites + allocated
    after_share = after / after.sum() if after.sum() > 0 else np.zeros(len(after))
    with np.errstate(divide="ignore", invalid="ignore"):
        after_er = np.where(burden > 0, after_share / burden, np.nan)

    out["allocated_sites"] = allocated
    out["trial_sites_after"] = after
    out["trial_share_after"] = after_share
    out["equity_ratio_after"] = after_er
    out["sites_to_parity"] = sites_to_parity(sites, burden)
    return out


@dataclass(frozen=True)
class SimulationSummary:
    alignment_p10: float
    alignment_median: float
    alignment_p90: float
    theil_p10: float
    theil_median: float
    theil_p90: float


def simulate_allocation(
    data: pd.DataFrame,
    extra_sites: int,
    strategy: Strategy,
    draws: int = 500,
    seed: int = 42,
) -> SimulationSummary:
    """Monte Carlo scenario uncertainty from placement and burden variation.

    Burden is perturbed with a conservative 10% log-normal coefficient of
    variation. New sites are sampled from the selected strategy's allocation
    probabilities. This is scenario uncertainty, not a causal prediction.
    """
    deterministic = allocate_sites(data, extra_sites, strategy)
    current = np.nan_to_num(data["trial_sites"].to_numpy(float), nan=0.0)
    burden = np.nan_to_num(data["burden_share"].to_numpy(float), nan=0.0)
    burden = burden / burden.sum()
    alloc = deterministic["allocated_sites"].to_numpy(float)
    probabilities = alloc / alloc.sum() if alloc.sum() else burden
    rng = np.random.default_rng(seed)
    alignments, theils = [], []
    sigma = np.sqrt(np.log(1.0 + 0.10**2))

    for _ in range(max(int(draws), 1)):
        simulated_burden = burden * rng.lognormal(-0.5 * sigma**2, sigma, len(burden))
        simulated_burden /= simulated_burden.sum()
        additions = rng.multinomial(max(int(extra_sites), 0), probabilities)
        simulated_sites = current + additions
        site_share = simulated_sites / simulated_sites.sum()
        alignments.append(alignment_score(site_share, simulated_burden))
        theils.append(theil_index(site_share, simulated_burden))

    aq = np.nanpercentile(alignments, [10, 50, 90])
    tq = np.nanpercentile(theils, [10, 50, 90])
    return SimulationSummary(*map(float, [aq[0], aq[1], aq[2], tq[0], tq[1], tq[2]]))


def forecast_trial_sites(
    annual_data: pd.DataFrame,
    cause_code: str,
    horizon: int = 3,
    lookback: int = 8,
    draws: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Log-linear global site forecast with residual-bootstrap intervals."""
    history = (
        annual_data.loc[annual_data["cause_code"] == cause_code]
        .groupby("year", as_index=False)["trial_sites"]
        .sum()
        .sort_values("year")
    )
    current_year = datetime.now(timezone.utc).year
    history = history[history["year"] < current_year].tail(max(int(lookback), 4))
    if len(history) < 4:
        return pd.DataFrame(columns=["year", "estimate", "lower", "upper", "kind"])

    x = history["year"].to_numpy(float)
    y = np.log1p(history["trial_sites"].to_numpy(float))
    slope, intercept = np.polyfit(x - x.min(), y, 1)
    fitted = intercept + slope * (x - x.min())
    residuals = y - fitted
    future_years = np.arange(int(x.max()) + 1, int(x.max()) + horizon + 1)
    future_base = intercept + slope * (future_years - x.min())
    rng = np.random.default_rng(seed)
    sampled = future_base[:, None] + rng.choice(residuals, (len(future_years), draws), replace=True)
    sampled = np.maximum(np.expm1(sampled), 0)
    forecast = pd.DataFrame({
        "year": future_years,
        "estimate": np.median(sampled, axis=1),
        "lower": np.percentile(sampled, 10, axis=1),
        "upper": np.percentile(sampled, 90, axis=1),
        "kind": "Forecast",
    })
    actual = history.rename(columns={"trial_sites": "estimate"}).assign(
        lower=lambda d: d["estimate"], upper=lambda d: d["estimate"], kind="Actual"
    )
    result = pd.concat([actual, forecast], ignore_index=True)
    # Fitted model, kept for the methods explanation and for auditing.
    result.attrs["model"] = {
        "first_year": int(x.min()), "last_year": int(x.max()), "n_years": int(len(x)),
        "intercept": float(intercept), "slope": float(slope),
        "residual_sd": float(np.std(residuals, ddof=2)) if len(x) > 2 else float("nan"),
        "r_squared": float(1 - residuals.var() / y.var()) if y.var() > 0 else float("nan"),
        "draws": int(draws), "seed": int(seed),
    }
    return result
