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
    """Assumption sensitivity from placement and burden variation.

    Burden is perturbed with a conservative 10% log-normal coefficient of
    variation. New sites are sampled from the selected strategy's allocation
    probabilities. The resulting quantiles are sensitivity ranges, not
    confidence intervals, prediction intervals or a causal prediction.
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
    """Governed forecast selected against a last-value benchmark.

    The candidate log-linear trend is used only when rolling-origin backtests
    beat a last-observation forecast. Prediction intervals use out-of-sample
    log errors for the selected model when at least five are available. This is
    deliberately conservative: a trend that cannot beat persistence is not
    presented as evidence of growth or decline.
    """
    observed = (
        annual_data.loc[annual_data["cause_code"] == cause_code]
        .groupby("year", as_index=False)["trial_sites"]
        .sum()
        .sort_values("year")
    )
    current_year = datetime.now(timezone.utc).year
    observed = observed[observed["year"] < current_year]
    if observed.empty:
        return pd.DataFrame(columns=["year", "estimate", "lower", "upper", "kind"])
    # Missing calendar years mean zero global activity, not an omitted time
    # point. Keeping a regular index also makes rolling validation honest.
    full_years = np.arange(int(observed["year"].min()), int(observed["year"].max()) + 1)
    series = observed.set_index("year")["trial_sites"].reindex(full_years, fill_value=0).astype(float)
    history = series.tail(max(int(lookback), 4))
    if len(history) < 4:
        return pd.DataFrame(columns=["year", "estimate", "lower", "upper", "kind"])

    x = history.index.to_numpy(float)
    y = np.log1p(history.to_numpy(float))
    slope, intercept = np.polyfit(x - x.min(), y, 1)
    fitted = intercept + slope * (x - x.min())
    residuals = y - fitted
    future_years = np.arange(int(x.max()) + 1, int(x.max()) + horizon + 1)
    loglinear_point = np.maximum(np.expm1(intercept + slope * (future_years - x.min())), 0)
    naive_point = np.repeat(float(history.iloc[-1]), len(future_years))

    # Rolling-origin evaluation over the complete available series. Each
    # origin uses only information that precedes its target.
    records = []
    min_train = 4
    years = series.index.to_numpy(int)
    values = series.to_numpy(float)
    for origin_pos in range(min_train - 1, len(series) - 1):
        train_start = max(0, origin_pos - max(int(lookback), 4) + 1)
        train_years = years[train_start:origin_pos + 1].astype(float)
        train_values = values[train_start:origin_pos + 1]
        train_log = np.log1p(train_values)
        bt_slope, bt_intercept = np.polyfit(train_years - train_years.min(), train_log, 1)
        for h in range(1, max(int(horizon), 1) + 1):
            target_pos = origin_pos + h
            if target_pos >= len(series):
                break
            actual = values[target_pos]
            trend = max(float(np.expm1(
                bt_intercept + bt_slope * (years[target_pos] - train_years.min())
            )), 0.0)
            naive = float(train_values[-1])
            records.append((h, actual, trend, naive))
    backtest = pd.DataFrame(records, columns=["horizon", "actual", "loglinear", "naive"])

    def wape(column: str) -> float:
        if backtest.empty:
            return float("inf")
        denominator = float(backtest["actual"].sum())
        return float((backtest[column] - backtest["actual"]).abs().sum() / max(denominator, 1.0))

    loglinear_wape, naive_wape = wape("loglinear"), wape("naive")
    selected_model = "log-linear trend" if loglinear_wape + 1e-12 < naive_wape else "last-value benchmark"
    point = loglinear_point if selected_model == "log-linear trend" else naive_point

    rng = np.random.default_rng(seed)
    lower, upper, interval_methods = [], [], []
    for index, h in enumerate(range(1, len(future_years) + 1)):
        errors = pd.Series(dtype=float)
        if not backtest.empty:
            scoped = backtest[backtest["horizon"] == h]
            prediction = scoped["loglinear"] if selected_model == "log-linear trend" else scoped["naive"]
            errors = np.log1p(scoped["actual"]) - np.log1p(prediction.clip(lower=0))
        if len(errors) >= 5:
            q_low, q_high = np.quantile(errors.to_numpy(float), [0.1, 0.9])
            interval_methods.append("rolling-origin empirical errors")
        else:
            if selected_model == "log-linear trend":
                sampled_errors = rng.choice(residuals, max(int(draws), 100), replace=True)
            else:
                differences = np.diff(np.log1p(history.to_numpy(float)))
                sampled_errors = rng.choice(differences if len(differences) else [0.0],
                                            max(int(draws), 100), replace=True) * np.sqrt(h)
            q_low, q_high = np.quantile(sampled_errors, [0.1, 0.9])
            interval_methods.append("in-sample residual fallback")
        centre = np.log1p(point[index])
        lo, hi = max(float(np.expm1(centre + q_low)), 0.0), max(float(np.expm1(centre + q_high)), 0.0)
        lower.append(min(lo, float(point[index])))
        upper.append(max(hi, float(point[index])))

    forecast = pd.DataFrame({
        "year": future_years,
        "estimate": point,
        "lower": lower,
        "upper": upper,
        "kind": "Forecast",
    })
    actual = history.rename("estimate").rename_axis("year").reset_index().assign(
        lower=lambda d: d["estimate"], upper=lambda d: d["estimate"], kind="Actual")
    result = pd.concat([actual, forecast], ignore_index=True)
    # Fitted model, kept for the methods explanation and for auditing.
    result.attrs["model"] = {
        "first_year": int(x.min()), "last_year": int(x.max()), "n_years": int(len(x)),
        "intercept": float(intercept), "slope": float(slope),
        "residual_sd": float(np.std(residuals, ddof=2)) if len(x) > 2 else float("nan"),
        "r_squared": float(1 - residuals.var() / y.var()) if y.var() > 0 else float("nan"),
        "selected_model": selected_model,
        "loglinear_backtest_wape": loglinear_wape,
        "naive_backtest_wape": naive_wape,
        "backtest_predictions": int(len(backtest)),
        "interval_method": ", ".join(sorted(set(interval_methods))),
        "draws": int(draws), "seed": int(seed),
    }
    return result
