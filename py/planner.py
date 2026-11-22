"""Resource-aware trial-site planning.

The equity view says how many additional study placements a country needs. A
placement does not have to mean a new site: a facility that has already run
studies of the disease can host more, and a newly established site can host
several studies once it exists. This module turns a budget into a placement
plan that uses a historical registry-activity proxy first and opens new sites
only where that is the better use of money.

Measured inputs (no imputation):
  • Facility activity proxy from ClinicalTrials.gov: the most studies of the disease a
    named facility has started in any window as long as the selected year
    range, up to its last year. Headroom is that historical peak minus the
    studies it already has in the selected range. Anonymised sponsor listings
    are excluded because they cannot be identified as distinct facilities.
    Registry history does not establish activation, enrolment or spare capacity.
  • New-site throughput assumption: a percentile of historical activity at named
    facilities in the same country, else the same World Bank income group,
    else all countries. The percentile is an observed value, not interpolated.
  • Country cost scaling: ICP 2021 price level index for health
    (World = 100) relative to the reference country, falling back to the ICP
    GDP price level index (always compared on the same basis). Countries with
    neither are excluded from allocation and reported.

Planner inputs (entered by the user): study start-up cost at a site, one-time
cost of establishing a new site, and the budget, all at reference prices.

Optimisation: greedy maximisation of the alignment score per unit of cost.
At each step the planner compares, for every eligible country, adding one study
against a site's historical activity headroom versus establishing a new site
and filling it to an assumed throughput or the country's parity need. The baseline "one new
site per placement" plan uses the same rule with no capacity reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analytics import _largest_remainder, alignment_score, sites_to_parity

CAPACITY_PERCENTILES = {"Median": 0.5, "75th percentile": 0.75, "90th percentile": 0.9}
_TOLERANCE = 1e-12


# ------------------------------------------------------------------ capacity

def facility_capacity(
    facilities: pd.DataFrame,
    cause_code: str,
    start: int,
    end: int,
    recency_windows: int = 1,
) -> pd.DataFrame:
    """Historical registry-activity ceiling and headroom proxy per facility.

    The selected range [start, end] is L years long. A facility's capacity is
    the most studies of the disease it started in any L-year window ending on
    or before ``end``; ``used`` is its studies in the selected range itself.
    Only years up to ``end`` are read, so the proxy never relies on activity
    that had not happened yet. A facility counts as active if it started a
    study in the selected range or the ``recency_windows`` L-year windows
    before it.
    """
    columns = ["country_iso3", "facility_key", "facility", "city", "lat", "lon",
               "capacity", "used", "active", "headroom"]
    start, end = int(start), int(end)
    length = end - start + 1
    scoped = facilities[(facilities["cause_code"] == cause_code) & ~facilities["anonymised"]
                        & (facilities["year"] <= end)]
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    counts = scoped.groupby(["facility_key", "year"])["nct_id"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=range(int(counts.columns.min()), end + 1), fill_value=0)
    cumulative = np.concatenate([np.zeros((len(counts), 1)), counts.to_numpy().cumsum(axis=1)], axis=1)
    window_end = np.arange(1, cumulative.shape[1])
    window_sums = cumulative[:, window_end] - cumulative[:, np.maximum(window_end - length, 0)]
    first_year = int(counts.columns.min())
    recent_from = max(start - length * int(recency_windows), first_year)

    out = (scoped.sort_values("year")
           .groupby("facility_key")
           .agg(country_iso3=("country_iso3", "first"), facility=("facility", "last"),
                city=("city", "last"), lat=("lat", "median"), lon=("lon", "median"))
           .reindex(counts.index))
    out["capacity"] = window_sums.max(axis=1).astype(int)
    out["used"] = counts.loc[:, max(start, first_year):end].sum(axis=1).astype(int)
    out["active"] = counts.loc[:, recent_from:end].sum(axis=1) > 0
    out["headroom"] = np.where(out["active"], np.maximum(out["capacity"] - out["used"], 0), 0).astype(int)
    return out.reset_index()[columns]


def new_site_capacity(
    capacity: pd.DataFrame,
    resources: pd.DataFrame,
    countries: pd.Series,
    percentile: float,
    min_facilities: int = 5,
) -> pd.DataFrame:
    """Assumed new-site throughput derived from historical registry activity.

    Uses facilities in the country when there are enough of them, otherwise the
    country's income group, otherwise all countries. Returns the observed value
    at the requested percentile and the basis used.
    """
    groups = resources.set_index("country_iso3")["income_group"]
    named = capacity[["country_iso3", "capacity"]].copy()
    named["income_group"] = named["country_iso3"].map(groups)

    def observed(values: pd.Series) -> float:
        return float(np.quantile(values.to_numpy(float), percentile, method="inverted_cdf"))

    global_value = observed(named["capacity"]) if len(named) else 1.0
    by_country = named.groupby("country_iso3")["capacity"]
    country_n = by_country.size()
    country_value = by_country.apply(observed)
    by_group = named.dropna(subset=["income_group"]).groupby("income_group")["capacity"]
    group_n = by_group.size()
    group_value = by_group.apply(observed)

    rows = []
    for iso3 in countries:
        group = groups.get(iso3)
        if country_n.get(iso3, 0) >= min_facilities:
            value, basis = country_value[iso3], "Country"
        elif isinstance(group, str) and group_n.get(group, 0) >= min_facilities:
            value, basis = group_value[group], f"Income group ({group})"
        else:
            value, basis = global_value, "All countries"
        rows.append({"country_iso3": iso3, "new_site_capacity": max(int(value), 1), "capacity_basis": basis})
    return pd.DataFrame(rows)


def cost_scale(resources: pd.DataFrame, countries: pd.Series, reference: str) -> pd.DataFrame:
    """Price level of each country relative to the reference, on a shared basis."""
    table = resources.set_index("country_iso3")
    reference_row = table.loc[reference] if reference in table.index else None
    rows = []
    for iso3 in countries:
        scale, basis = np.nan, "No ICP price level"
        if iso3 in table.index and reference_row is not None:
            row = table.loc[iso3]
            for column, label in (("health_price_level", "ICP health"), ("gdp_price_level", "ICP GDP")):
                if pd.notna(row[column]) and pd.notna(reference_row[column]):
                    scale, basis = float(row[column] / reference_row[column]), label
                    break
        rows.append({"country_iso3": iso3, "cost_scale": scale, "cost_basis": basis})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ optimiser

@dataclass
class PlanResult:
    countries: pd.DataFrame
    steps: pd.DataFrame
    alignment_before: float
    alignment_after: float
    spent: float
    placements: int
    new_sites: int
    from_existing: int
    stopped_by: str
    excluded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _gains(sites: np.ndarray, burden: np.ndarray, deviation_sum: float, additions: np.ndarray) -> np.ndarray:
    """Alignment gain (score points) from adding ``additions[c]`` to country c alone.

    ``deviation_sum`` is the current sum of absolute share deviations. The
    score is 100 x (1 - deviation_sum / 2), so the gain is half the reduction
    in that sum, scaled to 100.
    """
    total = sites.sum()
    gains = np.empty(len(sites))
    for m in np.unique(additions):
        new_total = total + m
        deviation = np.abs(sites / new_total - burden)
        mask = additions == m
        changed = np.abs((sites[mask] + m) / new_total - burden[mask])
        gains[mask] = 50 * (deviation_sum - (deviation.sum() - deviation[mask] + changed))
    return gains


def _round_cost(allocation, available, hub_capacity, study_price, site_price):
    """Exact cost of a multi-country round: capacity first, then new sites."""
    used = allocation > 0
    units = allocation[used]
    extra = units - np.minimum(units, available[used])
    new_sites = np.ceil(extra / hub_capacity[used])
    return float(np.sum(units * study_price[used] + new_sites * site_price[used]))


def _best_round(sites, burden, deviation_sum, eligible, available, hub_capacity,
                study_price, site_price, remaining):
    """Lookahead for when no single placement improves alignment.

    With few placements in total, one more placement moves any country's share
    past its burden share, so every single step looks harmful even though the
    portfolio is far from aligned. A round of N placements, shared out by each
    country's gap to its burden-proportional target at the larger total, grows
    the portfolio enough to remove that integer effect. Rounds of 2, 4, 8, ...
    placements are compared on alignment gained per dollar.
    Returns (allocation or None, whether any round improves alignment at all).
    """
    total = sites.sum()
    best, best_ratio, any_positive = None, 0.0, False
    size = 2
    while size <= max(4 * total, 64):
        target_gap = np.where(eligible, np.maximum(burden * (total + size) - sites, 0), 0)
        if target_gap.sum() <= 0:
            break
        allocation = _largest_remainder(target_gap, size)
        trial = sites + allocation
        gain = 50 * (deviation_sum - np.abs(trial / trial.sum() - burden).sum())
        if gain > _TOLERANCE:
            any_positive = True
            cost = _round_cost(allocation, available, hub_capacity, study_price, site_price)
            if cost <= remaining and gain / max(cost, _TOLERANCE) > best_ratio:
                best, best_ratio = allocation, gain / max(cost, _TOLERANCE)
        size *= 2
    return best, any_positive


def optimise_plan(
    portfolio: pd.DataFrame,
    headroom: pd.Series,
    new_capacity: pd.Series,
    scale: pd.Series,
    study_cost: float,
    site_cost: float,
    budget: float | None,
    reuse_capacity: bool = True,
    max_placements: int = 50_000,
) -> PlanResult:
    """Allocate placements to maximise alignment per unit of cost.

    ``portfolio`` needs ``country_iso3``, ``trial_sites`` and ``burden_share``.
    ``headroom``, ``new_capacity`` and ``scale`` are indexed by ISO3. A country
    with no finite cost scale cannot receive placements. ``budget=None`` runs
    until no action improves alignment, which traces the full cost frontier.
    With ``reuse_capacity=False`` every placement establishes a new site.
    """
    required = {"country_iso3", "trial_sites", "burden_share"}
    missing = required.difference(portfolio.columns)
    if missing:
        raise ValueError(f"Portfolio is missing required columns: {', '.join(sorted(missing))}")
    if portfolio.empty:
        raise ValueError("Portfolio is empty")
    try:
        study_cost = float(study_cost)
        site_cost = float(site_cost)
        budget = None if budget is None else float(budget)
    except (TypeError, ValueError) as exc:
        raise ValueError("Costs and budget must be numeric") from exc
    if not np.isfinite(study_cost) or study_cost <= 0:
        raise ValueError("Study start-up cost must be a positive finite value")
    if not np.isfinite(site_cost) or site_cost < 0:
        raise ValueError("New-site cost must be a non-negative finite value")
    if budget is not None and (not np.isfinite(budget) or budget < 0):
        raise ValueError("Budget must be a non-negative finite value or None")

    iso3 = portfolio["country_iso3"].to_numpy()
    sites = np.nan_to_num(portfolio["trial_sites"].to_numpy(float), nan=0.0)
    burden = np.nan_to_num(portfolio["burden_share"].to_numpy(float), nan=0.0)
    if np.any(sites < 0) or np.any(burden < 0):
        raise ValueError("Trial activity and burden shares must be non-negative")
    if burden.sum() <= 0:
        raise ValueError("Portfolio has no positive measured burden")
    burden = burden / burden.sum()
    factor = scale.reindex(iso3).to_numpy(float)
    eligible = np.isfinite(factor) & (factor > 0) & (burden > 0)
    factor = np.where(eligible, factor, np.inf)
    spare = headroom.reindex(iso3).fillna(0).to_numpy(float) if reuse_capacity else np.zeros(len(iso3))
    hub_capacity = new_capacity.reindex(iso3).fillna(1).clip(lower=1).to_numpy(int) if reuse_capacity else np.ones(len(iso3), int)
    open_slots = np.zeros(len(iso3))

    existing_added = np.zeros(len(iso3), int)
    hub_added = np.zeros(len(iso3), int)
    sites_opened = np.zeros(len(iso3), int)
    spend = np.zeros(len(iso3))
    remaining = np.inf if budget is None else float(budget)
    original = sites.copy()
    before = alignment_score(sites, burden) if sites.sum() > 0 else 0.0
    # With no activity at all the distributions share nothing: alignment is 0.
    deviation_sum = float(np.abs(sites / sites.sum() - burden).sum()) if sites.sum() > 0 else 2.0
    ones = np.ones(len(sites), int)
    study_price = study_cost * factor
    log, placed, stopped_by = [], 0, "No further action improves alignment"

    while placed < max_placements:
        # Option A: one more study where capacity already exists.
        gain_one = _gains(sites, burden, deviation_sum, ones)
        cost_a = np.where((spare + open_slots) > 0, study_price, np.inf)
        # Option B: establish a site and fill it up to capacity or parity need.
        if reuse_capacity:
            need = np.ceil(np.maximum(sites_to_parity(sites, burden), 1) - _TOLERANCE).astype(int)
            fill = np.minimum(hub_capacity, need)
            gain_b = gain_one if fill.max() == 1 else _gains(sites, burden, deviation_sum, fill)
        else:
            fill, gain_b = ones, gain_one
        cost_b = (site_cost + fill * study_cost) * factor

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_a = np.where((gain_one > _TOLERANCE) & np.isfinite(cost_a) & (cost_a <= remaining),
                               gain_one / np.maximum(cost_a, _TOLERANCE), -np.inf)
            ratio_b = np.where((gain_b > _TOLERANCE) & np.isfinite(cost_b) & (cost_b <= remaining),
                               gain_b / np.maximum(cost_b, _TOLERANCE), -np.inf)
        best_a, best_b = int(np.argmax(ratio_a)), int(np.argmax(ratio_b))
        if np.isfinite(ratio_a[best_a]) or np.isfinite(ratio_b[best_b]):
            if ratio_a[best_a] >= ratio_b[best_b]:
                moves = [(best_a, False, 1)]
            else:
                moves = [(best_b, True, int(fill[best_b]))]
        else:
            allocation, positive = _best_round(
                sites, burden, deviation_sum, eligible, spare + open_slots, hub_capacity,
                study_price, site_cost * factor, remaining)
            if allocation is None:
                if budget is not None and (positive or np.any((gain_one > _TOLERANCE) & eligible)):
                    stopped_by = "Budget exhausted"
                break
            # Each unit uses capacity while it lasts, then opens a site (whose
            # remaining slots serve the next units), matching _round_cost.
            moves = [(int(c), None, 1) for c in np.flatnonzero(allocation) for _ in range(int(allocation[c]))]

        for c, new_site, units in moves:
            if new_site is None:
                new_site = (spare[c] + open_slots[c]) <= 0
            if new_site:
                cost, action = (site_cost + units * study_cost) * factor[c], "Establish new site"
                sites_opened[c] += 1
                hub_added[c] += units
                open_slots[c] += hub_capacity[c] - units
            else:
                cost, action = study_price[c], "Study at site with capacity"
                if spare[c] > 0:
                    spare[c] -= 1
                    existing_added[c] += 1
                else:
                    open_slots[c] -= 1
                    hub_added[c] += 1
            sites[c] += units
            spend[c] += cost
            remaining -= cost
            placed += units
            deviation_sum = float(np.abs(sites / sites.sum() - burden).sum())
            log.append((iso3[c], action, units, cost, 100 * (1 - 0.5 * deviation_sum)))
    else:
        stopped_by = "Placement limit reached"

    steps = pd.DataFrame(log, columns=["country_iso3", "action", "placements", "cost", "alignment"])
    steps["cumulative_cost"] = steps["cost"].cumsum()
    steps["cumulative_placements"] = steps["placements"].cumsum()

    share_before = original / original.sum() if original.sum() > 0 else np.zeros(len(original))
    share_after = sites / sites.sum() if sites.sum() > 0 else np.zeros(len(sites))
    countries = pd.DataFrame({
        "country_iso3": iso3,
        "trial_sites": original,
        "burden_share": burden,
        "placements_existing": existing_added,
        "placements_new_sites": hub_added,
        "new_sites": sites_opened,
        "placements": existing_added + hub_added,
        "cost": spend,
        "trial_sites_after": sites,
        "equity_ratio_before": share_before / burden,
        "equity_ratio_after": share_after / burden,
        "eligible": eligible,
    })
    excluded = sorted(iso3[~eligible & (burden > 0)].tolist())
    excluded_burden = float(burden[~eligible].sum())
    warnings = [
        "This is a deterministic equity-allocation heuristic, not a country feasibility or recruitment forecast.",
        "ClinicalTrials.gov location history is an activity proxy, not evidence of spare operational capacity."
    ]
    if excluded_burden > 0:
        warnings.append(f"Countries holding {excluded_burden:.1%} of measured burden were excluded because costs were unavailable.")
    return PlanResult(
        countries=countries,
        steps=steps,
        alignment_before=before,
        alignment_after=alignment_score(sites, burden),
        spent=float(spend.sum()),
        placements=int(placed),
        new_sites=int(sites_opened.sum()),
        from_existing=int(existing_added.sum()),
        stopped_by=stopped_by,
        excluded=excluded,
        warnings=warnings,
    )


def assign_to_facilities(capacity: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Place each country's reuse allocation at its facilities, largest headroom first.

    Concentrating on facilities with the largest historical-activity headroom
    keeps the number of sites involved as small as the data allow.
    """
    rows = []
    wanted = plan.set_index("country_iso3")["placements_existing"]
    for iso3, count in wanted[wanted > 0].items():
        pool = capacity[(capacity["country_iso3"] == iso3) & (capacity["headroom"] > 0)]
        pool = pool.sort_values(["headroom", "capacity", "facility_key"], ascending=[False, False, True])
        left = int(count)
        for row in pool.itertuples():
            if left <= 0:
                break
            take = min(int(row.headroom), left)
            rows.append({**row._asdict(), "assigned": take})
            left -= take
    columns = list(capacity.columns) + ["assigned"]
    return pd.DataFrame(rows).drop(columns=["Index"], errors="ignore") if rows else pd.DataFrame(columns=columns)


def research_cities(facilities: pd.DataFrame, start: int, end: int, recency_windows: int = 1) -> pd.DataFrame:
    """Cities with named facilities active in any tracked disease recently.

    "Recently" is the selected range plus the same number of earlier windows
    used for facility activity. These are where research infrastructure
    demonstrably exists, which makes them the evidence-based candidates for a
    new site.
    """
    length = int(end) - int(start) + 1
    named = facilities[~facilities["anonymised"] & (facilities["city"].astype(str).str.len() > 0)
                       & facilities["year"].between(int(start) - length * int(recency_windows), int(end))]
    cities = (named.groupby(["country_iso3", "city"])
              .agg(facilities=("facility_key", "nunique"), studies=("nct_id", "nunique"),
                   lat=("lat", "median"), lon=("lon", "median"))
              .reset_index()
              .sort_values(["country_iso3", "facilities", "studies"], ascending=[True, False, False]))
    return cities
