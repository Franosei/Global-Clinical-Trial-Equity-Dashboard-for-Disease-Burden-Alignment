import numpy as np
import pandas as pd

from py.ctgov import facility_is_anonymised
from py.indices import scope_for_years
from py.planner import (
    assign_to_facilities,
    cost_scale,
    facility_capacity,
    new_site_capacity,
    optimise_plan,
    research_cities,
)


def facility_rows() -> pd.DataFrame:
    """Hospital A started three studies in 2016 to 2018 and one in 2022."""
    rows = [
        ("N1", "X", 2016, "AAA", "AAA|city|hospital a", "Hospital A", False),
        ("N2", "X", 2017, "AAA", "AAA|city|hospital a", "Hospital A", False),
        ("N3", "X", 2018, "AAA", "AAA|city|hospital a", "Hospital A", False),
        ("N4", "X", 2022, "AAA", "AAA|city|hospital a", "Hospital A", False),
        ("N5", "X", 2017, "AAA", "AAA|city|research site|anon0", "Research Site", True),
        ("N6", "X", 2017, "AAA", "AAA|city|research site|anon0", "Research Site", True),
        # Activity after the selected years must never count as capacity.
        ("N7", "X", 2026, "AAA", "AAA|city|hospital b", "Hospital B", False),
    ]
    frame = pd.DataFrame(rows, columns=["nct_id", "cause_code", "year", "country_iso3",
                                        "facility_key", "facility", "anonymised"])
    frame["city"], frame["lat"], frame["lon"] = "City", 1.0, 2.0
    return frame


def test_capacity_is_peak_in_an_equal_length_window_minus_current_use():
    capacity = facility_capacity(facility_rows(), "X", 2021, 2025)
    assert capacity["facility"].tolist() == ["Hospital A"]
    row = capacity.iloc[0]
    # Best 5-year window is 2014-2018 or later ones holding all three studies.
    assert (row.capacity, row.used, row.headroom) == (3, 1, 2)


def test_capacity_window_follows_the_selected_range_length():
    row = facility_capacity(facility_rows(), "X", 2022, 2023).iloc[0]
    # In any two-year window the facility has started at most two studies.
    assert (row.capacity, row.used, row.headroom) == (2, 1, 1)


def test_inactive_facility_offers_no_headroom():
    # 2024 to 2025 plus the previous two years (2022 to 2023) still sees N4.
    assert facility_capacity(facility_rows(), "X", 2024, 2025).iloc[0].active
    # A facility last active in 2022 is stale for 2025 alone with one window back.
    stale = facility_capacity(facility_rows(), "X", 2025, 2025).iloc[0]
    assert not stale.active and stale.headroom == 0


def test_research_cities_respect_the_selected_years():
    cities = research_cities(facility_rows(), 2021, 2025)
    assert cities["facilities"].tolist() == [1]  # Hospital B (2026) is in the future
    assert research_cities(facility_rows(), 2030, 2030).empty


def test_any_year_range_builds_a_consistent_cohort():
    ct = pd.DataFrame({
        "country_iso3": ["USA", "USA", "NGA", "NGA"],
        "year": [2023, 2026, 2024, 2019],
        "cause_code": ["X"] * 4,
        "trial_sites": [8.0, 5.0, 2.0, 9.0],
    })
    who = pd.DataFrame({
        "country_iso3": ["USA", "NGA", "USA", "NGA"],
        "cause_code": ["X"] * 4, "cause_name": ["X"] * 4,
        "year": [2023, 2023, 2015, 2015],
        "dalys": [100.0, 300.0, 50.0, 50.0],
        "daly_rate_per_100k": [1.0, 1.0, 1.0, 1.0],
    })
    scope = scope_for_years(ct, who, "X", 2023, 2026).set_index("country_iso3")
    assert scope.loc["USA", "trial_sites"] == 13 and scope.loc["NGA", "trial_sites"] == 2
    assert np.isclose(scope["trial_share"].sum(), 1) and np.isclose(scope["burden_share"].sum(), 1)
    assert scope.loc["NGA", "burden_share"] == 0.75
    assert scope.loc["NGA", "burden_years"] == "2023"
    # A range with no WHO estimate year inside uses the latest earlier one.
    later = scope_for_years(ct, who, "X", 2024, 2026)
    assert set(later["burden_years"]) == {"2023"}


def test_anonymised_sponsor_listings_are_detected():
    assert facility_is_anonymised("GSK Investigational Site")
    assert facility_is_anonymised("For additional information regarding investigative sites, call")
    assert not facility_is_anonymised("Clinical Research Center of Nigeria")


def test_new_site_capacity_falls_back_from_country_to_income_group():
    capacity = pd.DataFrame({
        "country_iso3": ["AAA"] * 5 + ["BBB"],
        "capacity": [1, 2, 3, 4, 5, 9],
    })
    resources = pd.DataFrame({"country_iso3": ["AAA", "BBB", "CCC"],
                              "income_group": ["Low income", "Low income", "Low income"]})
    result = new_site_capacity(capacity, resources, pd.Series(["AAA", "CCC"]), 0.5, 5).set_index("country_iso3")
    assert result.loc["AAA", "new_site_capacity"] == 3
    assert result.loc["AAA", "capacity_basis"] == "Country"
    assert result.loc["CCC", "new_site_capacity"] == 3  # observed median of six values
    assert result.loc["CCC", "capacity_basis"].startswith("Income group")


def test_cost_scale_uses_one_price_basis_and_never_imputes():
    resources = pd.DataFrame({
        "country_iso3": ["USA", "NGA", "TWN"],
        "health_price_level": [200.0, 50.0, np.nan],
        "gdp_price_level": [160.0, 40.0, np.nan],
    })
    scale = cost_scale(resources, pd.Series(["NGA", "TWN"]), "USA").set_index("country_iso3")
    assert scale.loc["NGA", "cost_scale"] == 0.25
    assert np.isnan(scale.loc["TWN", "cost_scale"])


def portfolio() -> pd.DataFrame:
    return pd.DataFrame({
        "country_iso3": ["HIGH", "LOWA", "LOWB"],
        "trial_sites": [90.0, 5.0, 5.0],
        "burden_share": [0.2, 0.4, 0.4],
    })


def series(values: dict) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_reusing_capacity_reaches_more_alignment_for_the_same_budget():
    common = dict(
        portfolio=portfolio(),
        headroom=series({"LOWA": 10, "LOWB": 0}),
        new_capacity=series({"HIGH": 4, "LOWA": 4, "LOWB": 4}),
        scale=series({"HIGH": 1.0, "LOWA": 0.25, "LOWB": 0.25}),
        study_cost=25_000,
        site_cost=100_000,
        budget=1_000_000,
    )
    smart = optimise_plan(**common)
    naive = optimise_plan(**common, reuse_capacity=False)
    assert smart.spent <= 1_000_000 and naive.spent <= 1_000_000
    assert smart.alignment_after > naive.alignment_after > smart.alignment_before
    assert smart.new_sites < naive.new_sites
    assert smart.from_existing == 10  # all measured headroom is used first
    assert smart.countries.set_index("country_iso3").loc["HIGH", "placements"] == 0


def test_country_without_price_level_is_excluded_not_estimated():
    plan = optimise_plan(portfolio(), series({}), series({}), series({"LOWA": 1.0}),
                         25_000, 100_000, 2_000_000)
    by_country = plan.countries.set_index("country_iso3")
    assert by_country.loc["LOWB", "placements"] == 0
    assert "LOWB" in plan.excluded


def test_unlimited_budget_stops_when_alignment_cannot_improve():
    plan = optimise_plan(portfolio(), series({"LOWA": 5, "LOWB": 5}), series({"LOWA": 3, "LOWB": 3}),
                         series({"HIGH": 1, "LOWA": 1, "LOWB": 1}), 1, 4, None)
    assert plan.stopped_by == "No further action improves alignment"
    assert plan.alignment_after > 95
    assert plan.steps["alignment"].is_monotonic_increasing


def test_facility_assignment_concentrates_on_largest_headroom():
    capacity = pd.DataFrame({
        "country_iso3": ["AAA", "AAA", "AAA"], "facility_key": ["a", "b", "c"],
        "facility": ["A", "B", "C"], "city": ["X"] * 3, "lat": [0.0] * 3, "lon": [0.0] * 3,
        "capacity": [6, 2, 3], "used": [1, 0, 0], "active": [True] * 3, "headroom": [5, 2, 3],
    })
    plan = pd.DataFrame({"country_iso3": ["AAA"], "placements_existing": [7]})
    assigned = assign_to_facilities(capacity, plan)
    assert assigned["facility"].tolist() == ["A", "C"]
    assert assigned["assigned"].tolist() == [5, 2]


def test_lookahead_escapes_integer_local_optimum_in_small_portfolios():
    # Nine placements in total: any single addition overshoots a country's
    # burden share, yet adding to several under-served countries together helps.
    countries = [f"C{i:02d}" for i in range(20)]
    small = pd.DataFrame({
        "country_iso3": countries,
        "trial_sites": [9.0] + [0.0] * 19,
        "burden_share": [0.05] + [0.95 / 19] * 19,
    })
    plan = optimise_plan(small, series({}), series({}), series({c: 1.0 for c in countries}),
                         1, 1, 1_000)
    assert plan.alignment_after > 90
    assert plan.stopped_by != "No further action improves alignment" or plan.alignment_after > 95
