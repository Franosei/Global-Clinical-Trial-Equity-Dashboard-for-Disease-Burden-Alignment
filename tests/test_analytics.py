from datetime import datetime, timezone

import numpy as np
import pandas as pd

from py.analytics import (
    alignment_score,
    allocate_sites,
    forecast_trial_sites,
    sites_to_parity,
)
from py.indices import VALID_COUNTRY_ISO3, assign_interval
from py.ctgov import match_condition_to_cause, unique_tokens_for_cause


def sample_portfolio() -> pd.DataFrame:
    return pd.DataFrame({
        "country_iso3": ["AAA", "BBB", "CCC"],
        "trial_sites": [10.0, 0.0, 5.0],
        "trial_share": [2 / 3, 0, 1 / 3],
        "burden_share": [0.2, 0.5, 0.3],
    })


def test_exact_sites_to_individual_parity_accounts_for_new_denominator():
    needed = sites_to_parity(sample_portfolio()["trial_sites"], sample_portfolio()["burden_share"])
    assert np.allclose(needed, [0, 15, 0])
    assert (0 + needed[1]) / (15 + needed[1]) == 0.5


def test_integer_allocation_preserves_requested_portfolio_size():
    result = allocate_sites(sample_portfolio(), 101, "Equity gap")
    assert result["allocated_sites"].sum() == 101
    assert np.issubdtype(result["allocated_sites"].dtype, np.integer)
    assert np.isclose(result["trial_share_after"].sum(), 1)


def test_alignment_score_bounds_and_identity():
    assert alignment_score([0.2, 0.8], [0.2, 0.8]) == 100
    assert 0 <= alignment_score([1, 0], [0, 1]) <= 100


def test_forecast_has_reproducible_ordered_intervals():
    annual = pd.DataFrame({
        "cause_code": ["X"] * 8,
        "year": list(range(2017, 2025)),
        "trial_sites": [100, 108, 111, 125, 121, 139, 151, 163],
    })
    result = forecast_trial_sites(annual, "X", horizon=3, seed=7)
    future = result[result["kind"] == "Forecast"]
    assert len(future) == 3
    assert (future["lower"] <= future["estimate"]).all()
    assert (future["estimate"] <= future["upper"]).all()


def test_interval_configuration_reaches_current_year():
    interval = assign_interval(datetime.now(timezone.utc).year)
    assert interval is not None
    assert interval[0] <= datetime.now(timezone.utc).year <= interval[1]


def test_short_disease_acronyms_use_token_boundaries():
    assert match_condition_to_cause(["Type 2 DM"], ["dm"])
    assert not match_condition_to_cause(["Pulmonary oedema"], ["dm"])
    assert "dm" not in unique_tokens_for_cause(["diabetes", "dm"], "Diabetes mellitus")


def test_gold_layer_excludes_who_regional_aggregates():
    assert not {"AFR", "AMR", "EMR", "EUR", "GLO", "SEA", "WHO", "WPR"}.intersection(VALID_COUNTRY_ISO3)


def test_forecast_reports_its_fitted_model():
    years = list(range(2017, 2025))
    annual = pd.DataFrame({"cause_code": ["X"] * 8, "year": years,
                           "trial_sites": [round(np.expm1(3 + 0.1 * i)) for i in range(8)]})
    model = forecast_trial_sites(annual, "X").attrs["model"]
    assert (model["first_year"], model["last_year"], model["n_years"]) == (2017, 2024, 8)
    assert abs(model["slope"] - 0.1) < 0.01 and model["r_squared"] > 0.99
