import numpy as np
import pandas as pd

import py.indices as indices


def test_gold_trial_denominator_excludes_countries_without_measured_burden(tmp_path, monkeypatch):
    who = pd.DataFrame({
        "country_iso3": ["USA", "NGA"],
        "cause_code": ["X", "X"],
        # Display labels may evolve; the stable cause code is the join key.
        "cause_name": ["Legacy example label", "Legacy example label"],
        "year": [2023, 2023],
        "dalys": [100.0, 300.0],
        "daly_rate_per_100k": [1.0, 3.0],
    })
    trials = pd.DataFrame({
        "country_iso3": ["USA", "NGA", "TWN"],
        "year": [2023, 2023, 2023],
        "cause_code": ["X", "X", "X"],
        "cause_name": ["Example", "Example", "Example"],
        "phase": ["II", "II", "II"],
        "study_type": ["INTERVENTIONAL"] * 3,
        "trial_sites": [1, 3, 20],
    })
    who_path, trial_path, out_path = tmp_path / "who.parquet", tmp_path / "trials.parquet", tmp_path / "gold.parquet"
    who.to_parquet(who_path, index=False)
    trials.to_parquet(trial_path, index=False)
    monkeypatch.setattr(indices, "IN_WHO", who_path)
    monkeypatch.setattr(indices, "IN_CT", trial_path)
    monkeypatch.setattr(indices, "OUT_GOLD", out_path)
    monkeypatch.setattr(indices, "GOLD_DIR", tmp_path)

    out = indices.run_indices()
    latest = out[(out["interval_start"] == 2021) & (out["cause_code"] == "X")].set_index("country_iso3")
    assert len(latest) == 3
    assert set(latest["cause_name"]) == {"Example"}
    assert np.isclose(latest.loc["USA", "trial_share"], 0.25)
    assert np.isclose(latest.loc["NGA", "trial_share"], 0.75)
    assert latest.loc["TWN", "trial_share"] == 0
    assert np.isclose(latest["trial_share"].sum(), 1)


def test_gold_rejects_legacy_unfiltered_clinicaltrials_snapshot(tmp_path, monkeypatch):
    who = pd.DataFrame({
        "country_iso3": ["USA"], "cause_code": ["X"], "cause_name": ["Example"],
        "year": [2023], "dalys": [100.0], "daly_rate_per_100k": [1.0],
    })
    trials = pd.DataFrame({
        "country_iso3": ["USA"], "year": [2023], "cause_code": ["X"],
        "cause_name": ["Example"], "phase": ["NA"], "trial_sites": [1],
    })
    who_path, trial_path = tmp_path / "who.parquet", tmp_path / "trials.parquet"
    who.to_parquet(who_path, index=False)
    trials.to_parquet(trial_path, index=False)
    monkeypatch.setattr(indices, "IN_WHO", who_path)
    monkeypatch.setattr(indices, "IN_CT", trial_path)
    try:
        indices.run_indices()
    except RuntimeError as exc:
        assert "predates study-type filtering" in str(exc)
    else:
        raise AssertionError("legacy unfiltered snapshot was accepted")
