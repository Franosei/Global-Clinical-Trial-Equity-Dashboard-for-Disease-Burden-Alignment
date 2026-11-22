"""Scientific and data-contract validation for planning snapshots.

Run ``python -m py.validation`` after every data rebuild. A passing report means
the implemented analytical contract is internally consistent. It does not turn
registry activity into participant-level evidence or replace protocol-specific
country feasibility review.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .global_health_sources import DISEASE_PATTERNS, GHE_CAUSES
from .disease_validation import rules_sha256


ROOT = Path(__file__).resolve().parents[1]
CT_PATH = ROOT / "data" / "clean" / "ctgov_sites.parquet"
FACILITY_PATH = ROOT / "data" / "clean" / "ctgov_facilities.parquet"
WHO_PATH = ROOT / "data" / "clean" / "who_daly_selected.parquet"
GOLD_PATH = ROOT / "data" / "gold" / "country_interval_disease.parquet"
CFG_PATH = ROOT / "config" / "sources.yaml"
DISEASE_REVIEW_SUMMARY_PATH = ROOT / "data" / "validation" / "disease_matching" / "metrics_summary.json"


@dataclass(frozen=True)
class ValidationReport:
    checks: pd.DataFrame

    @property
    def analytical_ready(self) -> bool:
        return not (self.checks["status"] == "FAIL").any()

    @property
    def operational_ready(self) -> bool:
        return self.analytical_ready and not (self.checks["status"] == "WARN").any()

    def to_dict(self) -> dict:
        return {
            "analytical_ready": self.analytical_ready,
            "operational_ready": self.operational_ready,
            "checks": self.checks.to_dict("records"),
        }


def validate_snapshots() -> ValidationReport:
    rows: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str, warning: bool = False) -> None:
        rows.append({"check": name, "status": "PASS" if passed else ("WARN" if warning else "FAIL"),
                     "detail": detail})

    required = [CT_PATH, FACILITY_PATH, WHO_PATH, GOLD_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    add("Required snapshots", not missing, "All required snapshots exist" if not missing else f"Missing: {', '.join(missing)}")
    if missing:
        return ValidationReport(pd.DataFrame(rows))

    ct = pd.read_parquet(CT_PATH)
    has_type = "study_type" in ct
    types = set(ct["study_type"].dropna().astype(str).str.upper()) if has_type else set()
    add("Interventional-only numerator", has_type and types == {"INTERVENTIONAL"},
        f"Observed study types: {sorted(types) if has_type else 'column missing'}")
    expected_causes = set(DISEASE_PATTERNS)
    trial_causes = set(ct["cause_code"].astype(str))
    add("Trial disease coverage", trial_causes == expected_causes,
        f"Expected {len(expected_causes)} diseases; found {len(trial_causes)}")
    add("Non-negative trial activity", bool((pd.to_numeric(ct["trial_sites"], errors="coerce") >= 0).all()),
        f"Rows: {len(ct):,}")
    trial_key = ["country_iso3", "year", "cause_code", "phase", "study_type"] if has_type else []
    add("Unique trial aggregates", bool(trial_key) and not ct.duplicated(trial_key).any(),
        "One row per country-year-disease-phase-study type")

    facilities = pd.read_parquet(FACILITY_PATH, columns=["nct_id", "cause_code", "facility_key", "study_type"])
    facility_types = set(facilities["study_type"].dropna().astype(str).str.upper())
    add("Interventional-only capacity layer", facility_types == {"INTERVENTIONAL"},
        f"Observed facility study types: {sorted(facility_types)}")
    add("Unique facility records", not facilities.duplicated(["nct_id", "cause_code", "facility_key"]).any(),
        f"Facility rows: {len(facilities):,}")

    who = pd.read_parquet(WHO_PATH)
    expected_codes = {cause: "+".join(str(code) for code in codes) for cause, (codes, _) in GHE_CAUSES.items()}
    observed_codes = who.groupby("cause_code")["ghe_cause_code"].agg(lambda value: set(value.astype(str)))
    code_ok = all(observed_codes.get(cause, set()) == {code} for cause, code in expected_codes.items())
    add("WHO cause-code contract", code_ok, f"Verified {len(expected_codes)} disease mappings")
    dalys = pd.to_numeric(who["dalys"], errors="coerce")
    add("Finite non-negative burden", bool(np.isfinite(dalys).all() and (dalys >= 0).all()),
        f"{who['country_iso3'].nunique()} countries; {who['year'].nunique()} estimate years")

    gold = pd.read_parquet(GOLD_PATH)
    gold_key = ["country_iso3", "interval_start", "cause_code"]
    add("Unique gold cohorts", not gold.duplicated(gold_key).any(),
        "One row per country-interval-disease")
    expected_names = {cause: definition[0] for cause, definition in DISEASE_PATTERNS.items()}
    observed_names = gold.groupby("cause_code")["cause_name"].agg(lambda value: set(value.astype(str)))
    names_ok = all(observed_names.get(cause, set()) == {name} for cause, name in expected_names.items())
    add("Canonical disease labels", names_ok,
        "Display labels are metadata keyed by stable disease code")
    grouped = gold.groupby(["interval_start", "cause_code"])[["trial_share", "burden_share"]].sum()
    trial_sum_ok = np.isclose(grouped["trial_share"], 1, atol=1e-9) | np.isclose(grouped["trial_share"], 0, atol=1e-9)
    burden_sum_ok = np.isclose(grouped["burden_share"], 1, atol=1e-9)
    add("Probability-simplex shares", bool(trial_sum_ok.all() and burden_sum_ok.all()),
        f"Checked {len(grouped)} interval-disease cohorts")
    outside = gold["burden_dalys"].fillna(0).le(0)
    add("Burden-cohort denominator", bool(np.isclose(gold.loc[outside, "trial_share"], 0).all()),
        "Countries without measured burden contribute zero trial share")
    gold_types = set(gold.get("study_type", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    add("Gold provenance", gold_types == {"INTERVENTIONAL"}, f"Gold study types: {sorted(gold_types)}")

    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    verified_costs = bool(cfg.get("planner", {}).get("cost_inputs_verified", False))
    add("Protocol-specific costs", verified_costs,
        "Planner costs are finance-approved" if verified_costs else
        "Default costs are placeholders; replace and approve them before an operational decision", warning=True)
    review_cfg = cfg.get("disease_matching_review", {})
    review_verified = bool(review_cfg.get("independently_verified", False))
    review_summary = {}
    if DISEASE_REVIEW_SUMMARY_PATH.exists():
        try:
            review_summary = json.loads(DISEASE_REVIEW_SUMMARY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            review_summary = {}
    current_rules = review_summary.get("rules_sha256") == rules_sha256()
    review_ok = bool(
        review_verified
        and review_summary.get("validation_complete")
        and review_summary.get("diseases_with_precision_and_recall") == len(DISEASE_PATTERNS)
        and current_rules
    )
    if review_ok:
        review_detail = (
            f"Independent blinded review covers {len(DISEASE_PATTERNS)} diseases; "
            f"reviewers: {', '.join(review_summary.get('reviewers', []))}"
        )
    elif not review_summary:
        review_detail = "No scored blinded review is present; run python -m py.disease_validation --help"
    elif not current_rules:
        review_detail = "The scored review predates the current matching rules and must be repeated"
    elif not review_summary.get("review_complete"):
        review_detail = "The blinded review is incomplete; see docs/disease-matching-validation.md"
    elif not review_verified:
        review_detail = "A review report exists but independent verification is not approved in config/sources.yaml"
    else:
        review_detail = "The review has not met the operational validation contract"
    add("External disease-classifier validation", review_ok, review_detail, warning=True)
    add("Candidate-site feasibility", False,
        "Registry location history is an activity proxy, not evidence of activation, enrolment or spare capacity",
        warning=True)

    return ValidationReport(pd.DataFrame(rows))


def main() -> None:
    report = validate_snapshots()
    print(json.dumps(report.to_dict(), indent=2))
    if not report.analytical_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
