"""Blinded manual validation of the ClinicalTrials.gov disease classifier.

The production matcher is deterministic, but contract tests are not estimates
of classification performance. This module creates a reproducible review set
from a separately published ClinicalTrials.gov search frame, keeps classifier
predictions out of the reviewer file, and reports weighted precision and recall
by disease after human labels are entered.

Commands::

    python -m py.disease_validation rules
    python -m py.disease_validation sample --per-stratum 20
    python -m py.disease_validation score

The sample command requires network access. The score command never infers or
fills a human label. ``include`` means the registered condition falls within
the dashboard disease construct; ``exclude`` means it does not; ``uncertain``
is omitted from the performance estimates and reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .ctgov import (
    BASE_CT_V2,
    INCLUDED_STUDY_TYPES,
    extract_condition_texts,
    http_get_json,
    match_condition_to_cause,
)
from .global_health_sources import (
    DISEASE_PATTERNS,
    EXCLUDED_PHRASES,
    EXCLUDE_STUDY_IF,
    GHE_CAUSES,
)


ROOT = Path(__file__).resolve().parents[1]
QUERY_PATH = ROOT / "config" / "disease_review_queries.yaml"
OUT_DIR = ROOT / "data" / "validation" / "disease_matching"
REVIEW_PATH = OUT_DIR / "review_sample.csv"
PREDICTIONS_PATH = OUT_DIR / "review_predictions.csv"
METADATA_PATH = OUT_DIR / "sample_metadata.json"
RULES_PATH = OUT_DIR / "matching_rules.csv"
METRICS_PATH = OUT_DIR / "metrics.csv"
SUMMARY_PATH = OUT_DIR / "metrics_summary.json"
REPORT_PATH = ROOT / "docs" / "disease-matching-validation.md"

REVIEW_COLUMNS = [
    "review_id", "cause_code", "cause_name", "target_definition", "nct_id",
    "brief_title", "conditions", "source_queries", "review_label", "reviewer",
    "review_date", "review_notes",
]
PREDICTION_COLUMNS = [
    "review_id", "classifier_prediction", "sampling_stratum",
    "stratum_population", "stratum_sample_size", "sample_weight",
]
VALID_LABELS = {"include": True, "exclude": False}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rules_sha256() -> str:
    rows = rules_frame().to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(rows.encode("utf-8")).hexdigest()


def rules_frame() -> pd.DataFrame:
    """Return every production inclusion term and exclusion as tidy data."""
    rows: list[dict[str, str]] = []
    for cause_code, (cause_name, terms) in DISEASE_PATTERNS.items():
        for term in terms:
            rows.append({"cause_code": cause_code, "cause_name": cause_name,
                         "rule_type": "include_term", "term": term})
        for term in EXCLUDED_PHRASES.get(cause_code, []):
            rows.append({"cause_code": cause_code, "cause_name": cause_name,
                         "rule_type": "excluded_phrase", "term": term})
        for term in EXCLUDE_STUDY_IF.get(cause_code, []):
            rows.append({"cause_code": cause_code, "cause_name": cause_name,
                         "rule_type": "exclude_study_if", "term": term})
    return pd.DataFrame(rows)


def publish_rules(path: Path = RULES_PATH) -> pd.DataFrame:
    frame = rules_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_review_queries(path: Path = QUERY_PATH) -> dict[str, list[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = {str(code): [str(value) for value in values]
               for code, values in raw.items()}
    missing = set(DISEASE_PATTERNS).difference(queries)
    extra = set(queries).difference(DISEASE_PATTERNS)
    empty = {code for code, values in queries.items() if not values}
    if missing or extra or empty:
        raise ValueError(
            f"Review-query coverage mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}, empty={sorted(empty)}"
        )
    return queries


def predict(cause_code: str, conditions: Iterable[str]) -> bool:
    cause_name, terms = DISEASE_PATTERNS[cause_code]
    return match_condition_to_cause(
        list(conditions), terms + [cause_name],
        EXCLUDED_PHRASES.get(cause_code, ()),
        EXCLUDE_STUDY_IF.get(cause_code, ()),
    )


def _study_record(study: dict, query: str) -> dict | None:
    protocol = study.get("protocolSection", {}) or {}
    identity = protocol.get("identificationModule", {}) or {}
    design = protocol.get("designModule", {}) or {}
    nct_id = identity.get("nctId")
    study_type = str(design.get("studyType") or "").upper()
    if not nct_id or study_type not in INCLUDED_STUDY_TYPES:
        return None
    conditions = list(dict.fromkeys(extract_condition_texts(study)))
    return {
        "nct_id": str(nct_id),
        "brief_title": str(identity.get("briefTitle") or ""),
        "conditions_list": conditions,
        "source_queries_list": [query],
    }


def _query_studies(query: str, max_records: int, page_size: int = 100) -> Iterable[dict]:
    """Yield at most ``max_records`` API records in the API's relevance order."""
    token = ""
    yielded = 0
    fields = ",".join([
        "protocolSection.identificationModule",
        "protocolSection.conditionsModule",
        "protocolSection.designModule.studyType",
        "derivedSection.conditionBrowseModule",
    ])
    use_fields = True
    while yielded < max_records:
        params = {
            "query.cond": query,
            "pageSize": str(min(page_size, max_records - yielded)),
            "format": "json",
        }
        if use_fields:
            params["fields"] = fields
        if token:
            params["pageToken"] = token
        try:
            page = http_get_json(BASE_CT_V2 + "studies", params)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 400 and use_fields:
                use_fields = False
                continue
            raise
        studies = page.get("studies", []) or []
        for study in studies:
            yield study
            yielded += 1
            if yielded >= max_records:
                return
        token = str(page.get("nextPageToken") or "")
        if not token:
            return


def _merge_candidate(pool: dict[str, dict], candidate: dict) -> None:
    existing = pool.get(candidate["nct_id"])
    if existing is None:
        pool[candidate["nct_id"]] = candidate
        return
    existing["source_queries_list"] = list(dict.fromkeys(
        existing["source_queries_list"] + candidate["source_queries_list"]
    ))
    existing["conditions_list"] = list(dict.fromkeys(
        existing["conditions_list"] + candidate["conditions_list"]
    ))


def generate_sample(
    per_stratum: int = 20,
    max_records_per_query: int = 250,
    seed: int = 20261122,
    force: bool = False,
    review_path: Path = REVIEW_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reviewer and prediction files; predictions never enter the former."""
    if per_stratum < 1 or max_records_per_query < 1:
        raise ValueError("per_stratum and max_records_per_query must be positive")
    if (review_path.exists() or predictions_path.exists()) and not force:
        raise FileExistsError(
            "A review sample already exists. Use --force only to discard that review."
        )

    queries_by_cause = load_review_queries()
    review_rows: list[dict] = []
    prediction_rows: list[dict] = []
    populations: dict[str, dict[str, int]] = {}

    for cause_code, (cause_name, _) in DISEASE_PATTERNS.items():
        pool: dict[str, dict] = {}
        queries = queries_by_cause[cause_code]
        for query in queries:
            for study in _query_studies(query, max_records=max_records_per_query):
                candidate = _study_record(study, query)
                if candidate:
                    _merge_candidate(pool, candidate)

        for candidate in pool.values():
            candidate["classifier_prediction"] = predict(
                cause_code, candidate["conditions_list"]
            )

        strata = {
            True: [row for row in pool.values() if row["classifier_prediction"]],
            False: [row for row in pool.values() if not row["classifier_prediction"]],
        }
        populations[cause_code] = {
            "classifier_positive": len(strata[True]),
            "classifier_negative": len(strata[False]),
        }
        for prediction, candidates in strata.items():
            stratum_name = "classifier_positive" if prediction else "classifier_negative"
            sample_size = min(per_stratum, len(candidates))
            rng = random.Random(f"{seed}:{cause_code}:{stratum_name}")
            sampled = rng.sample(candidates, sample_size) if sample_size else []
            weight = len(candidates) / sample_size if sample_size else math.nan
            for candidate in sampled:
                review_id = hashlib.sha256(
                    f"{cause_code}|{candidate['nct_id']}".encode("utf-8")
                ).hexdigest()[:16]
                review_rows.append({
                    "review_id": review_id,
                    "cause_code": cause_code,
                    "cause_name": cause_name,
                    "target_definition": GHE_CAUSES[cause_code][1],
                    "nct_id": candidate["nct_id"],
                    "brief_title": candidate["brief_title"],
                    "conditions": " | ".join(candidate["conditions_list"]),
                    "source_queries": " | ".join(candidate["source_queries_list"]),
                    "review_label": "",
                    "reviewer": "",
                    "review_date": "",
                    "review_notes": "",
                })
                prediction_rows.append({
                    "review_id": review_id,
                    "classifier_prediction": prediction,
                    "sampling_stratum": stratum_name,
                    "stratum_population": len(candidates),
                    "stratum_sample_size": sample_size,
                    "sample_weight": weight,
                })

    review = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS).sort_values(
        ["cause_code", "review_id"]
    )
    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).sort_values(
        "review_id"
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    publish_rules(review_path.parent / RULES_PATH.name)
    metadata = {
        "generated_at_utc": _now(),
        "seed": seed,
        "per_stratum": per_stratum,
        "max_records_per_query": max_records_per_query,
        "study_types": sorted(INCLUDED_STUDY_TYPES),
        "search_frame": str(QUERY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "rules_sha256": rules_sha256(),
        "populations": populations,
        "review_rows": len(review),
        "blinded": "Classifier predictions are stored only in review_predictions.csv.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return review, predictions


def _normalise_labels(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.lower()
    allowed = set(VALID_LABELS).union({"", "uncertain"})
    invalid = sorted(set(values).difference(allowed))
    if invalid:
        raise ValueError(
            f"Invalid review_label values {invalid}; use include, exclude, uncertain, or blank"
        )
    return values


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def score_reviews(
    review_path: Path = REVIEW_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    metrics_path: Path = METRICS_PATH,
    summary_path: Path = SUMMARY_PATH,
    report_path: Path = REPORT_PATH,
    min_reviewed_per_disease: int = 10,
    independently_verified: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Score human labels and publish a per-disease report.

    Weighted counts expand each reviewed record to its classifier-stratum
    population. This corrects the deliberate over-sampling of near-matches.
    Estimates apply to the published query frame, not to every registry record.
    """
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(predictions_path)
    missing_review = set(REVIEW_COLUMNS).difference(review.columns)
    missing_predictions = set(PREDICTION_COLUMNS).difference(predictions.columns)
    if missing_review or missing_predictions:
        raise ValueError(
            f"Missing review columns={sorted(missing_review)}; "
            f"prediction columns={sorted(missing_predictions)}"
        )
    if review["review_id"].duplicated().any() or predictions["review_id"].duplicated().any():
        raise ValueError("review_id must be unique in both files")
    if set(review["review_id"]) != set(predictions["review_id"]):
        raise ValueError("Reviewer and prediction files contain different review IDs")

    review["review_label"] = _normalise_labels(review["review_label"])
    merged = review.merge(predictions, on="review_id", validate="one_to_one")
    merged["classifier_prediction"] = merged["classifier_prediction"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    )
    merged["sample_weight"] = pd.to_numeric(merged["sample_weight"], errors="raise")
    merged["is_labeled"] = merged["review_label"].isin(VALID_LABELS)
    merged["actual_positive"] = merged["review_label"].map(VALID_LABELS)

    rows: list[dict] = []
    for cause_code, (cause_name, _) in DISEASE_PATTERNS.items():
        disease = merged[merged["cause_code"] == cause_code]
        labeled = disease[disease["is_labeled"]].copy()
        pred = labeled["classifier_prediction"]
        actual = labeled["actual_positive"].astype(bool) if len(labeled) else pd.Series(dtype=bool)
        weight = labeled["sample_weight"]
        tp = float(weight[pred & actual].sum())
        fp = float(weight[pred & ~actual].sum())
        fn = float(weight[~pred & actual].sum())
        tn = float(weight[~pred & ~actual].sum())
        rows.append({
            "cause_code": cause_code,
            "cause_name": cause_name,
            "sampled": len(disease),
            "reviewed": len(labeled),
            "uncertain": int((disease["review_label"] == "uncertain").sum()),
            "unreviewed": int((disease["review_label"] == "").sum()),
            "weighted_tp": tp,
            "weighted_fp": fp,
            "weighted_fn": fn,
            "weighted_tn": tn,
            "precision": _safe_ratio(tp, tp + fp),
            "recall": _safe_ratio(tp, tp + fn),
            "near_matches_reviewed": int((~pred).sum()) if len(labeled) else 0,
        })
    metrics = pd.DataFrame(rows)

    has_metrics = metrics["precision"].notna() & metrics["recall"].notna()
    enough = metrics["reviewed"] >= min_reviewed_per_disease
    named_reviewers = review.loc[review["review_label"].isin(VALID_LABELS), "reviewer"].str.strip()
    reviewer_complete = bool(len(named_reviewers)) and bool(named_reviewers.ne("").all())
    review_complete = bool((has_metrics & enough).all()) and reviewer_complete
    validation_complete = review_complete and independently_verified
    summary = {
        "generated_at_utc": _now(),
        "diseases_expected": len(DISEASE_PATTERNS),
        "diseases_with_precision_and_recall": int(has_metrics.sum()),
        "min_reviewed_per_disease": min_reviewed_per_disease,
        "review_complete": review_complete,
        "independently_verified": independently_verified,
        "validation_complete": validation_complete,
        "reviewers": sorted(set(named_reviewers[named_reviewers.ne("")])),
        "rules_sha256": rules_sha256(),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    publish_rules(metrics_path.parent / RULES_PATH.name)
    _write_report(metrics, summary, report_path)
    return metrics, summary


def _metric(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def _write_report(metrics: pd.DataFrame, summary: dict, path: Path) -> None:
    status = "COMPLETE" if summary["validation_complete"] else "INCOMPLETE"
    lines = [
        "# Disease matching validation",
        "",
        f"**Status: {status}.** Generated {summary['generated_at_utc']}. ",
        "The status becomes complete only when every disease has enough usable labels, "
        "precision and recall are estimable, reviewers are named, and the project owner "
        "records that the review was independently verified.",
        "",
        "## Method",
        "",
        "ClinicalTrials.gov interventional records are retrieved through the broad, "
        "versioned queries in `config/disease_review_queries.yaml`. Records are stratified "
        "by the production classifier prediction, and a seeded random sample is taken from "
        "each stratum. Reviewers see the NCT ID, title, registered conditions and target "
        "definition, but not the prediction. `include` and `exclude` labels are scored; "
        "blank and `uncertain` labels are not.",
        "",
        "Precision and recall use inverse-sampling weights because classifier-negative "
        "near-matches are deliberately over-sampled. The estimates describe the published "
        "ClinicalTrials.gov search frame, not all possible registry terminology. Contract "
        "tests and records used to tune the rules must not be added to the blinded sample.",
        "",
        "## Results by disease",
        "",
        "| Disease | Reviewed | Near matches | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.cause_name} | {row.reviewed} | {row.near_matches_reviewed} | "
            f"{_metric(row.precision)} | {_metric(row.recall)} |"
        )
    lines.extend([
        "",
        "## Published matching rules",
        "",
        "The machine-readable source is `data/validation/disease_matching/matching_rules.csv`. "
        "`exclude phrase` removes only that phrase before matching; `exclude study if` "
        "rejects the whole study for that disease.",
        "",
        "| Disease | Include terms | Excluded phrases | Exclude whole study if |",
        "| --- | --- | --- | --- |",
    ])
    for cause_code, (cause_name, terms) in DISEASE_PATTERNS.items():
        include = "; ".join(terms)
        phrases = "; ".join(EXCLUDED_PHRASES.get(cause_code, ())) or "—"
        whole = "; ".join(EXCLUDE_STUDY_IF.get(cause_code, ())) or "—"
        lines.append(f"| {cause_name} | {include} | {phrases} | {whole} |")
    lines.extend([
        "",
        "## Reproduction",
        "",
        "```powershell",
        "py -m py.disease_validation sample --per-stratum 20",
        "# Complete review_sample.csv without opening review_predictions.csv.",
        "py -m py.disease_validation score --min-reviewed-per-disease 10",
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _verified_from_config() -> bool:
    config = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
    return bool(config.get("disease_matching_review", {}).get("independently_verified", False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rules", help="Publish the exact production term and exclusion lists")
    sample = sub.add_parser("sample", help="Fetch and create a blinded review sample")
    sample.add_argument("--per-stratum", type=int, default=20)
    sample.add_argument("--max-records-per-query", type=int, default=250)
    sample.add_argument("--seed", type=int, default=20261122)
    sample.add_argument("--force", action="store_true")
    score = sub.add_parser("score", help="Score completed human labels and write the report")
    score.add_argument("--min-reviewed-per-disease", type=int, default=10)
    args = parser.parse_args()

    if args.command == "rules":
        frame = publish_rules()
        print(f"Wrote {RULES_PATH} ({len(frame)} rules)")
    elif args.command == "sample":
        review, predictions = generate_sample(
            per_stratum=args.per_stratum,
            max_records_per_query=args.max_records_per_query,
            seed=args.seed,
            force=args.force,
        )
        print(f"Wrote blinded review: {REVIEW_PATH} ({len(review)} records)")
        print(f"Wrote sealed predictions: {PREDICTIONS_PATH} ({len(predictions)} records)")
    else:
        metrics, summary = score_reviews(
            min_reviewed_per_disease=args.min_reviewed_per_disease,
            independently_verified=_verified_from_config(),
        )
        print(metrics[["cause_code", "reviewed", "precision", "recall"]].to_string(index=False))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
