import math

import pandas as pd
import pytest

from py.disease_validation import (
    PREDICTION_COLUMNS,
    REVIEW_COLUMNS,
    load_review_queries,
    rules_frame,
    score_reviews,
)
from py.global_health_sources import DISEASE_PATTERNS


def test_review_queries_and_published_rules_cover_every_disease():
    assert set(load_review_queries()) == set(DISEASE_PATTERNS)
    rules = rules_frame()
    assert set(rules["cause_code"]) == set(DISEASE_PATTERNS)
    assert {"include_term", "excluded_phrase", "exclude_study_if"}.issuperset(
        set(rules["rule_type"])
    )
    assert not rules[["cause_code", "rule_type", "term"]].duplicated().any()


def test_scoring_uses_sealed_predictions_and_sampling_weights(tmp_path):
    labels = ["include", "exclude", "include", "exclude"]
    review = pd.DataFrame([
        {
            **{column: "" for column in REVIEW_COLUMNS},
            "review_id": f"r{index}",
            "cause_code": "HIV",
            "cause_name": "HIV/AIDS",
            "review_label": label,
            "reviewer": "Independent reviewer",
        }
        for index, label in enumerate(labels)
    ])
    predictions = pd.DataFrame([
        {
            "review_id": f"r{index}",
            "classifier_prediction": prediction,
            "sampling_stratum": "classifier_positive" if prediction else "classifier_negative",
            "stratum_population": 4 if prediction else 16,
            "stratum_sample_size": 2,
            "sample_weight": 2 if prediction else 8,
        }
        for index, prediction in enumerate([True, True, False, False])
    ], columns=PREDICTION_COLUMNS)
    review_path = tmp_path / "review.csv"
    predictions_path = tmp_path / "predictions.csv"
    review.to_csv(review_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    metrics, summary = score_reviews(
        review_path=review_path,
        predictions_path=predictions_path,
        metrics_path=tmp_path / "metrics.csv",
        summary_path=tmp_path / "summary.json",
        report_path=tmp_path / "report.md",
        min_reviewed_per_disease=2,
    )

    hiv = metrics.set_index("cause_code").loc["HIV"]
    assert math.isclose(hiv["precision"], 0.5)
    assert math.isclose(hiv["recall"], 0.2)
    assert hiv["near_matches_reviewed"] == 2
    assert not summary["review_complete"]  # The other 32 diseases have no labels.
    assert not summary["validation_complete"]


def test_scoring_rejects_non_protocol_labels(tmp_path):
    review = pd.DataFrame([{**{column: "" for column in REVIEW_COLUMNS},
                            "review_id": "r1", "cause_code": "HIV",
                            "review_label": "probably"}])
    predictions = pd.DataFrame([{
        "review_id": "r1", "classifier_prediction": True,
        "sampling_stratum": "classifier_positive", "stratum_population": 1,
        "stratum_sample_size": 1, "sample_weight": 1,
    }], columns=PREDICTION_COLUMNS)
    review_path = tmp_path / "review.csv"
    predictions_path = tmp_path / "predictions.csv"
    review.to_csv(review_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    with pytest.raises(ValueError, match="Invalid review_label"):
        score_reviews(
            review_path=review_path,
            predictions_path=predictions_path,
            metrics_path=tmp_path / "metrics.csv",
            summary_path=tmp_path / "summary.json",
            report_path=tmp_path / "report.md",
        )
