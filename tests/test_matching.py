"""Disease matching on real ClinicalTrials.gov condition labels.

Each case is a condition text as registered, and whether it should count for
the disease. The negative cases are false positives the earlier matcher made.
"""

import pytest

from py.ctgov import match_condition_to_cause
from py.global_health_sources import (
    DISEASE_GROUPS,
    DISEASE_PATTERNS,
    EXCLUDE_STUDY_IF,
    EXCLUDED_PHRASES,
    GHE_CAUSES,
)


def matches(cause: str, *conditions: str) -> bool:
    name, terms = DISEASE_PATTERNS[cause]
    return match_condition_to_cause(list(conditions), terms + [name],
                                    EXCLUDED_PHRASES.get(cause, []), EXCLUDE_STUDY_IF.get(cause, []))


CASES = [
    # Fixed false positives in the original 13 diseases
    ("KIDNEY", "Adrenal Insufficiency", False),
    ("KIDNEY", "Renal Cell Carcinoma", False),
    ("KIDNEY", "Chronic Kidney Disease Stage 3", True),
    ("KIDNEY", "End-Stage Renal Disease", True),
    ("HIV", "Hearing Aids", False),
    ("HIV", "HIV-1 Infection", True),
    ("HIV", "AIDS-Related Kaposi Sarcoma", True),
    ("RTI", "Respiratory Tract Infections", False),
    ("RTI", "Road Traffic Accidents", True),
    ("IHD", "Ischemic Stroke", False),
    ("IHD", "Coronary Artery Disease", True),
    ("IHD", "ST Elevation Myocardial Infarction", True),
    ("LUNG_CA", "Tracheal Stenosis", False),
    ("LUNG_CA", "Non-Small Cell Lung Cancer", True),
    ("DIAB", "Diabetes Insipidus", False),
    ("DIAB", "Type 2 Diabetes Mellitus", True),
    ("DIAB", "Type 2 DM", True),
    ("DIAB", "Pulmonary Oedema", False),
    ("STROKE", "Heat Stroke", False),
    ("STROKE", "Acute Ischemic Stroke", True),
    ("LRI", "Community-acquired Pneumonia", True),
    # New diseases
    ("MALARIA", "Plasmodium Falciparum Malaria", True),
    ("HEPATITIS", "Chronic Hepatitis B", True),
    ("HEPATITIS", "Hepatitis C, Chronic", True),
    ("HEPATITIS", "Alcoholic Hepatitis", False),
    ("HEPATITIS", "Autoimmune Hepatitis", False),
    ("NTD", "Visceral Leishmaniasis", True),
    ("NTD", "Dengue Fever", True),
    ("MEASLES", "Measles", True),
    ("SICKLE", "Sickle Cell Disease", True),
    ("MATERNAL", "Pre-Eclampsia", True),
    ("MATERNAL", "Postpartum Hemorrhage", True),
    ("BREAST_CA", "HER2-positive Breast Cancer", True),
    ("BREAST_CA", "Ductal Carcinoma In Situ", False),
    ("CERVIX_CA", "Cervical Spondylosis", False),
    ("CERVIX_CA", "Cervical Intraepithelial Neoplasia", False),
    ("CERVIX_CA", "Uterine Cervical Neoplasms", True),
    ("COLORECTAL_CA", "Metastatic Colorectal Cancer", True),
    ("LIVER_CA", "Hepatocellular Carcinoma", True),
    ("LIVER_CA", "Liver Metastases", False),
    ("PROSTATE_CA", "Metastatic Castration-resistant Prostate Cancer", True),
    ("DEPRESSION", "Major Depressive Disorder", True),
    ("DEPRESSION", "Respiratory Depression", False),
    ("DEPRESSION", "Bone Marrow Depression", False),
    ("ANXIETY", "Generalized Anxiety Disorder", True),
    ("ANXIETY", "Preoperative Anxiety", False),
    # WHO GHE 1120 is hypertensive heart disease, not uncomplicated
    # hypertension. The numerator must use the same clinical construct.
    ("HHD", "Essential Hypertension", False),
    ("HHD", "Hypertensive Heart Disease", True),
    ("HHD", "Pulmonary Arterial Hypertension", False),
    ("HHD", "Ocular Hypertension", False),
    # MeSH headings put the main word first.
    ("HHD", "Hypertension, Pulmonary", False),
    ("HHD", "Hypertension, Portal", False),
    ("DEPRESSION", "Depression, Respiratory", False),
    ("CIRRHOSIS", "Liver Cirrhosis", True),
    ("ASTHMA", "Severe Asthma", True),
    ("SELF_HARM", "Suicidal Ideation", True),
    ("CONGENITAL", "Congenital Heart Disease", True),
    ("CONGENITAL", "Congenital Adrenal Hyperplasia", False),
    # Every configured disease has at least one direct positive contract test.
    ("TB", "Pulmonary Tuberculosis", True),
    ("DIARR", "Rotavirus Gastroenteritis", True),
    ("DIARR", "Irritable Bowel Syndrome With Diarrhea", False),
    ("MENING", "Bacterial Meningitis", True),
    ("NEON", "Neonatal Sepsis", True),
    ("NEON", "Neonatal Diabetes", False),
    ("STOMACH_CA", "Gastric Adenocarcinoma", True),
    ("COPD", "Chronic Obstructive Pulmonary Disease", True),
    ("ALZH", "Alzheimer Disease", True),
]


@pytest.mark.parametrize("cause,condition,expected", CASES)
def test_condition_matching(cause, condition, expected):
    assert matches(cause, condition) is expected


def test_covid_pneumonia_is_not_a_lower_respiratory_infection():
    # WHO counts COVID-19 as its own cause.
    assert not matches("LRI", "COVID-19", "Pneumonia, Viral")
    assert matches("LRI", "Pneumonia, Viral")


def test_excluded_phrase_only_removes_that_phrase():
    # A study on both conditions still counts for the real disease.
    assert matches("DIAB", "Diabetes Insipidus", "Type 1 Diabetes")


def test_every_disease_is_defined_once_everywhere():
    grouped = [cause for _, causes in DISEASE_GROUPS for cause in causes]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(DISEASE_PATTERNS) == set(GHE_CAUSES)
    assert set(EXCLUDED_PHRASES) <= set(DISEASE_PATTERNS)
    assert set(EXCLUDE_STUDY_IF) <= set(DISEASE_PATTERNS)
    assert {cause for cause, _, expected in CASES if expected} == set(DISEASE_PATTERNS)


def test_composite_causes_use_disjoint_who_codes():
    for codes, _ in GHE_CAUSES.values():
        assert len(codes) == len(set(codes))
