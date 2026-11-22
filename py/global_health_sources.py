# py/global_health_sources.py
"""
WHO Global Health Estimates (GHE) burden for the 13 tracked diseases.

Source: WHO Global Health Estimates 2023 summary tables, "DALY by country,
age and sex", one workbook per estimate year (2000, 2010, 2015, 2019 to 2023).
https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates

Each disease maps to exactly one GHE cause code, so no indicator is chosen by
keyword. The GHO OData API is not used because it publishes cause-specific
DALYs only for regions, not countries.

Output (Parquet): data/clean/who_daly_selected.parquet
  country_iso3, cause_code, cause_name, year,
  dalys                absolute DALYs, both sexes, all ages
  population           WHO GHE population for that year
  daly_rate_per_100k   dalys / population x 100,000
  who_data_quality     WHO GHE country data-quality colour code (see WHO notes)
  ghe_cause_code, ghe_cause_name, source

Run:
  python -m py.global_health_sources
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import unicodedata
from typing import Dict, List, Tuple

import openpyxl
import pandas as pd
import pycountry
import requests
from requests.adapters import HTTPAdapter, Retry
import yaml

# ------------------------ Paths & Config ------------------------

REPO_ROOT   = Path(__file__).resolve().parents[1]
CFG_PATH    = REPO_ROOT / "config" / "sources.yaml"
DATA_DIR    = REPO_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw" / "who_ghe"
CLEAN_DIR   = DATA_DIR / "clean"
OUT_PARQUET = CLEAN_DIR / "who_daly_selected.parquet"

CFG = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
GHE = CFG["who_ghe"]
GHE_YEARS: List[int] = [int(year) for year in GHE["years"]]
GHE_URL_TEMPLATE: str = GHE["daly_by_country_url"]
GHE_RELEASE: str = str(GHE["release"])

VALID_ISO3 = frozenset(country.alpha_3 for country in pycountry.countries)

# ------------------------ Disease dictionary ------------------------
#
# Each disease has: a display name, ClinicalTrials.gov condition terms, a group
# for the dashboard menu, and one or more WHO GHE cause codes. Composite causes
# sum disjoint GHE leaves, so no burden is counted twice within a disease.
#
# Matching rules (py/ctgov.py): a term must start at a word boundary; a term
# whose last word has three characters or fewer ("hepatitis b", "dm") must also
# end at one. Phrases in EXCLUDED_PHRASES are removed from condition text before
# matching, and a study mentioning any EXCLUDE_STUDY_IF term is not counted for
# that disease. The order of terms matters: the first ones are also the search
# queries sent to ClinicalTrials.gov.

# cause_code -> (cause name, ClinicalTrials.gov condition terms)
DISEASE_PATTERNS: Dict[str, Tuple[str, List[str]]] = {
    # Infectious and parasitic
    "HIV":      ("HIV/AIDS", ["hiv", "acquired immunodeficiency", "aids related", "aids defining"]),
    "TB":       ("Tuberculosis", ["tuberculosis", "tb", "mycobacterium tuberculosis"]),
    "MALARIA":  ("Malaria", ["malaria", "plasmodium"]),
    "LRI":      ("Lower respiratory infections", ["pneumonia", "lower respiratory tract infection", "lower respiratory infection",
                 "bronchiolitis", "acute lower respiratory", "lri"]),
    "DIARR":    ("Diarrhoeal diseases", ["diarrhoea", "diarrhea", "diarrhoeal", "diarrheal", "gastroenteritis", "rotavirus",
                 "cholera", "shigella", "shigellosis"]),
    "HEPATITIS": ("Viral hepatitis", ["hepatitis b", "hepatitis c", "viral hepatitis", "hepatitis a", "hepatitis e",
                  "hepatitis d", "hepatitis delta", "hbv", "hcv", "hev"]),
    "MENING":   ("Meningitis", ["meningitis", "meningococcal", "meningococcus", "neisseria meningitidis"]),
    "MEASLES":  ("Measles", ["measles", "rubeola", "morbillivirus"]),
    "NTD":      ("Neglected tropical diseases", ["dengue", "leishmaniasis", "schistosomiasis", "chagas", "trypanosomiasis",
                 "lymphatic filariasis", "onchocerciasis", "trachoma", "rabies", "leprosy", "hookworm", "soil transmitted helminth",
                 "cysticercosis", "echinococcosis", "ascariasis", "trichuriasis", "fascioliasis", "clonorchiasis",
                 "opisthorchiasis", "sleeping sickness", "bilharzia", "river blindness", "hydatid", "filariasis",
                 "hansen disease", "helminth"]),
    # Maternal and neonatal
    "MATERNAL": ("Maternal conditions", ["preeclampsia", "pre eclampsia", "eclampsia", "postpartum hemorrhage",
                 "postpartum haemorrhage", "gestational hypertension", "miscarriage", "ectopic pregnancy",
                 "hyperemesis gravidarum", "maternal sepsis", "puerperal", "post partum hemorrhage",
                 "pregnancy induced hypertension", "obstructed labor", "obstructed labour", "maternal mortality",
                 "maternal death"]),
    "NEON":     ("Neonatal conditions", ["neonatal", "newborn", "preterm infant", "preterm birth", "prematurity",
                 "birth asphyxia", "neonatal sepsis", "bronchopulmonary dysplasia", "neonatal encephalopathy",
                 "hypoxic ischemic encephalopathy"]),
    # Cancers
    "LUNG_CA":  ("Trachea, bronchus and lung cancers", ["lung cancer", "non small cell lung", "small cell lung", "nsclc",
                 "lung carcinoma", "lung neoplasm", "lung adenocarcinoma", "sclc", "bronchogenic carcinoma",
                 "lung tumor", "lung tumour"]),
    "BREAST_CA": ("Breast cancer", ["breast cancer", "breast carcinoma", "breast neoplasm", "breast tumor",
                  "breast tumour", "invasive lobular carcinoma"]),
    "CERVIX_CA": ("Cervical cancer", ["cervical cancer", "cervix cancer", "cervical carcinoma", "cervical neoplasm",
                  "uterine cervical neoplasm", "cervix carcinoma"]),
    "COLORECTAL_CA": ("Colorectal cancer", ["colorectal cancer", "colon cancer", "rectal cancer", "colorectal carcinoma",
                      "colorectal neoplasm", "colon carcinoma", "rectal carcinoma", "colonic neoplasm",
                      "rectal neoplasm", "crc"]),
    "LIVER_CA": ("Liver cancer", ["hepatocellular carcinoma", "liver cancer", "hcc", "hepatocellular cancer",
                 "liver neoplasm", "hepatoma", "liver carcinoma"]),
    "STOMACH_CA": ("Stomach cancer", ["gastric cancer", "stomach cancer", "gastric carcinoma", "gastric adenocarcinoma",
                   "stomach neoplasm", "gastric neoplasm"]),
    "PROSTATE_CA": ("Prostate cancer", ["prostate cancer", "prostatic cancer", "prostate carcinoma", "prostatic neoplasm",
                    "prostate adenocarcinoma", "prostatic adenocarcinoma", "crpc", "mcrpc"]),
    # Cardiovascular and metabolic
    "IHD":      ("Ischaemic heart disease", ["coronary artery disease", "myocardial infarction", "acute coronary syndrome",
                 "ischemic heart", "ischaemic heart", "coronary heart disease", "coronary disease", "angina pectoris",
                 "myocardial ischemia", "myocardial ischaemia", "ihd"]),
    "STROKE":   ("Stroke", ["stroke", "cerebrovascular", "cerebral infarction", "intracerebral hemorrhage",
                 "intracerebral haemorrhage", "subarachnoid hemorrhage", "subarachnoid haemorrhage", "cbvd"]),
    # Keep the trial construct aligned to WHO GHE 1120 (hypertensive heart
    # disease). Generic hypertension/high blood pressure is not equivalent and
    # previously made this numerator much broader than its burden denominator.
    "HHD":      ("Hypertensive heart disease", ["hypertensive heart disease", "hypertensive heart failure",
                  "hypertensive cardiomyopathy", "hypertensive cardiac disease"]),
    "DIAB":     ("Diabetes mellitus", ["diabetes", "diabetic", "t2dm", "t1dm", "t2d", "t1d", "dm"]),
    "KIDNEY":   ("Kidney diseases", ["chronic kidney", "kidney disease", "kidney failure", "renal failure",
                 "renal insufficiency", "end stage renal", "end stage kidney", "diabetic nephropathy",
                 "glomerulonephritis", "esrd", "eskd", "ckd", "dialysis"]),
    # Respiratory
    "COPD":     ("Chronic obstructive pulmonary disease", ["copd", "chronic obstructive pulmonary", "chronic obstructive lung",
                 "emphysema", "chronic bronchitis"]),
    "ASTHMA":   ("Asthma", ["asthma", "asthmatic"]),
    # Mental and neurological
    "DEPRESSION": ("Depressive disorders", ["depression", "depressive", "major depressive", "dysthymia", "dysthymic"]),
    "ANXIETY":  ("Anxiety disorders", ["anxiety disorder", "generalized anxiety", "generalised anxiety", "panic disorder",
                 "social anxiety", "agoraphobia", "phobia", "phobic disorder", "separation anxiety"]),
    "ALZH":     ("Alzheimer's disease and dementias", ["alzheimer", "dementia"]),
    # Digestive
    "CIRRHOSIS": ("Cirrhosis of the liver", ["cirrhosis", "liver fibrosis", "hepatic fibrosis", "chronic liver disease"]),
    # Injuries
    "RTI":      ("Road traffic injuries", ["road traffic", "traffic accident", "traffic injury", "motor vehicle",
                 "road injury", "transport injury", "car crash", "vehicle collision"]),
    "SELF_HARM": ("Self-harm", ["suicide", "suicidal", "self harm", "self injury", "self injurious", "parasuicide"]),
    # Inherited and congenital
    "SICKLE":   ("Sickle cell disease", ["sickle cell", "drepanocytosis", "hbss", "hemoglobin ss", "haemoglobin ss"]),
    "CONGENITAL": ("Congenital anomalies", ["congenital heart", "congenital anomal", "congenital malformation",
                   "birth defect", "neural tube defect", "spina bifida", "cleft lip", "cleft palate", "down syndrome",
                   "trisomy 21", "tetralogy of fallot", "hypoplastic left heart", "gastroschisis",
                   "congenital diaphragmatic hernia"]),
}

# Phrases removed from condition text before matching, because they contain a
# disease term but describe something else.
EXCLUDED_PHRASES: Dict[str, List[str]] = {
    "DIAB": ["diabetes insipidus"],
    "DEPRESSION": ["respiratory depression", "bone marrow depression", "marrow depression", "myocardial depression",
                   "cortical spreading depression", "immune depression", "st segment depression"],
    "HHD": ["pulmonary hypertension", "pulmonary arterial hypertension", "portal hypertension", "intracranial hypertension",
            "ocular hypertension", "gestational hypertension", "pregnancy induced hypertension", "intra abdominal hypertension"],
    "DIARR": ["chemotherapy induced diarrhea", "chemotherapy induced diarrhoea", "irritable bowel syndrome with diarrhea"],
    "STROKE": ["heat stroke", "heatstroke"],
    "NEON": ["neonatal diabetes"],
}

# A study mentioning any of these is not counted for the disease. WHO counts
# COVID-19 as its own cause, so COVID pneumonia is not a lower respiratory infection.
EXCLUDE_STUDY_IF: Dict[str, List[str]] = {
    "LRI": ["covid", "sars cov 2", "coronavirus disease 2019", "2019 ncov"],
}

# Menu groups, in display order.
DISEASE_GROUPS: List[Tuple[str, List[str]]] = [
    ("Infectious and parasitic", ["HIV", "TB", "MALARIA", "LRI", "DIARR", "HEPATITIS", "MENING", "MEASLES", "NTD"]),
    ("Maternal and neonatal", ["MATERNAL", "NEON"]),
    ("Cancers", ["LUNG_CA", "BREAST_CA", "CERVIX_CA", "COLORECTAL_CA", "LIVER_CA", "STOMACH_CA", "PROSTATE_CA"]),
    ("Cardiovascular and metabolic", ["IHD", "STROKE", "HHD", "DIAB", "KIDNEY"]),
    ("Respiratory", ["COPD", "ASTHMA"]),
    ("Mental and neurological", ["DEPRESSION", "ANXIETY", "ALZH"]),
    ("Digestive", ["CIRRHOSIS"]),
    ("Injuries", ["RTI", "SELF_HARM"]),
    ("Inherited and congenital", ["SICKLE", "CONGENITAL"]),
]

# cause_code -> (GHE cause codes summed, label for the WHO measure)
GHE_CAUSES: Dict[str, Tuple[Tuple[int, ...], str]] = {
    "HIV":           ((100,), "HIV/AIDS"),
    "TB":            ((30,), "Tuberculosis"),
    "MALARIA":       ((220,), "Malaria"),
    "LRI":           ((390,), "Lower respiratory infections"),
    "DIARR":         ((110,), "Diarrhoeal diseases"),
    # Acute hepatitis A, B, C and E plus liver cancer and cirrhosis caused by
    # hepatitis B and C: the chronic burden WHO attributes to viral hepatitis.
    "HEPATITIS":     ((186, 190, 200, 205, 661, 662, 1231, 1232),
                      "Acute hepatitis A, B, C, E; liver cancer and cirrhosis due to hepatitis B and C"),
    "MENING":        ((170,), "Meningitis"),
    "MEASLES":       ((150,), "Measles"),
    # WHO neglected tropical diseases available in GHE (snakebite is not).
    "NTD":           ((230, 240, 250, 260, 270, 280, 285, 295, 300, 310, 320, 330, 365),
                      "Trypanosomiasis, Chagas, schistosomiasis, leishmaniasis, lymphatic filariasis, onchocerciasis, "
                      "cysticercosis, echinococcosis, dengue, trachoma, rabies, intestinal nematodes and trematodes, leprosy"),
    "MATERNAL":      ((420,), "Maternal conditions"),
    "NEON":          ((490,), "Neonatal conditions"),
    "LUNG_CA":       ((680,), "Trachea, bronchus, lung cancers"),
    "BREAST_CA":     ((700,), "Breast cancer"),
    "CERVIX_CA":     ((710,), "Cervix uteri cancer"),
    "COLORECTAL_CA": ((650,), "Colon and rectum cancers"),
    "LIVER_CA":      ((660,), "Liver cancer"),
    "STOMACH_CA":    ((640,), "Stomach cancer"),
    "PROSTATE_CA":   ((740,), "Prostate cancer"),
    "IHD":           ((1130,), "Ischaemic heart disease"),
    "STROKE":        ((1140,), "Stroke"),
    "HHD":           ((1120,), "Hypertensive heart disease"),
    "DIAB":          ((800,), "Diabetes mellitus"),
    "KIDNEY":        ((1270,), "Kidney diseases"),
    "COPD":          ((1180,), "Chronic obstructive pulmonary disease"),
    "ASTHMA":        ((1190,), "Asthma"),
    "DEPRESSION":    ((830,), "Depressive disorders"),
    "ANXIETY":       ((880,), "Anxiety disorders"),
    "ALZH":          ((950,), "Alzheimer disease and other dementias"),
    "CIRRHOSIS":     ((1230,), "Cirrhosis of the liver"),
    "RTI":           ((1530,), "Road injury"),
    "SELF_HARM":     ((1610,), "Self-harm"),
    "SICKLE":        ((812,), "Sickle cell disorders and trait"),
    "CONGENITAL":    ((1400,), "Congenital anomalies"),
}

# ------------------------ HTTP ------------------------

def session_with_retries() -> requests.Session:
    retries = CFG.get("retries", {})
    s = requests.Session()
    retry = Retry(
        total=int(retries.get("total", 5)),
        backoff_factor=float(retries.get("backoff_factor", 0.5)),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.headers.update({"User-Agent": "Bayezian-EquityDashboard/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = session_with_retries()

# ------------------------ Helpers ------------------------

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def download_workbook(year: int, refresh: bool = False) -> Path:
    """Cache the WHO workbook locally; WHO republishes only with a new release."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"daly_bycountry_{year}_{GHE_RELEASE}.xlsx"
    if path.exists() and not refresh:
        return path
    response = SESSION.get(GHE_URL_TEMPLATE.format(year=year), timeout=300)
    response.raise_for_status()
    temporary = path.with_suffix(".xlsx.tmp")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    return path


def parse_workbook(path: Path, year: int) -> pd.DataFrame:
    """Read the 'All ages' sheet: both sexes, all ages, values in thousands."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["All ages"]
    sheet.reset_dimensions()  # the file under-reports its used range
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()

    iso_row = next(i for i, row in enumerate(rows) if row[6] == "ISO-3 Code")
    first_col = 7
    iso3 = [str(code).strip() if code else None for code in rows[iso_row][first_col:]]
    quality = rows[iso_row + 1][first_col:]
    population_row = next(row for row in rows if row[0] == "Persons" and str(row[3] or "").startswith("Population"))
    wanted = {code for codes, _ in GHE_CAUSES.values() for code in codes}
    found = {}
    for row in rows:
        if row[0] == "Persons" and isinstance(row[1], (int, float)) and int(row[1]) in wanted:
            found[int(row[1])] = row
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"GHE {year} workbook lacks cause codes {sorted(missing)}")

    records = []
    for cause_code, (codes, label) in GHE_CAUSES.items():
        for index, country in enumerate(iso3):
            values = [found[code][first_col + index] for code in codes]
            population = population_row[first_col + index]
            if country is None or all(value is None for value in values):
                continue
            records.append({
                "country_iso3": country,
                "cause_code": cause_code,
                "cause_name": DISEASE_PATTERNS[cause_code][0],
                "year": year,
                "dalys": sum(float(value) for value in values if value is not None) * 1_000,
                "population": float(population) * 1_000 if population is not None else None,
                "who_data_quality": int(quality[index]) if isinstance(quality[index], (int, float)) else None,
                "ghe_cause_code": "+".join(str(code) for code in codes),
                "ghe_cause_name": label,
            })
    return pd.DataFrame(records)

# ------------------------ Main ------------------------

def run_pipeline(refresh: bool = False) -> pd.DataFrame:
    frames = []
    for year in GHE_YEARS:
        path = download_workbook(year, refresh=refresh)
        frame = parse_workbook(path, year)
        print(f"[GHE {year}] {frame['country_iso3'].nunique()} countries, {len(frame)} rows")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["country_iso3"].isin(VALID_ISO3)].copy()
    out["daly_rate_per_100k"] = out["dalys"] / out["population"] * 100_000
    out["source"] = f"WHO Global Health Estimates {GHE_RELEASE}"
    out = out[[
        "country_iso3", "cause_code", "cause_name", "year", "dalys", "population",
        "daly_rate_per_100k", "who_data_quality", "ghe_cause_code", "ghe_cause_name", "source",
    ]].sort_values(["cause_code", "year", "country_iso3"]).reset_index(drop=True)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PARQUET.with_suffix(".parquet.tmp")
    out.to_parquet(temporary, index=False)
    temporary.replace(OUT_PARQUET)
    print(f"[OK] Wrote {OUT_PARQUET} (rows={len(out)})")
    return out


if __name__ == "__main__":
    try:
        run_pipeline(refresh="--refresh" in sys.argv)
    except Exception as exc:
        print(f"\n[ERROR] global_health_sources.py failed:\n{exc}")
        sys.exit(1)
