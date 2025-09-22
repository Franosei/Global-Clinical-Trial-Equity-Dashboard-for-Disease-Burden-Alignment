# py/ctgov.py
"""
ClinicalTrials.gov v2 → country-year-disease site counts (no sponsor classification)

- Queries per disease (from DISEASE_PATTERNS in py/global_health_sources.py)
- Uses v2 endpoint: https://clinicaltrials.gov/api/v2/studies
- Paging via nextPageToken
- Country→ISO3 via pycountry ONLY (no manual aliases)
- Keeps only ISO3 codes from your provided ALLOWED_ISO3 universe
- De-dupes per (nct_id, country_iso3, year, cause_code)

Outputs:
  data/clean/ctgov_sites.parquet
  data/clean/country_iso3_lookup.parquet
  data/clean/iso3_universe.parquet

Columns:
  country_iso3, year, cause_code, cause_name, phase, trial_sites
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Iterable
import json
import re
import unicodedata
import yaml

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import pycountry

# Reuse disease dictionary + normaliser from WHO module
from global_health_sources import DISEASE_PATTERNS, norm_text

# ------------------------ Paths & Config ------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "config" / "sources.yaml"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
OUT_PARQUET = CLEAN_DIR / "ctgov_sites.parquet"
OUT_COUNTRY_AUDIT = CLEAN_DIR / "country_iso3_lookup.parquet"
OUT_UNIVERSE = CLEAN_DIR / "iso3_universe.parquet"

def load_cfg() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_cfg()
YEARS = list(range(CFG["years"]["start"], CFG["years"]["end"] + 1))

# v2 base endpoint
BASE_CT_V2 = "https://clinicaltrials.gov/api/v2/"

RETRY_TOTAL = int(CFG.get("retries", {}).get("total", 5))
RETRY_BACKOFF = float(CFG.get("retries", {}).get("backoff_factor", 0.5))

# ------------------------ Allowed ISO3 Universe (your list) ------------------------

ALLOWED_ISO3 = {
    'AFG','AFR','AGO','ALB','AMR','AND','ARE','ARG','ARM','ATG','AUS','AUT','AZE','BDI','BEL','BEN','BFA',
    'BGD','BGR','BHR','BHS','BIH','BLR','BLZ','BOL','BRA','BRB','BRN','BTN','BWA','CAF','CAN','CHE','CHL',
    'CHN','CIV','CMR','COD','COG','COK','COL','COM','CPV','CRI','CUB','CYP','CZE','DEU','DJI','DMA','DNK',
    'DOM','DZA','ECU','EGY','EMR','ERI','ESP','EST','ETH','EUR','FIN','FJI','FRA','FSM','GAB','GBR','GEO',
    'GHA','GIN','GLO','GMB','GNB','GNQ','GRC','GRD','GTM','GUY','HND','HRV','HTI','HUN','IDN','IND','IRL',
    'IRN','IRQ','ISL','ISR','ITA','JAM','JOR','JPN','KAZ','KEN','KGZ','KHM','KIR','KNA','KOR','KWT','LAO',
    'LBN','LBR','LBY','LCA','LKA','LSO','LTU','LUX','LVA','MAR','MCO','MDA','MDG','MDV','ME1','MEX','MHL',
    'MKD','MLI','MLT','MMR','MNE','MNG','MOZ','MRT','MUS','MWI','MYS','NAM','NER','NGA','NIC','NIU','NLD',
    'NOR','NPL','NRU','NZL','OMN','PAK','PAN','PER','PHL','PLW','PNG','POL','PRI','PRK','PRT','PRY','QAT',
    'ROU','RUS','RWA','SAU','SDN','SEA','SEN','SGP','SLB','SLE','SLV','SMR','SOM','SRB','SSD','STP','SUR',
    'SVK','SVN','SWE','SWZ','SYC','SYR','TCD','TGO','THA','TJK','TKM','TLS','TON','TTO','TUN','TUR','TUV',
    'TZA','UGA','UKR','URY','USA','UZB','VCT','VEN','VNM','VUT','WHO','WPR','WSM','YEM','ZAF','ZMB','ZWE'
}

def _build_universe_df() -> pd.DataFrame:
    """Mark which of the ALLOWED_ISO3 are true ISO-3166 countries (others are regions/tags)."""
    rows = []
    for code in sorted(ALLOWED_ISO3):
        name = None
        is_country = True
        try:
            c = next((x for x in pycountry.countries if getattr(x, "alpha_3", None) == code), None)
            if c:
                name = c.name
            else:
                is_country = False
        except Exception:
            is_country = False
        rows.append({"iso3": code, "is_country": is_country, "name": name})
    return pd.DataFrame(rows)

# ------------------------ HTTP ------------------------

def session_with_retries(total: int = RETRY_TOTAL, backoff: float = RETRY_BACKOFF) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total, connect=total, read=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.headers.update({"User-Agent": "Bayezian-EquityDashboard/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

SESSION = session_with_retries()

def http_get_json(url: str, params: dict) -> dict:
    r = SESSION.get(url, params=params, timeout=180)
    r.raise_for_status()
    return r.json()

# ------------------------ v2 paging ------------------------

# We request only the fields we need to minimise payload.
V2_FIELDS = ",".join([
    "protocolSection.identificationModule.nctId",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.contactsLocationsModule.locations.country",
    "protocolSection.designModule.phases",
    "protocolSection.statusModule.startDateStruct.date",
    "protocolSection.statusModule.studyFirstPostDateStruct.date",
])

def ctgov_v2_iter_all(expr: str, page_size: int = 1000) -> Iterable[dict]:
    """
    Yields pages (dicts) of v2 studies results for the boolean expression `expr`.
    """
    token = ""
    while True:
        params = {
            "query.term": expr,
            "pageSize": str(page_size),
            "format": "json",
            "fields": V2_FIELDS,
        }
        if token:
            params["pageToken"] = token
        page = http_get_json(BASE_CT_V2 + "studies", params)
        yield page
        token = page.get("nextPageToken") or ""
        if not token:
            break

# ------------------------ Helpers ------------------------

def parse_year_maybe(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    m = re.search(r"(\d{4})", str(date_str))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def standardize_phase_v2(phases: List[str]) -> str:
    """
    v2 phases come like ["PHASE1","PHASE2"] etc. Return canonical labels:
    I, II, III, IV, I/II, II/III, III/IV, NA, with EARLY_PHASE1 → I.
    """
    ph = set((phases or []))
    # combined (check in order)
    if {"PHASE1", "PHASE2"} <= ph:
        return "I/II"
    if {"PHASE2", "PHASE3"} <= ph:
        return "II/III"
    if {"PHASE3", "PHASE4"} <= ph:
        return "III/IV"
    # singles / fallbacks
    if "PHASE4" in ph:
        return "IV"
    if "PHASE3" in ph:
        return "III"
    if "PHASE2" in ph:
        return "II"
    if "PHASE1" in ph or "EARLY_PHASE1" in ph:
        return "I"
    return "NA"

def disease_expr(keywords: List[str], cause_name: str) -> str:
    toks = [k for k in (keywords + [cause_name]) if k]
    parts = [f'"{t.strip()}"' if " " in t else t.strip() for t in toks]
    return "(" + " OR ".join(sorted(set(parts))) + ")"

def match_condition_to_cause(conditions: List[str], cause_keywords: List[str]) -> bool:
    if not conditions:
        return False
    cond_n = [norm_text(c) for c in conditions if c]
    needles = [norm_text(k) for k in cause_keywords]
    return any(n and (n in c) for c in cond_n for n in needles)

def _ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def country_to_iso3(label: Optional[str], *, audit: Dict[str, Optional[str]] | None = None) -> Optional[str]:
    """
    Map a v2 location country label to ISO3 via pycountry only; keep only ALLOWED_ISO3.
    """
    if not label or not str(label).strip():
        if audit is not None:
            audit[label or ""] = None
        return None
    name = str(label).strip()
    code = None
    # exact lookup
    try:
        c = pycountry.countries.lookup(name)
        code = getattr(c, "alpha_3", None)
    except LookupError:
        pass
    # fuzzy on ASCII variant if needed
    if not code:
        for cand in (name, _ascii(name)):
            try:
                res = pycountry.countries.search_fuzzy(cand)
                if res:
                    code = getattr(res[0], "alpha_3", None)
                    break
            except Exception:
                continue
    if code and code in ALLOWED_ISO3:
        if audit is not None:
            audit[name] = code
        return code
    if audit is not None:
        audit[name] = None
    return None

# ------------------------ Main Extraction ------------------------

def pull_ctgov_sites() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # persist the universe for transparency
    _build_universe_df().to_parquet(OUT_UNIVERSE, index=False)

    site_rows: List[dict] = []
    audit_map: Dict[str, Optional[str]] = {}

    for cause_code, (cause_name, keywords) in DISEASE_PATTERNS.items():
        expr = disease_expr(keywords, cause_name)
        print(f"[CT.gov v2] {cause_code} :: {expr}")

        all_studies: List[dict] = []
        for page in ctgov_v2_iter_all(expr, page_size=1000):
            studies = page.get("studies", []) or []
            all_studies.extend(studies)

        # Save raw provenance (trimmed to the subset of fields we requested)
        with open(RAW_DIR / f"ctgov_v2_{cause_code}.json", "w", encoding="utf-8") as f:
            json.dump(all_studies, f, ensure_ascii=False)

        # Transform to site-level rows
        for s in all_studies:
            ps = s.get("protocolSection", {}) or {}
            ident = ps.get("identificationModule", {}) or {}
            condm = ps.get("conditionsModule", {}) or {}
            locm = ps.get("contactsLocationsModule", {}) or {}
            desm = ps.get("designModule", {}) or {}
            stat = ps.get("statusModule", {}) or {}

            nct = ident.get("nctId")
            if not nct:
                continue

            conditions = condm.get("conditions") or []
            if not match_condition_to_cause(conditions, keywords + [cause_name]):
                continue

            # Pick a year: prefer startDateStruct, fallback to studyFirstPostDateStruct
            y = (stat.get("startDateStruct", {}) or {}).get("date")
            if not y:
                y = (stat.get("studyFirstPostDateStruct", {}) or {}).get("date")
            year = parse_year_maybe(y)
            if year is None or year not in YEARS:
                continue

            phase_std = standardize_phase_v2(desm.get("phases") or [])

            locations = (locm.get("locations") or [])
            for loc in locations:
                cname = (loc or {}).get("country")
                iso3 = country_to_iso3(cname, audit=audit_map)
                if not iso3:
                    continue
                site_rows.append({
                    "nct_id": nct,
                    "country_iso3": iso3,
                    "year": int(year),
                    "cause_code": cause_code,
                    "cause_name": cause_name,
                    "phase": phase_std,
                })

    if not site_rows:
        print("[WARN] No CT.gov v2 rows collected.")
        pd.DataFrame({"ctgov_label": list(audit_map.keys()),
                      "iso3": list(audit_map.values())}).to_parquet(OUT_COUNTRY_AUDIT, index=False)
        return pd.DataFrame()

    # De-duplicate unique trial per country/year/cause
    df = pd.DataFrame(site_rows).drop_duplicates(subset=["nct_id", "country_iso3", "year", "cause_code"])

    # Aggregate to counts
    agg = (df.groupby(["country_iso3", "year", "cause_code", "cause_name", "phase"], as_index=False)
             .agg(trial_sites=("nct_id", "nunique")))

    # Persist outputs
    agg.to_parquet(OUT_PARQUET, index=False)

    audit_df = (pd.DataFrame({"ctgov_label": list(audit_map.keys()),
                              "iso3": list(audit_map.values())})
                  .sort_values(["iso3", "ctgov_label"], na_position="last")
                  .reset_index(drop=True))
    audit_df.to_parquet(OUT_COUNTRY_AUDIT, index=False)

    print(f"[OK] Wrote {OUT_PARQUET} (rows={len(agg)})")
    print(f"[OK] Wrote {OUT_COUNTRY_AUDIT} (labels={audit_df.shape[0]})")
    print(f"[OK] Wrote {OUT_UNIVERSE} (codes={len(ALLOWED_ISO3)})")

    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(agg.head(12).to_string(index=False))
    return agg

# ------------------------ Entry ------------------------

if __name__ == "__main__":
    pull_ctgov_sites()
