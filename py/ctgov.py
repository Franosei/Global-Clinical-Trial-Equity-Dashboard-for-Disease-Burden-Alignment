"""
ClinicalTrials.gov v2 → country-year-disease site counts (single parquet output)

- Per-disease queries using DISEASE_PATTERNS (from py/global_health_sources.py)
- Uses v2 endpoint: https://clinicaltrials.gov/api/v2/studies
- Search via query.cond, but we query PER TOKEN (no OR chains) and dedupe afterwards
- Country→ISO3 via pycountry (strict ISO-3166 countries only)
- Trials that match multiple diseases are split across those diseases
- De-dup per (nct_id, country_iso3, year, cause_code)

Output (single file):
  data/clean/ctgov_sites.parquet

Schema:
  country_iso3, year, cause_code, cause_name, phase, trial_sites

Run:
  python -m py.ctgov     OR     python py/ctgov.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set
import re
import unicodedata
import yaml
import sys

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import pycountry

# IMPORTANT: use absolute import as requested
from global_health_sources import DISEASE_PATTERNS, norm_text

# Paths & Config ------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "config" / "sources.yaml"
DATA_DIR = REPO_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
OUT_PARQUET = CLEAN_DIR / "ctgov_sites.parquet"

def load_cfg() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_cfg()
YEARS = list(range(int(CFG["years"]["start"]), int(CFG["years"]["end"]) + 1))

# v2 base endpoint
BASE_CT_V2 = "https://clinicaltrials.gov/api/v2/"

RETRY_TOTAL = int(CFG.get("retries", {}).get("total", 5))
RETRY_BACKOFF = float(CFG.get("retries", {}).get("backoff_factor", 0.5))
DEFAULT_PAGE_SIZE = int(CFG.get("ctgov", {}).get("page_size", 200))

# Year picking rule (explicit & auditable)
DATE_RULE = str(CFG.get("ctgov", {}).get("date_rule", "start")).lower()  # start | first_posted | primary_completion
if DATE_RULE not in {"start", "first_posted", "primary_completion"}:
    DATE_RULE = "start"

# HTTP ------------------------

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

# v2 paging ------------------------

def ctgov_v2_iter_token(token_query: str, page_size: int = DEFAULT_PAGE_SIZE) -> Iterable[dict]:
    """
    Yield pages of studies for a single token query (query.cond = token).
    We do **not** pass 'fields' to avoid 400s from strict gateways.
    We also back off pageSize on 400 automatically.
    """
    token = ""
    size = max(50, int(page_size))
    while True:
        params = {
            "query.cond": token_query,  # simple single-token condition search
            "pageSize": str(size),
            "format": "json",
        }
        if token:
            params["pageToken"] = token
        try:
            page = http_get_json(BASE_CT_V2 + "studies", params)
        except requests.HTTPError as e:
            status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
            if status == 400 and size > 100:
                print(f"[WARN] 400 for '{token_query}'; retry with smaller pageSize (was {size} → 100)")
                size = 100
                continue
            msg = e.response.text if getattr(e, "response", None) else str(e)
            raise RuntimeError(f"CT.gov v2 error (status={status}) for params={params} :: {msg}") from e

        yield page
        token = page.get("nextPageToken") or ""
        if not token:
            break

# Helpers ------------------------

def parse_year_maybe(date_str: Optional[str]) -> Optional[int]:
    """
    Accepts 'YYYY', 'YYYY-MM', or full date strings and extracts the year.
    """
    if not date_str:
        return None
    m = re.search(r"(\d{4})", str(date_str))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def pick_year(stat: dict, rule: str = DATE_RULE) -> Optional[int]:
    choices = {
        "start": (stat.get("startDateStruct") or {}).get("date"),
        "first_posted": (stat.get("studyFirstPostDateStruct") or {}).get("date"),
        "primary_completion": (stat.get("primaryCompletionDateStruct") or {}).get("date"),
    }
    order = {
        "start": ["start", "primary_completion", "first_posted"],
        "first_posted": ["first_posted", "start", "primary_completion"],
        "primary_completion": ["primary_completion", "start", "first_posted"],
    }[rule]
    for key in order:
        y = parse_year_maybe(choices.get(key))
        if y is not None:
            return y
    return None

def standardize_phase_v2(phases: List[str] | None) -> str:
    """
    v2 phases look like ["PHASE1","PHASE2"] etc.
    Return canonical labels: I, II, III, IV, I/II, II/III, III/IV, NA (EARLY_PHASE1 → I).
    """
    phset = set(phases or [])
    if {"PHASE1", "PHASE2"} <= phset:
        return "I/II"
    if {"PHASE2", "PHASE3"} <= phset:
        return "II/III"
    if {"PHASE3", "PHASE4"} <= phset:
        return "III/IV"
    if "PHASE4" in phset:
        return "IV"
    if "PHASE3" in phset:
        return "III"
    if "PHASE2" in phset:
        return "II"
    if "PHASE1" in phset or "EARLY_PHASE1" in phset:
        return "I"
    return "NA"

def extract_condition_texts(study: dict) -> List[str]:
    """
    Collect condition texts from:
      - protocolSection.conditionsModule.conditions
      - protocolSection.conditionsModule.conditionsBrowseLeafList[].leaf
      - protocolSection.conditionsModule.conditionsMeshList[].term
      - derivedSection.conditionBrowseModule.meshes[].term (fallback)
    """
    texts: List[str] = []
    ps = study.get("protocolSection", {}) or {}
    condm = ps.get("conditionsModule", {}) or {}
    # base list
    texts.extend(condm.get("conditions") or [])
    # leaves
    leaves = condm.get("conditionsBrowseLeafList") or []
    texts.extend([x.get("leaf") for x in leaves if isinstance(x, dict) and x.get("leaf")])
    # mesh list
    meshes = condm.get("conditionsMeshList") or []
    texts.extend([x.get("term") for x in meshes if isinstance(x, dict) and x.get("term")])
    # derived fallback
    der = study.get("derivedSection", {}) or {}
    cb = der.get("conditionBrowseModule", {}) or {}
    d_mesh = cb.get("meshes") or []
    texts.extend([x.get("term") for x in d_mesh if isinstance(x, dict) and x.get("term")])
    # clean
    return [t for t in texts if t]

def match_condition_to_cause(conditions: List[str], cause_keywords: List[str]) -> bool:
    if not conditions:
        return False
    cond_n = [norm_text(c) for c in conditions if c]
    needles = [norm_text(k) for k in cause_keywords]
    return any(n and (n in c) for c in cond_n for n in needles)

def _ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def country_to_iso3(label: Optional[str]) -> Optional[str]:
    """
    Map a v2 location country label to ISO3 via pycountry only; keep strict ISO3 (exclude regions/globals).
    """
    if not label or not str(label).strip():
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
    # accept only strict ISO3 (exclude WHO regions, globals)
    if code and re.fullmatch(r"[A-Z]{3}", code):
        return code
    return None

def unique_tokens_for_cause(keywords: List[str], cause_name: str, max_tokens: int = 8) -> List[str]:
    """
    Build a small set of unique, useful tokens to query individually with query.cond.
    We include the cause_name and the raw keywords, deduped case-insensitively.
    """
    toks = [t.strip() for t in (keywords + [cause_name]) if t and t.strip()]
    # lightweight dedupe preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in toks:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
        if len(uniq) >= max_tokens:
            break
    return uniq

# Main Extraction ------------------------

def pull_ctgov_sites() -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    site_rows: List[dict] = []

    for cause_code, (cause_name, keywords) in DISEASE_PATTERNS.items():
        tokens = unique_tokens_for_cause(keywords, cause_name, max_tokens=10)
        print(f"[CT.gov v2] {cause_code} :: tokens -> {', '.join(tokens)}")

        # query one token at a time to avoid 400s from long OR strings
        for tkn in tokens:
            for page in ctgov_v2_iter_token(tkn, page_size=DEFAULT_PAGE_SIZE):
                for s in page.get("studies", []) or []:
                    ps = s.get("protocolSection", {}) or {}
                    ident = ps.get("identificationModule", {}) or {}
                    locm = ps.get("contactsLocationsModule", {}) or {}
                    desm = ps.get("designModule", {}) or {}
                    stat = ps.get("statusModule", {}) or {}

                    nct = ident.get("nctId")
                    if not nct:
                        continue

                    # Match disease using conditions + browse leaves + MeSH (+ derived fallback)
                    conditions = extract_condition_texts(s)
                    if not match_condition_to_cause(conditions, keywords + [cause_name]):
                        continue

                    # Pick a year per config rule
                    year = pick_year(stat, DATE_RULE)
                    if year is None or year not in YEARS:
                        continue

                    phase_std = standardize_phase_v2(desm.get("phases"))

                    locations = (locm.get("locations") or [])
                    for loc in locations:
                        cname = (loc or {}).get("country")
                        iso3 = country_to_iso3(cname)
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
        # still write an empty parquet with the right schema
        empty = pd.DataFrame(columns=["country_iso3","year","cause_code","cause_name","phase","trial_sites"])
        empty.to_parquet(OUT_PARQUET, index=False)
        return empty

    # 1) De-duplicate unique trial per country/year/cause
    events = (pd.DataFrame(site_rows)
                .drop_duplicates(subset=["nct_id", "country_iso3", "year", "cause_code"])
                .reset_index(drop=True))

    # 2) Aggregate to counts (one row per country-year-disease-phase)
    agg = (events.groupby(["country_iso3", "year", "cause_code", "cause_name", "phase"], as_index=False)
                 .agg(trial_sites=("nct_id", "nunique")))

    # 3) Write single parquet output with required columns & types
    cols = ["country_iso3","year","cause_code","cause_name","phase","trial_sites"]
    agg = agg[cols]
    agg["country_iso3"] = agg["country_iso3"].astype(str)
    agg["year"] = agg["year"].astype(int)
    agg["cause_code"] = agg["cause_code"].astype(str)
    agg["cause_name"] = agg["cause_name"].astype(str)
    agg["phase"] = agg["phase"].astype(str)
    agg["trial_sites"] = agg["trial_sites"].astype(int)

    agg.to_parquet(OUT_PARQUET, index=False)

    print(f"[OK] Wrote {OUT_PARQUET} (rows={len(agg)})")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(agg.head(12).to_string(index=False))
    return agg

# Entry ------------------------

if __name__ == "__main__":
    try:
        pull_ctgov_sites()
    except Exception as e:
        print("\n[ERROR] ctgov.py failed:")
        print(str(e))
        sys.exit(1)
