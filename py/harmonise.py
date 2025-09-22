# py/harmonise.py
"""
Harmonisation utilities:
- ISO3 → full country names (for mapping/labels)
- Canonical disease names from DISEASE_PATTERNS
- Safe joins/filters against allowed ISO3 universe

Inputs (from data/clean/):
  - who_daly_selected.parquet
  - ctgov_sites.parquet
  - iso3_universe.parquet   (produced by py/ctgov.py)

Outputs (to data/clean/):
  - who_daly_selected_harmonised.parquet
  - ctgov_sites_harmonised.parquet
  - country_reference.parquet
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pycountry

# Reuse canonical disease dictionary
from global_health_sources import DISEASE_PATTERNS

# ------------------------ Paths ------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"

IN_WHO = CLEAN_DIR / "who_daly_selected.parquet"
IN_CT  = CLEAN_DIR / "ctgov_sites.parquet"
IN_UNI = CLEAN_DIR / "iso3_universe.parquet"

OUT_WHO = CLEAN_DIR / "who_daly_selected_harmonised.parquet"
OUT_CT  = CLEAN_DIR / "ctgov_sites_harmonised.parquet"
OUT_REF = CLEAN_DIR / "country_reference.parquet"

# ------------------------ Country reference ------------------------

# Non-country codes in your universe (regions/tags) → readable display names
NON_COUNTRY_NAMES = {
    "AFR": "WHO African Region",
    "AMR": "WHO Region of the Americas",
    "EMR": "WHO Eastern Mediterranean Region",
    "EUR": "WHO European Region",
    "SEA": "WHO South-East Asia Region",
    "WPR": "WHO Western Pacific Region",
    "WHO": "World Health Organization (global)",
    "GLO": "Global (all countries)",
    "ME1": "Middle East (ME1 group)",
}

def iso3_to_record(iso3: str) -> dict:
    """
    Return a dict with ISO3, alpha2, official/common name, and is_country.
    Falls back to NON_COUNTRY_NAMES for region/tag codes.
    """
    rec = {"iso3": iso3, "alpha2": None, "name_official": None, "name_common": None, "is_country": True}
    try:
        c = next((x for x in pycountry.countries if getattr(x, "alpha_3", None) == iso3), None)
        if c:
            rec["alpha2"] = getattr(c, "alpha_2", None)
            # Prefer official_name if present, otherwise name
            official = getattr(c, "official_name", None)
            rec["name_official"] = official or getattr(c, "name", None)
            # some entries have a 'common_name'
            rec["name_common"] = getattr(c, "common_name", None) or rec["name_official"]
            rec["is_country"] = True
            return rec
    except Exception:
        pass

    # Not an ISO country → treat as non-country code (region/tag)
    rec["is_country"] = False
    rec["alpha2"] = None
    display = NON_COUNTRY_NAMES.get(iso3, iso3)  # fallback to the code itself if unknown
    rec["name_official"] = display
    rec["name_common"] = display
    return rec

def build_country_reference(universe_path: Path = IN_UNI) -> pd.DataFrame:
    """
    Build a reference table from iso3_universe.parquet if present; otherwise,
    derive from pycountry and NON_COUNTRY_NAMES.
    """
    if universe_path.exists():
        uni = pd.read_parquet(universe_path)
        if "iso3" in uni.columns:
            codes = sorted(set(uni["iso3"].astype(str)))
        else:
            codes = []
    else:
        # fallback: all ISO countries + known non-country codes
        codes = sorted({*(getattr(c, "alpha_3", None) for c in pycountry.countries), *NON_COUNTRY_NAMES.keys()})
        codes = [c for c in codes if c]

    rows = [iso3_to_record(code) for code in codes]
    ref = pd.DataFrame(rows)
    # Create a display_name used by charts/maps (prefer common/official)
    ref["display_name"] = ref["name_common"].fillna(ref["name_official"])
    return ref.sort_values(["is_country", "iso3"], ascending=[False, True]).reset_index(drop=True)

# ------------------------ Disease reference ------------------------

def build_disease_reference() -> pd.DataFrame:
    rows = [{"cause_code": code, "cause_name": name} for code, (name, _kw) in DISEASE_PATTERNS.items()]
    return pd.DataFrame(rows)

# ------------------------ Harmonisers ------------------------

def harmonise_countries(df: pd.DataFrame, ref: pd.DataFrame, *, on: str = "country_iso3") -> pd.DataFrame:
    df = df.copy()
    df[on] = df[on].astype(str).str.upper().str.strip()
    cols_keep = ["iso3", "alpha2", "display_name", "name_official", "name_common", "is_country"]
    out = df.merge(ref.rename(columns={"iso3": on})[[on] + cols_keep[1:]], on=on, how="left")
    # rename to simple columns for downstream:
    out = out.rename(columns={
        "display_name": "country_name",        # preferred label
        "name_official": "country_name_official",
        "name_common": "country_name_common",
        "alpha2": "country_iso2",
        "is_country": "is_country_code",
    })
    return out

def harmonise_diseases(df: pd.DataFrame, dref: pd.DataFrame) -> pd.DataFrame:
    # Overwrite cause_name from canonical dictionary to ensure consistency
    out = df.drop(columns=["cause_name"], errors="ignore").merge(dref, on="cause_code", how="left")
    return out

# ------------------------ Main ------------------------

def run() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # Build & persist country reference
    country_ref = build_country_reference(IN_UNI)
    country_ref.to_parquet(OUT_REF, index=False)

    # Disease reference (in-memory)
    disease_ref = build_disease_reference()

    # WHO DALY
    if IN_WHO.exists():
        who = pd.read_parquet(IN_WHO)
        who = harmonise_diseases(who, disease_ref)
        who = harmonise_countries(who, country_ref, on="country_iso3")
        # Optional: drop rows that aren’t countries (regions/tags) since maps use polygons
        who_h = who.copy()
        who_h.to_parquet(OUT_WHO, index=False)
        print(f"[OK] wrote {OUT_WHO} (rows={len(who_h)})")
    else:
        print(f"[SKIP] {IN_WHO} not found.")

    # CT.gov sites
    if IN_CT.exists():
        ct = pd.read_parquet(IN_CT)
        ct = harmonise_diseases(ct, disease_ref)
        ct = harmonise_countries(ct, country_ref, on="country_iso3")
        ct_h = ct.copy()
        ct_h.to_parquet(OUT_CT, index=False)
        print(f"[OK] wrote {OUT_CT} (rows={len(ct_h)})")
    else:
        print(f"[SKIP] {IN_CT} not found.")

    # Small preview
    if IN_WHO.exists():
        print("\n[WHO preview]")
        print(pd.read_parquet(OUT_WHO).head(8).to_string(index=False))
    if IN_CT.exists():
        print("\n[CT.gov preview]")
        print(pd.read_parquet(OUT_CT).head(8).to_string(index=False))

if __name__ == "__main__":
    run()
