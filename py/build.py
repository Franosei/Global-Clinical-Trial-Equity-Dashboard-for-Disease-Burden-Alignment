"""Build the complete dashboard data pipeline.

Run with ``python -m py.build`` from the repository root.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .ctgov import get_api_version


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "data" / "gold" / "refresh_status.json"
CT_PATH = ROOT / "data" / "clean" / "ctgov_sites.parquet"
GOLD_PATH = ROOT / "data" / "gold" / "country_interval_disease.parquet"


STEPS = [
    ("global_health_sources", "Fetch and harmonise WHO disease burden"),
    ("ctgov", "Fetch ClinicalTrials.gov trial sites and facilities"),
    ("resources", "Fetch World Bank price levels and infrastructure"),
    ("indices", "Compute dashboard-ready equity indices"),
    ("validation", "Validate scientific and planning data contracts"),
]


def run_step(module: str, description: str) -> None:
    print(f"\n=== [{module}] {description} ===")
    subprocess.run([sys.executable, "-m", f"py.{module}"], check=True)


def publish_status() -> dict:
    """Publish provenance for a successfully validated full rebuild."""
    try:
        upstream = get_api_version()
    except Exception:
        upstream = {}
    payload = {
        "state": "ready",
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cause_code": "ALL",
        "cause_name": "All configured diseases",
        "ctgov_data_timestamp": upstream.get("dataTimestamp"),
        "ctgov_api_version": upstream.get("apiVersion"),
        "trial_rows": int(len(pd.read_parquet(CT_PATH, columns=["cause_code"]))),
        "gold_rows": int(len(pd.read_parquet(GOLD_PATH, columns=["cause_code"]))),
        "message": "Full validated pipeline snapshot published",
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(STATUS_PATH)
    return payload


def main() -> None:
    print(">>> Global Clinical Trial Equity Dashboard: build pipeline <<<")
    try:
        for module, description in STEPS:
            run_step(module, description)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Pipeline step failed (exit {exc.returncode})")
        raise SystemExit(exc.returncode) from exc
    publish_status()
    print("\n[OK] Build finished successfully.")


if __name__ == "__main__":
    main()
