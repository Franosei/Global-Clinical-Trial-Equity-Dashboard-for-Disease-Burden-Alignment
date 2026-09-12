"""Near-real-time snapshot refresh for the dashboard.

ClinicalTrials.gov publishes a refreshed dataset on weekdays. This module
refreshes one disease on demand, atomically rebuilds the gold layer, and writes
machine-readable provenance for the dashboard.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterator

from .ctgov import DISEASE_PATTERNS, get_api_version, pull_ctgov_sites
from .indices import run_indices


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = REPO_ROOT / "data" / "gold"
STATUS_PATH = GOLD_DIR / "refresh_status.json"
LOCK_PATH = GOLD_DIR / ".refresh.lock"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_status() -> dict:
    if not STATUS_PATH.exists():
        return {"state": "snapshot", "message": "Bundled repository snapshot"}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unknown", "message": "Refresh metadata is unreadable"}


def _write_status(payload: dict) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(STATUS_PATH)


@contextmanager
def refresh_lock() -> Iterator[None]:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("A data refresh is already running") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def refresh(cause_code: str) -> dict:
    if cause_code not in DISEASE_PATTERNS:
        raise ValueError(f"Unknown disease code: {cause_code}")
    previous = read_status()
    started = _utc_now()
    with refresh_lock():
        _write_status({
            **previous,
            "state": "running",
            "started_at": started,
            "cause_code": cause_code,
            "message": f"Refreshing {DISEASE_PATTERNS[cause_code][0]}",
        })
        try:
            upstream = get_api_version()
            refreshed = pull_ctgov_sites([cause_code])
            gold = run_indices()
            payload = {
                "state": "ready",
                "started_at": started,
                "completed_at": _utc_now(),
                "cause_code": cause_code,
                "cause_name": DISEASE_PATTERNS[cause_code][0],
                "ctgov_data_timestamp": upstream.get("dataTimestamp"),
                "ctgov_api_version": upstream.get("apiVersion"),
                "refreshed_cause_rows": int(len(refreshed)),
                "gold_rows": int(len(gold)),
                "message": "Live ClinicalTrials.gov snapshot published",
            }
            _write_status(payload)
            return payload
        except Exception as exc:
            _write_status({
                **previous,
                "state": "failed",
                "started_at": started,
                "failed_at": _utc_now(),
                "cause_code": cause_code,
                "message": str(exc),
            })
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cause", choices=sorted(DISEASE_PATTERNS), help="Disease to refresh")
    parser.add_argument("--status", action="store_true", help="Print current refresh metadata")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_status(), indent=2))
        return
    if not args.cause:
        parser.error("--cause is required unless --status is used")
    print(json.dumps(refresh(args.cause), indent=2))


if __name__ == "__main__":
    main()
