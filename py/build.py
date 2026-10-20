"""Build the complete dashboard data pipeline.

Run with ``python -m py.build`` from the repository root.
"""

from __future__ import annotations

import subprocess
import sys


STEPS = [
    ("global_health_sources", "Fetch and harmonise WHO disease burden"),
    ("ctgov", "Fetch ClinicalTrials.gov trial sites and facilities"),
    ("resources", "Fetch World Bank price levels and infrastructure"),
    ("indices", "Compute dashboard-ready equity indices"),
]


def run_step(module: str, description: str) -> None:
    print(f"\n=== [{module}] {description} ===")
    subprocess.run([sys.executable, "-m", f"py.{module}"], check=True)


def main() -> None:
    print(">>> Global Clinical Trial Equity Dashboard: build pipeline <<<")
    try:
        for module, description in STEPS:
            run_step(module, description)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Pipeline step failed (exit {exc.returncode})")
        raise SystemExit(exc.returncode) from exc
    print("\n[OK] Build finished successfully.")


if __name__ == "__main__":
    main()
