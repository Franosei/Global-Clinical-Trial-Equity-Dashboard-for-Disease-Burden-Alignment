# py/build.py
"""
Build orchestrator for the Global Clinical Trial Equity Dashboard project.

Steps:
  1. Pull:     Fetch raw datasets (ctgov, WHO burden, etc.).
  2. Clean:    Harmonise country ISO3, disease codes, names.
  3. Gold:     Compute equity indices (ER, density, Gini, Theil).

Outputs:
  - data/raw/...
  - data/clean/...
  - data/gold/...

Run:
  python -m py.build
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_DIR = REPO_ROOT / "py"

STEPS = [
    ("ctgov", "Fetch CT.gov trials → data/raw/ctgov_sites.parquet"),
    ("harmonise", "Harmonise WHO + CT.gov + ISO tables → data/clean/*.parquet"),
    ("indices", "Compute ER, density, Gini/Theil → data/gold/country_year_disease.parquet"),
]

def run_step(module: str, description: str) -> None:
    print(f"\n=== [{module}] {description} ===")
    try:
        subprocess.run([sys.executable, "-m", f"py.{module}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Step {module} failed (exit {e.returncode})")
        sys.exit(e.returncode)

def main() -> None:
    print(">>> Global Clinical Trial Equity Dashboard: build pipeline <<<")
    for module, description in STEPS:
        run_step(module, description)
    print("\n[OK] Build finished successfully.")

if __name__ == "__main__":
    main()
