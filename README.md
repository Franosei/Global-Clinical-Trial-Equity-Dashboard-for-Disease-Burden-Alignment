# Global-Clinical-Trial-Equity-Dashboard-for-Disease-Burden-Alignment
To develop an automated, real-time analytical dashboard that quantifies and visualises the global mismatch between disease burden and clinical trial activity, and provides actionable insights for trial planners, sponsors, and policy makers.

```
clinical-trial-equity/
├─ README.md
├─ .gitignore
├─ .env.example
├─ config/
│  └─ sources.yaml           # endpoints, years, disease map
├─ data/
│  ├─ raw/                   # API dumps (json/csv)
│  ├─ clean/                 # harmonised country/year/disease
│  └─ gold/                  # final parquet for Shiny
├─ py/
│  ├─ __init__.py
│  ├─ global_health_sources.py   # your existing burden code (drop in here)
│  ├─ ctgov.py                   # trial site pull (v2 API, minimal fields)
│  ├─ harmonise.py               # ISO mapping + disease names
│  ├─ indices.py                 # ER, density, gini/theil (bootstrap optional)
│  └─ build.py                   # orchestrates: pull → clean → gold
├─ r/
│  ├─ renv.lock                 # optional; commit once stable
│  └─ shiny_app.R               # single-file Shiny app reading data/gold/*.parquet
└─ Makefile
```