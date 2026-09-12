# Global Clinical Trial Equity Intelligence

## Overview

Clinical research is most useful when the populations represented in trials reflect the populations affected by disease. In practice, research activity is often concentrated in a relatively small number of countries, while substantial disease burden remains under-represented.

This project examines that imbalance. It brings together registered clinical trial activity from ClinicalTrials.gov and disease burden estimates from the World Health Organization, then presents the results through an interactive analytical dashboard. The aim is to support better questions about geographic representation, not to produce a league table of countries or an automated site-selection decision.

The dashboard is intended for trial planners, epidemiologists, public health researchers, research funders and policy teams who need a clear view of where research activity appears disproportionate to health need.

## Questions the dashboard can help answer

The application is designed around practical research planning questions:

- Which countries carry a material share of the selected disease burden but host little or no registered trial activity?
- How closely does the geographic distribution of trial activity follow the distribution of measured burden?
- Which countries have the largest estimated shortfall relative to proportional representation?
- Has global trial activity for the selected disease been increasing or declining?
- How might an additional portfolio of trial sites change the overall distribution?
- How sensitive are scenario results to uncertainty in burden and site placement?

The dashboard provides an executive overview, a country-level analytical table, a short-term forecast and a scenario planning workspace. All calculations are performed from versioned Parquet snapshots so that results can be reproduced outside the user interface.

## Data sources

### Clinical trial activity

Trial records are retrieved from the [ClinicalTrials.gov API version 2](https://clinicaltrials.gov/data-api/api). The source is normally refreshed on weekdays. The application records both the upstream data timestamp and the time at which the local analytical snapshot was published.

The historical extraction uses a unique NCT study and country combination as its activity unit. A study listed in three countries contributes one activity unit to each country. Multiple facilities belonging to the same study within one country are counted once. For that reason, the measure should be interpreted as registered study-country activity, not as a literal count of individual research facilities.

Conditions are assigned to disease groups using a controlled set of disease names and synonyms. A study can contribute to more than one disease group when its registered conditions genuinely match more than one group. Short acronyms are matched as complete terms to reduce false classifications.

### Disease burden

Burden data are drawn from the [WHO Global Health Observatory](https://www.who.int/data/gho/info/gho-odata-api). The current repository snapshot uses cumulative DALY rates per 100,000 population across the configured years. WHO regional aggregates are excluded from country-level calculations.

Burden estimates are published less frequently than trial registrations. They are therefore treated as a reference distribution rather than a live epidemiological signal. The dashboard does not imply that both sources have the same reporting date or update frequency.

There is an important interpretive limitation. A share calculated from country-level DALY rates describes the distribution of recorded burden intensity across countries. It is not the same as a share of absolute global DALY counts. Population-weighted absolute burden should be used before this system is relied upon for formal resource allocation.

## Core measures

### Equity Ratio

For country `c`, the Equity Ratio is:

```text
Equity Ratio(c) = trial activity share(c) / burden indicator share(c)
```

An Equity Ratio of 1 indicates proportional alignment within the selected analytical cohort. A value below 1 indicates less registered activity than expected from the burden reference. A value above 1 indicates greater registered activity than expected.

The ratio is descriptive. It does not measure trial quality, recruitment performance, regulatory readiness, health-system capacity or the appropriateness of a particular site.

### Alignment score

The alignment score converts the total variation distance between the trial activity distribution and the burden distribution to a scale from 0 to 100:

```text
Alignment score = 100 x (1 - total variation distance)
```

A score of 100 represents proportional alignment. Lower values indicate greater geographic divergence. The score is useful for communication, while the underlying country-level shares should be retained for substantive interpretation.

### Theil divergence

The Theil measure summarises the divergence between trial activity share and burden share. A value of zero indicates proportional alignment. Larger values indicate greater concentration away from the burden reference. Unlike the headline alignment score, the Theil measure is not bounded at 100.

### Sites required for individual parity

The country diagnostic solves the following equation:

```text
(current activity + additional sites) / (global activity + additional sites) = burden share
```

This accounts for the fact that adding activity changes both the country numerator and the global denominator. The results for individual countries must not be added together and treated as a global plan.

## Forecasting approach

The forecast models annual global study-country activity for the selected disease. It fits a log-linear trend to the most recent complete years and uses residual bootstrap sampling to produce an 80 per cent uncertainty interval.

The model is deliberately transparent. It is suitable for directional portfolio discussion, but it does not account for protocol pipelines, regulatory approvals, sponsor strategy, competitive trials, recruitment capacity or unexpected changes in registration practice. It should not be described as a forecast of enrolment or study success.

## Scenario planning

The scenario laboratory allocates a user-defined number of additional sites using one of three rules:

- **Equity gap:** prioritises countries with the largest shortfall against their burden-aligned target.
- **Disease burden:** allocates in proportion to the burden reference.
- **Hybrid:** combines equity gap and disease burden, with greater weight placed on the equity gap.

Allocations are whole numbers. The dashboard recalculates country shares, Equity Ratios, the alignment score and Theil divergence after allocation. Monte Carlo simulation introduces variation in both burden and site placement, producing a range rather than a single deterministic claim.

These scenarios are planning aids. A real site strategy would also need evidence on eligible population, standard of care, competing trials, investigator capability, laboratory and pharmacy infrastructure, regulatory timelines, contracting, cost, travel burden and community acceptability.

## Live data behaviour

The application uses near-real-time refresh rather than continuous streaming. A user can refresh the selected disease directly from ClinicalTrials.gov. Once extraction and validation complete, the clean and gold data layers are published atomically. A running dashboard will therefore see either the previous complete snapshot or the new complete snapshot, never a partly written file.

The interface checks for a newer local snapshot every five minutes. A freshness badge distinguishes a recently published live snapshot from the repository snapshot supplied with the project.

To refresh one disease from the command line:

```powershell
py -m py.live --cause DIAB
```

To inspect the current refresh status:

```powershell
py -m py.live --status
```

## Installation and local use

Python 3.11 or later is recommended.

```powershell
py -m pip install -r requirements.txt
py app.py
```

Open `http://127.0.0.1:8050` in a browser.

For a local production-style process, use Waitress:

```powershell
py -m waitress --listen=0.0.0.0:8050 app:server
```

## Rebuilding the analytical data

The complete pipeline can be run with:

```powershell
py -m py.build
```

The build performs three stages:

1. Retrieve and harmonise WHO disease burden data.
2. Retrieve and classify ClinicalTrials.gov records.
3. Construct the country, period and disease equity measures used by the dashboard.

A full ClinicalTrials.gov rebuild can take some time because records returned by several disease synonyms must be retrieved and deduplicated.

## Project structure

| Component | Purpose |
| --- | --- |
| `app.py` | Dash application and user-facing analytical views |
| `assets/style.css` | Responsive visual design |
| `config/sources.yaml` | Source, refresh and modelling settings |
| `py/ctgov.py` | ClinicalTrials.gov extraction and condition classification |
| `py/global_health_sources.py` | WHO burden extraction and harmonisation |
| `py/indices.py` | Country-level equity calculations |
| `py/analytics.py` | Forecasting and allocation simulation |
| `py/live.py` | Atomic selected-disease refresh and provenance metadata |
| `tests/` | Analytical unit tests |
| `data/clean/` | Harmonised source snapshots |
| `data/gold/` | Dashboard-ready analytical snapshots |
| `r/shiny_app.R` | Earlier Shiny implementation retained for reference |

Underscores appear only in literal code identifiers and filenames, where changing them would break imports or data contracts. User-facing headings and metric names use normal written English.

## Quality checks

Run the automated checks before publishing a new version:

```powershell
py -m pytest -q
py -m ruff check app.py py tests
py -m compileall -q app.py py
```

The tests cover the parity calculation, integer allocation, alignment score, forecast intervals, current-year interval handling, disease acronym matching and exclusion of WHO regional aggregates.

## Appropriate use

This application is a research planning and policy analysis tool. It should be used to identify questions, test allocation assumptions and expose geographic imbalances that warrant further investigation. It is not a clinical decision tool, a causal model, a substitute for country feasibility assessment or a basis for automated funding decisions.
