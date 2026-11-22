"""Statistical methods reference shown in the dashboard's Methods tab.

Each section is Markdown with MathJax ($...$ inline, $$...$$ display). The text
describes the code as implemented; when a method changes, change this file in
the same commit. Parameter values that live in config/sources.yaml are filled
in at runtime from ``params``.
"""

from __future__ import annotations

import re

_DISPLAY = re.compile(r"^\$\$(.+)\$\$$", re.MULTILINE)


def display_math(markdown: str) -> str:
    """Put each one-line $$...$$ equation on its own lines.

    Dash's Markdown renders $$ as a centred display equation only when the
    delimiters stand on separate lines; inline they render as text-size maths.
    """
    return _DISPLAY.sub(lambda match: f"$$\n{match.group(1)}\n$$", markdown)


def sections(params: dict) -> list[tuple[str, str, str]]:
    """Return (anchor, title, markdown) for each methods section."""
    return [(anchor, title, display_math(body)) for anchor, title, body in _sections(params)]


def _sections(params: dict) -> list[tuple[str, str, str]]:
    p = params
    return [
        ("notation", "1. Notation and analytical frame", r"""
The dashboard compares two distributions across countries for one disease and one range of years.

| Symbol | Meaning |
| --- | --- |
| $c = 1,\dots,N$ | Countries with measured burden for the disease (the *cohort*). |
| $s_c$ | Registered trial activity in country $c$ over the selected years (study-country units, defined in section 2). |
| $S = \sum_c s_c$ | Total activity in the cohort. |
| $d_c$ | Mean annual disability-adjusted life years (DALYs) in country $c$ (section 2). |
| $p_c = s_c / S$ | Trial activity share. |
| $b_c = d_c / \sum_k d_k$ | Burden share. |

Both $p$ and $b$ are probability distributions over the same countries ($\sum_c p_c = \sum_c b_c = 1$). Every equity measure in the dashboard is a way of describing how far $p$ is from $b$. The reference point is **proportional representation**: research activity distributed in the same proportions as the burden it addresses. This is a normative benchmark, not a claim that every country should host trials in exact proportion to burden.

Countries without a WHO burden estimate are excluded from the cohort, because their share of burden is unknown rather than zero.
"""),
        ("data", "2. Data construction", r"""
**Trial activity.** Records come from the ClinicalTrials.gov API v2. Only records whose registered study type is **Interventional** are included; observational studies and expanded-access records are excluded. The dashboard covers 33 diseases in nine groups. A study is assigned to a disease when any registered condition or MeSH term contains one of the disease's curated terms. Three rules keep the matching precise:

- Every term must start at a word boundary, so *renal* does not match *adrenal*.
- A term whose last word has three characters or fewer must also end at one, so *hepatitis a* does not match *hepatitis, alcoholic* and *DM* does not match inside another word.
- Each disease can list phrases that are removed before matching, because they contain a disease term but describe something else (*respiratory depression*, *pulmonary hypertension*, *diabetes insipidus*).

Studies that mention COVID-19 are not counted as lower respiratory infections, because WHO estimates COVID-19 burden separately. A study can count towards more than one disease. Each study is dated by its **start date**, falling back to the primary completion date and then the first-posted date. Current locations and countries recorded by ClinicalTrials.gov as removed from an earlier version are used for country activity; removed countries have no facility identity and never contribute capacity. Location country names are mapped to ISO 3166 codes; regional or unmappable labels are dropped.

The activity unit is the **unique study–country pair**: a study with ten facilities in Brazil and two in Kenya adds 1 to Brazil and 1 to Kenya. It measures *where research is placed*, not how many facilities or participants are involved. For a year range $[y_0, y_1]$:

$$s_c = \#\{\text{studies with start year in } [y_0,y_1] \text{ and at least one site in } c\}$$

**Disease burden.** WHO Global Health Estimates (GHE 2023 release) give absolute DALYs by country and cause for estimate years 2000, 2010, 2015 and 2019 to 2023. Each disease maps to one GHE cause code, except two composites that sum separate WHO causes:

- **Viral hepatitis:** acute hepatitis A, B, C and E, plus liver cancer and cirrhosis caused by hepatitis B and C. Most of the burden is chronic and WHO records it under those consequences.
- **Neglected tropical diseases:** the WHO neglected tropical diseases that GHE estimates separately.

Because viral hepatitis includes part of liver cancer and cirrhosis, those diseases overlap by design; each disease is analysed on its own. For a selected range, burden is the mean of the estimate years inside it:

$$d_c = \frac{1}{|Y|} \sum_{t \in Y} \text{DALY}_{c,t}, \qquad Y = \{t \in \text{GHE years} : y_0 \le t \le y_1\}$$

If no estimate year falls inside the range, $Y$ is the latest estimate year before $y_0$. Absolute DALYs (not rates) are used so that a country's share reflects how many people are affected, not only how intensely.

**Facility layer.** For the resource planner each study's locations are also kept at facility level (name, city, coordinates). Sponsors anonymise many listings ("Research Site", "GSK Investigational Site"); a listing is treated as anonymised when its name matches a placeholder pattern or when the same name appears in two or more cities within one study. Anonymised listings are excluded from the historical-activity proxy.

**Country context.** World Bank ICP 2021 price level indices for health and for GDP (World = 100), World Development Indicators for infrastructure, and World Bank country metadata (income group, capital). No missing value is imputed.
"""),
        ("equity", "3. Descriptive equity measures", r"""
**Equity Ratio (ER).**

$$\text{ER}_c = \frac{p_c}{b_c}$$

$\text{ER}_c = 1$ is proportional representation; $0.5$ means half the activity its burden share implies; $2$ means twice. The ratio is scale-free, so it compares small and large countries on the same footing. The map colours $\log_2 \text{ER}$, clipped to $[1/8, 8]$, so that under- and over-representation by the same factor are equally far from the neutral midpoint. Countries are banded as *critical* ($<0.5$), *under* ($0.5$–$0.8$), *aligned* ($0.8$–$1.25$), *above* ($1.25$–$2$) and *high* ($>2$); the aligned band is symmetric on the log scale ($0.8 = 1/1.25$).

**Burden underserved.** The share of burden located in countries with $\text{ER}_c < 0.8$:

$$U = \sum_{c\,:\,\text{ER}_c < 0.8} b_c$$

**Alignment score.** One minus the total variation distance (TVD) between the two distributions, on a 0 to 100 scale:

$$A = 100\left(1 - \tfrac{1}{2}\sum_c |p_c - b_c|\right)$$

TVD is the largest difference in probability the two distributions assign to any set of countries, equivalently the fraction of activity that would have to move to reach proportional representation. $A = 100$ means identical distributions; $A = 0$ means no overlap. TVD is a true metric (symmetric, satisfies the triangle inequality) and is bounded, which makes it suitable for communication and for optimisation.

**Theil divergence.** The Kullback–Leibler divergence of activity from burden:

$$T = \sum_{c\,:\,p_c > 0} p_c \ln \frac{p_c}{b_c}$$

$T \ge 0$, with equality only when $p = b$. It is unbounded and asymmetric, and it weights each country by its activity share, so it is most sensitive to activity concentrated where burden is small. Countries with no activity contribute zero, so $T$ does not directly penalise empty countries; that is why the alignment score is the headline measure and $T$ a complement.

**Placements needed for individual parity.** The number of additional placements $x_c$ that would bring country $c$ alone to $\text{ER}_c = 1$, allowing for the fact that adding activity also grows the total:

$$\frac{s_c + x_c}{S + x_c} = b_c \;\;\Longrightarrow\;\; x_c = \max\!\left(\frac{b_c S - s_c}{1 - b_c},\; 0\right)$$

Each $x_c$ assumes no other country changes, so these figures must not be added up into a global plan. The planners in sections 5 and 6 solve the joint problem.

The gold export file also carries a burden-weighted Gini coefficient of ER per period, which is not shown in the dashboard.
"""),
        ("forecast", "4. Forecast model", rf"""
**Target.** Annual global study–country activity for the selected disease, $Y_t = \sum_c s_{{c,t}}$. The forecast always uses the most recent complete years, independent of the year range selected at the top of the page, because a forecast should start from the latest information. The current (incomplete) year is excluded from fitting.

**Candidate model.** A log-linear trend is fitted by ordinary least squares to the last {p['lookback']} complete years (at least 4 are required):

$$\ln(1 + Y_t) = \alpha + \beta\,(t - t_0) + \varepsilon_t$$

$\beta$ is the average annual log growth rate; $e^{{\beta}} - 1$ is the implied annual growth. The $\log(1+\cdot)$ transform stabilises variance and keeps forecasts non-negative.

**Governance and validation.** At successive historical cut-offs the candidate trend and a last-value forecast are both used to predict the next one to {p['horizon']} years. Their weighted absolute percentage error is computed from these rolling-origin predictions. The trend is used only when it beats the last-value benchmark; otherwise the last observed complete year is carried forward.

**Prediction interval.** For each horizon, errors from the selected model's rolling-origin predictions are measured on the $\ln(1+Y)$ scale. The 10th and 90th percentiles are applied to the point forecast, producing an empirical 80% prediction interval. When fewer than five historical errors exist, an explicitly labelled in-sample residual fallback is used.

**Assumptions and limits.** Historical errors are assumed informative about future registry activity. Structural changes in registration, pipelines or sponsor strategy can still invalidate the forecast. This remains a transparent forecast of registered study-country activity, not enrolment, approvals, recruitment or study success.
"""),
        ("scenario", "5. Scenario lab: allocation rules and assumption sensitivity", rf"""
The scenario lab adds $X$ placements to the current distribution under one of three rules, then recomputes every share and measure.

**Targets.** After allocation the total is $S + X$, so each country's burden-proportional target is $b_c(S + X)$ and its gap is

$$g_c = \max\big(b_c (S + X) - s_c,\; 0\big)$$

| Rule | Weight $w_c$ |
| --- | --- |
| Equity gap | $g_c$ |
| Disease burden | $b_c$ |
| Hybrid | $0.7\,\dfrac{{g_c}}{{\sum_k g_k}} + 0.3\,b_c$ |

**Integer allocation.** Placements are whole numbers, assigned by the largest-remainder method: each country first receives $\lfloor X w_c / \sum_k w_k \rfloor$, and the remaining units go to the countries with the largest fractional parts. The total is exactly $X$.

**Assumption sensitivity.** {p['sim_draws']} simulations (seed {p['seed']}) vary two things at once:

1. *Burden.* Each $b_c$ is multiplied by an independent log-normal factor with mean 1 and coefficient of variation 10%, $\;\varepsilon_c \sim \text{{LogNormal}}(-\sigma^2/2,\, \sigma^2)$ with $\sigma = \sqrt{{\ln(1 + 0.1^2)}}$, then renormalised.
2. *Placement.* The $X$ placements are drawn from a multinomial distribution with probabilities equal to the deterministic allocation shares, representing sites that do not materialise exactly as planned.

The panel reports the 10th, 50th and 90th percentiles of the resulting alignment score and Theil divergence. These are **not confidence or prediction intervals**. The 10% variation and multinomial placement process are stated sensitivity assumptions; WHO does not publish country-level uncertainty intervals in the tables used here.
"""),
        ("planner", "6. Resource planner: cost-aware optimisation", rf"""
The scenario lab treats every placement as equal. The resource planner recognises that a placement can be made **at a facility that already runs trials** or **at a new site**, and that costs differ between countries.

**6.1 Historical activity proxy.** For a selected range of length $L$ years ending in year $y_1$, let $n_{{f,t}}$ be the number of studies of the disease that named facility $f$ started in year $t$. Its historical activity ceiling is its best $L$-year window up to $y_1$, and its *headroom proxy* is that peak minus its use in the selected range:

$$K_f = \max_{{\tau \le y_1}} \sum_{{t=\tau-L+1}}^{{\tau}} n_{{f,t}}, \qquad H_f = \mathbb{{1}}[f \text{{ active}}]\,\max\!\Big(K_f - \sum_{{t=y_0}}^{{y_1}} n_{{f,t}},\; 0\Big)$$

A facility is *active* if it started a study in the selected range or in the {p['recency']} equal-length window(s) before it. Only years up to $y_1$ are read, so the plan never uses information from after the period. A registry location is not proof that a site activated or enrolled participants; this quantity is a prioritisation proxy, not measured spare operational capacity.

**6.2 New-site throughput assumption.** A newly established site is assumed able to reach the chosen percentile (median, 75th or 90th) of $K_f$ among named facilities in the same country, if there are at least {p['min_facilities']}; otherwise in the same World Bank income group; otherwise across all countries. The percentile is an observed registry-history value (inverted empirical distribution), never interpolated, and is at least 1. It requires feasibility validation before use as an operational constraint.

**6.3 Cost model.** With planner inputs $a$ (study start-up cost per site) and $e$ (extra one-time cost of a new site), entered at the reference country's prices, and ICP 2021 health price level indices $\pi_c$:

$$\text{{cost of a study assigned to historical headroom}} = a\,\frac{{\pi_c}}{{\pi_{{\text{{ref}}}}}}, \qquad \text{{cost of a new site filled with }} m \text{{ studies}} = (e + m a)\,\frac{{\pi_c}}{{\pi_{{\text{{ref}}}}}}$$

If either country lacks a health index, both use the ICP GDP index instead, so the ratio is always on one basis. A country with neither cannot be costed and receives no placements; it is listed with its share of burden.

**6.4 Optimisation.** The goal is to maximise the alignment score $A$ subject to total cost $\le B$ (the budget). This is a budgeted integer problem with fixed costs, so the planner uses a greedy rule: at each step it evaluates, for every eligible country,

- **(a)** one more study at existing or already-opened capacity, and
- **(b)** a new site filled with $m_c = \min(\text{{capacity}}_c,\ \lceil x_c \rceil)$ studies, where $x_c$ is the parity need from section 3,

computes the exact change in $A$ (recalculating every share at the new total), and takes the action with the largest gain per dollar that fits the remaining budget. Existing headroom is used before slots in newly opened sites.

**Lookahead.** When the portfolio is small, any single extra placement can push a country past its burden share, so no single step improves $A$ even though the distributions are far apart (an integer local optimum). The planner then evaluates rounds of $N = 2, 4, 8, \dots$ placements, shared out by largest remainder in proportion to each country's gap to its target at total $S + N$, costs each round exactly (capacity first, then new sites), and applies the round with the best gain per dollar.

The planner stops when the budget is exhausted or when no action or round improves alignment. Greedy gain-per-cost is a well-established heuristic for budgeted problems of this kind; it does not guarantee the global optimum, but every step is explainable and the result is deterministic.

**6.5 Comparisons and outputs.**

- *Baseline:* the same rule with no capacity reuse, so each placement opens a new site costing $(e + a)\,\pi_c/\pi_{{\text{{ref}}}}$. This represents planning "one site per placement".
- *Spend frontier:* the greedy path run to three times the budget, plotting $A$ against cumulative spend for both plans.
- *Saving for the same equity:* the baseline's spend minus the resource-aware plan's spend at the point its frontier first reaches the baseline's alignment.
- *Sensitivity:* the plan recomputed with $e$ multiplied by 0.5, 1, 2 and 4. A plan that barely changes is robust to the new-site cost, which has no public benchmark.
- *Facility assignment:* a country's proxy-reuse placements go first to facilities with the most historical-activity headroom, which keeps the number of sites involved as small as the data allow.
- *Candidate cities:* for countries needing new sites, the cities with the most named facilities active in any tracked disease over the same recent window. A country with none is flagged, and its capital is shown on the map for reference only.
"""),
        ("validity", "7. Validity, uncertainty and what is not modelled", r"""
**Measurement.**

- Registration on ClinicalTrials.gov is near-universal for industry and US-funded trials but incomplete elsewhere, so activity in some regions is undercounted.
- Synonym matching can misclassify studies.
- Facility names are free text: variants of one hospital are counted as separate facilities and anonymised listings are dropped. More fundamentally, a registered location does not show activation, enrolment or unused capacity, so the facility layer is only a historical-activity signal.

**Burden.** GHE estimates are themselves modelled, with large uncertainty where vital registration is weak. WHO grades each country's data availability; those grades are kept in the clean data.

**Causality.** All measures are descriptive. A low ER does not show that trials *should* be placed in that country, nor that adding sites would recruit participants. Feasibility depends on eligible populations, standard of care, investigators, regulation, ethics, contracting, supply chains and community acceptability, none of which are in the model.

**Costs.** The price level index scales costs for the health sector as a whole; real site budgets depend on protocol, phase and local agreements. The planner's monetary defaults are placeholders for your own figures.

**Uncertainty that is quantified.**

- The forecast's empirical 80% prediction interval from rolling-origin errors (with an explicitly labelled residual fallback when history is sparse).
- The scenario lab's assumption-sensitivity range under its stated burden and placement perturbations.
- The planner's sensitivity to the new-site cost.

**Uncertainty that is not.** Parameter uncertainty in the forecast, GHE estimate intervals, and capacity measurement error.
"""),
        ("reproducibility", "8. Reproducibility", r"""
- Every calculation runs from versioned Parquet snapshots (`data/clean`, `data/gold`) published atomically, so a refresh never exposes a partial file.
- Random procedures use fixed seeds.
- Source timestamps and the WHO estimate years used are shown in the dashboard.
- `python -m py.validation` checks the interventional-study scope, disease and WHO code coverage, unique keys, finite burden, share denominators and gold-layer provenance. Warnings remain until protocol costs are approved, the disease classifier has independent blinded clinical validation, and candidate-site feasibility is verified outside the registry.
- The methods are implemented in `py/indices.py` (cohort and burden), `py/analytics.py` (measures, forecast, scenario lab) and `py/planner.py` (resource planner), with unit tests in `tests/` covering parity, integer allocation, alignment bounds, forecast intervals, capacity windows, cost scaling, the lookahead and year-range construction.

The full pipeline is rebuilt with `py -m py.build`.
"""),
    ]
