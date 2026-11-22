"""Global Clinical Trial Equity Intelligence dashboard.

Run locally with ``python app.py``. The application reads atomic parquet
snapshots produced by the Python pipeline and can refresh one disease directly
from ClinicalTrials.gov.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import os
from pathlib import Path

from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Group
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from py.analytics import (
    alignment_score,
    allocate_sites,
    forecast_trial_sites,
    simulate_allocation,
    sites_to_parity,
    theil_index,
)
from py.global_health_sources import DISEASE_GROUPS, DISEASE_PATTERNS
from py.indices import scope_for_years
from py.live import read_status, refresh
from py.methods import display_math, sections as methods_sections
from py.planner import (
    CAPACITY_PERCENTILES,
    assign_to_facilities,
    cost_scale,
    facility_capacity,
    new_site_capacity,
    optimise_plan,
    research_cities,
)


ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "data" / "gold" / "country_interval_disease.parquet"
CT_PATH = ROOT / "data" / "clean" / "ctgov_sites.parquet"
WHO_PATH = ROOT / "data" / "clean" / "who_daly_selected.parquet"
FACILITIES_PATH = ROOT / "data" / "clean" / "ctgov_facilities.parquet"
RESOURCES_PATH = ROOT / "data" / "clean" / "country_resources.parquet"
CFG = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
POLL_MS = int(CFG.get("live", {}).get("dashboard_poll_seconds", 300)) * 1000
FRESHNESS_WARNING_HOURS = int(CFG.get("live", {}).get("freshness_warning_hours", 36))
FORECAST_HORIZON = int(CFG.get("live", {}).get("forecast_horizon_years", 3))
FORECAST_LOOKBACK = int(CFG.get("live", {}).get("forecast_lookback_years", 8))
SIMULATION_DRAWS = int(CFG.get("live", {}).get("simulation_draws", 500))
CURRENT_YEAR = datetime.now(timezone.utc).year
PLANNER = CFG.get("planner", {})
FIRST_YEAR = int(CFG.get("years", {}).get("start", 2000))
RECENCY_WINDOWS = int(PLANNER.get("recency_windows", 1))
# Default view: the five most recent complete years.
DEFAULT_YEARS = [CURRENT_YEAR - 5, CURRENT_YEAR - 1]

# Two-ink print palette on warm paper. Prussian blue carries data; vermilion is
# reserved for under-representation (the gap) and for new ground in the plan.
# Both validated together for colour-vision deficiency (protan dE 20.8).
PAPER = "#F4F1EA"
WASH = "#E8E2D5"      # no-data land, quiet fills
RULE = "#CFC8B8"      # hairlines
GRID = "#E0D9CB"
INK = "#16140F"       # text
INK2 = "#57534A"      # secondary text
FAINT = "#9A9486"     # tertiary text, baselines
PRUSSIAN = "#2560A6"
VERMILION = "#D2491F"
PRUSSIAN_SOFT = "rgba(37, 96, 166, 0.12)"
SERIF = "Fraunces, Georgia, serif"
MONO = "'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace"
# Diverging scale for the equity ratio on a log2 axis: vermilion below parity,
# warm neutral at parity, Prussian above.
ER_SCALE = [[0.0, VERMILION], [0.25, "#E7A389"], [0.5, "#E6E0D3"], [0.75, "#9DB2D1"], [1.0, PRUSSIAN]]
BAND_COLOURS = [VERMILION, "#E7A389", "#CBC4B4", "#9DB2D1", PRUSSIAN]
# How each disease reads mid-sentence in the headline ("...the world's ___ burden...").
SENTENCE_NAMES = {
    "HIV": "HIV/AIDS", "TB": "tuberculosis", "MALARIA": "malaria", "LRI": "lower respiratory infection",
    "DIARR": "diarrhoeal disease", "HEPATITIS": "viral hepatitis", "MENING": "meningitis", "MEASLES": "measles",
    "NTD": "neglected tropical disease", "MATERNAL": "maternal", "NEON": "neonatal",
    "LUNG_CA": "lung cancer", "BREAST_CA": "breast cancer", "CERVIX_CA": "cervical cancer",
    "COLORECTAL_CA": "colorectal cancer", "LIVER_CA": "liver cancer", "STOMACH_CA": "stomach cancer",
    "PROSTATE_CA": "prostate cancer", "IHD": "ischaemic heart disease", "STROKE": "stroke",
    "HHD": "hypertensive heart disease", "DIAB": "diabetes", "KIDNEY": "kidney disease", "COPD": "COPD",
    "ASTHMA": "asthma", "DEPRESSION": "depression", "ANXIETY": "anxiety disorder",
    "ALZH": "Alzheimer's and dementia", "CIRRHOSIS": "liver cirrhosis", "RTI": "road injury",
    "SELF_HARM": "self-harm", "SICKLE": "sickle cell disease", "CONGENITAL": "congenital anomaly",
}

def _snapshot_token() -> str:
    paths = (GOLD_PATH, CT_PATH, WHO_PATH, FACILITIES_PATH, RESOURCES_PATH)
    stamps = [path.stat().st_mtime_ns for path in paths if path.exists()]
    return str(max(stamps, default=0))


@lru_cache(maxsize=2)
def _annual_cached(_token: str) -> pd.DataFrame:
    if not CT_PATH.exists():
        return pd.DataFrame(columns=["country_iso3", "year", "cause_code", "trial_sites"])
    frame = pd.read_parquet(CT_PATH)
    if "study_type" not in frame:
        raise RuntimeError("The trial snapshot is legacy/unfiltered. Run `python -m py.ctgov` before planning.")
    study_type = frame["study_type"].astype(str).str.upper()
    if set(study_type.dropna()) != {"INTERVENTIONAL"}:
        raise RuntimeError("Planning requires a fully rebuilt INTERVENTIONAL-only trial snapshot.")
    frame = frame[study_type == "INTERVENTIONAL"].copy()
    frame["trial_sites"] = pd.to_numeric(frame["trial_sites"], errors="coerce").fillna(0)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    return frame.dropna(subset=["year"]).astype({"year": int})


def _read_annual() -> pd.DataFrame:
    return _annual_cached(_snapshot_token())


@lru_cache(maxsize=2)
def _who_cached(_token: str) -> pd.DataFrame:
    if not WHO_PATH.exists():
        raise FileNotFoundError("Run `python -m py.global_health_sources` to create the burden layer.")
    return pd.read_parquet(WHO_PATH)


def _years_label(start: int, end: int) -> str:
    text = f"{start} to {end}" if start != end else str(start)
    return text + (" (to date)" if end == CURRENT_YEAR else "")


def _scope(token: str, cause_code: str, start: int, end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All countries for the selected disease and years, and the burden-observed cohort."""
    scope = scope_for_years(_annual_cached(token), _who_cached(token), cause_code, start, end)
    if scope.empty:
        raise ValueError("No data are available for the selected scope")
    selected = scope[scope["burden_dalys"].fillna(0) > 0].copy()
    return scope, selected


def _base_figure(title: str | None = None) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title)
    return style_figure(figure)


def style_figure(figure: go.Figure, height: int = 380) -> go.Figure:
    """Print-like chart styling: no boxes, dotted value grid, monospaced labels."""
    figure.update_layout(
        height=height,
        margin=dict(l=6, r=18, t=30, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, color=INK2, size=11),
        hoverlabel=dict(bgcolor=PAPER, bordercolor=INK, font=dict(family=MONO, color=INK, size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=10, color=INK2), bgcolor="rgba(0,0,0,0)"),
        bargap=0.45,
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor=RULE, ticks="outside", tickcolor=RULE,
                        ticklen=4, title_font=dict(size=10, color=FAINT))
    figure.update_yaxes(gridcolor=GRID, griddash="dot", zeroline=False, showline=False,
                        title_font=dict(size=10, color=FAINT))
    return figure


def style_geo(figure: go.Figure, height: int = 440) -> go.Figure:
    figure.update_geos(showframe=False, showcoastlines=False, showcountries=False, showland=True,
                       landcolor=WASH, showocean=False, showlakes=False, bgcolor="rgba(0,0,0,0)",
                       projection_type="equal earth", lataxis_range=[-58, 85])
    return style_figure(figure, height).update_layout(margin=dict(l=0, r=0, t=6, b=0))


def _pct_tick(value: float) -> str:
    return f"{value * 100:g}%"


def ledger_item(label: str, value_id: str, detail_id: str | None = None, detail: str = "") -> html.Div:
    return html.Div([
        html.Div(label, className="ledger-label"),
        html.Div(id=value_id, className="ledger-value"),
        html.Div(id=detail_id, className="ledger-detail") if detail_id else html.Div(detail, className="ledger-detail"),
    ], className="ledger-item")


def figure_block(number: int, title: str, caption: str, graph_id: str) -> html.Figure:
    """A chart panel: title, then how to read it, then the chart."""
    return html.Figure([
        html.Div([html.Span(f"Fig. {number:02d}", className="fig-no"), html.H3(title)], className="fig-head"),
        html.Figcaption(caption),
        dcc.Loading(dcc.Graph(id=graph_id, config={"displayModeBar": False}), type="dot", color=VERMILION),
    ], className="fig")


def chapter(label: str, value: str, children: list) -> dcc.Tab:
    return dcc.Tab(label=label, value=value, className="chapter",
                   selected_className="chapter--active", children=html.Div(children, className="chapter-page"))


TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={"fontFamily": "'IBM Plex Sans', Arial, sans-serif", "fontSize": 12, "padding": "9px 12px",
                "border": "none", "borderBottom": f"1px solid {GRID}", "backgroundColor": "transparent",
                "color": INK, "whiteSpace": "normal", "minWidth": 90, "maxWidth": 260, "textAlign": "left"},
    style_header={"fontFamily": MONO, "fontSize": 10, "fontWeight": 500, "textTransform": "uppercase",
                  "letterSpacing": "0.06em", "color": INK2, "backgroundColor": "transparent",
                  "borderBottom": f"1px solid {INK}"},
    style_filter={"backgroundColor": "transparent", "borderBottom": f"1px solid {GRID}"},
    style_data_conditional=[{"if": {"state": "active"}, "backgroundColor": WASH, "border": "none"}],
)

# ------------------------------------------------------------ resource planner

PLAN_TABLE_COLUMNS = [
    ("Country", "country_name"), ("Burden share", "burden_share"),
    ("Equity ratio now", "equity_ratio_before"), ("Equity ratio after", "equity_ratio_after"),
    ("Placements", "placements"), ("Against historical headroom", "placements_existing"),
    ("New sites", "new_sites"), ("Assumed studies per new site", "new_site_capacity"),
    ("Throughput-assumption basis", "capacity_basis"), ("Cost (US$)", "cost"),
    ("Cost index vs reference", "cost_scale"), ("Cost basis", "cost_basis"),
    ("Candidate cities for new sites", "candidate_cities"),
    ("Physicians per 1,000", "physicians_per_1000"), ("Hospital beds per 1,000", "hospital_beds_per_1000"),
    ("Electricity access %", "electricity_access_pct"),
    ("Health spend per capita (US$)", "health_spend_per_capita_usd"),
    ("Logistics index (1 to 5)", "logistics_performance_index"),
    ("Researchers per million", "researchers_per_million"),
]


@lru_cache(maxsize=2)
def _facilities(_token: str) -> pd.DataFrame:
    if not FACILITIES_PATH.exists():
        return pd.DataFrame(columns=["nct_id", "cause_code", "year", "country_iso3", "facility_key",
                                     "facility", "city", "lat", "lon", "anonymised"])
    try:
        study_types = pd.read_parquet(FACILITIES_PATH, columns=["study_type"])
    except Exception as exc:
        raise RuntimeError(
            "The facility snapshot is legacy/unfiltered. Run `python -m py.ctgov` before planning."
        ) from exc
    if set(study_types["study_type"].dropna().astype(str).str.upper()) != {"INTERVENTIONAL"}:
        raise RuntimeError("Planning requires a fully rebuilt INTERVENTIONAL-only facility snapshot.")
    return pd.read_parquet(FACILITIES_PATH, columns=[
        "nct_id", "cause_code", "year", "country_iso3", "facility_key",
        "facility", "city", "lat", "lon", "anonymised",
    ])


@lru_cache(maxsize=2)
def _resources(_token: str) -> pd.DataFrame:
    if not RESOURCES_PATH.exists():
        raise FileNotFoundError("Run `python -m py.resources` to create the country resource layer.")
    return pd.read_parquet(RESOURCES_PATH)


@lru_cache(maxsize=32)
def _capacity(token: str, cause_code: str, start: int, end: int) -> pd.DataFrame:
    return facility_capacity(_facilities(token), cause_code, start, end, RECENCY_WINDOWS)


@lru_cache(maxsize=8)
def _cities(token: str, start: int, end: int) -> pd.DataFrame:
    return research_cities(_facilities(token), start, end, RECENCY_WINDOWS)


@lru_cache(maxsize=64)
def _plan_inputs(token: str, cause_code: str, start: int, end: int, reference: str, percentile: float, scaled: bool):
    _, selected = _scope(token, cause_code, start, end)
    portfolio = selected[["country_iso3", "trial_sites", "burden_share", "equity_ratio"]].reset_index(drop=True)
    capacity = _capacity(token, cause_code, start, end)
    resources = _resources(token)
    headroom = capacity.groupby("country_iso3")["headroom"].sum()
    hubs = new_site_capacity(capacity, resources, portfolio["country_iso3"], percentile,
                             int(PLANNER.get("min_facilities_for_benchmark", 5))).set_index("country_iso3")
    if scaled:
        scale = cost_scale(resources, portfolio["country_iso3"], reference).set_index("country_iso3")
    else:
        scale = pd.DataFrame({"cost_scale": 1.0, "cost_basis": "Not scaled"}, index=portfolio["country_iso3"])
    return portfolio, capacity, headroom, hubs, scale


@lru_cache(maxsize=64)
def _plans(token: str, cause_code: str, start: int, end: int, budget: float, study_cost: float,
           site_cost: float, reference: str, percentile: float, scaled: bool):
    portfolio, _, headroom, hubs, scale = _plan_inputs(token, cause_code, start, end, reference, percentile, scaled)
    args = (portfolio, headroom, hubs["new_site_capacity"], scale["cost_scale"], study_cost)
    smart = optimise_plan(*args, site_cost, budget)
    naive = optimise_plan(*args, site_cost, budget, reuse_capacity=False)
    # The frontier extends past the budget so the planner can see what more money buys.
    smart_curve = optimise_plan(*args, site_cost, 3 * budget).steps
    naive_curve = optimise_plan(*args, site_cost, 3 * budget, reuse_capacity=False).steps
    sensitivity = [(factor, optimise_plan(*args, site_cost * factor, budget)) for factor in (0.5, 1, 2, 4)]
    return smart, naive, smart_curve, naive_curve, sensitivity


def _money(value: float) -> str:
    if not np.isfinite(value):
        return "Not available"
    if abs(value) >= 1e6:
        return f"US${value / 1e6:,.2f}M"
    return f"US${value:,.0f}"


def _cost_to_reach(curve: pd.DataFrame, target: float) -> float:
    reached = curve[curve["alignment"] >= target - 1e-9]
    return float(reached["cumulative_cost"].iloc[0]) if len(reached) else np.nan


def planner_controls() -> html.Div:
    resources = _resources(_snapshot_token())
    priced = resources[resources["health_price_level"].notna()].sort_values("country_name")
    reference_options = [{"label": f"{row.country_name} (health price level {row.health_price_level:.0f})",
                          "value": row.country_iso3} for row in priced.itertuples()]

    def money_input(label: str, input_id: str, value: float, note: str) -> html.Div:
        return html.Div([
            html.Label(label),
            dcc.Input(id=input_id, type="number", min=0, step=1000, value=value, debounce=True, className="money-input"),
            html.Small(note),
        ], className="planner-field")

    return html.Div([
        money_input("Budget (US$)", "plan-budget", PLANNER.get("budget", 5_000_000),
                    "Total spend available for site start-up"),
        money_input("Study start-up cost per site (US$)", "plan-study-cost", PLANNER.get("study_startup_cost", 25_000),
                    "Activating one study at one qualified site. Default: widely cited Tufts CSDD figure, US prices"),
        money_input("Extra one-time cost of a new site (US$)", "plan-site-cost", PLANNER.get("new_site_cost", 100_000),
                    "Qualifying and equipping a facility new to this disease. No public benchmark: enter your own"),
        html.Div([
            html.Label("Costs are entered at the prices of"),
            dcc.Dropdown(id="plan-reference", options=reference_options,
                         value=PLANNER.get("reference_country", "USA"), clearable=False),
            dcc.Checklist(id="plan-scaled", options=[{"label": " Scale each country's costs by its health price level", "value": "scaled"}],
                          value=["scaled"], className="planner-check"),
        ], className="planner-field"),
        html.Div([
            html.Label("Studies a new site can host in a range this long"),
            dcc.RadioItems(id="plan-capacity", options=list(CAPACITY_PERCENTILES), value="Median",
                           inline=True, className="choice"),
            html.Small("Measured from comparable facilities: same country, else income group, else all countries"),
        ], className="planner-field"),
    ], className="planner-controls")


METHOD_PARAMS = {
    "lookback": FORECAST_LOOKBACK, "horizon": FORECAST_HORIZON, "forecast_draws": 1000,
    "sim_draws": SIMULATION_DRAWS, "seed": 42, "recency": RECENCY_WINDOWS,
    "min_facilities": int(PLANNER.get("min_facilities_for_benchmark", 5)),
}


def methods_tab_content() -> html.Div:
    parts = methods_sections(METHOD_PARAMS)
    contents = html.Nav([html.B("Contents")] + [html.A(title, href=f"#{anchor}") for anchor, title, _ in parts]
                        + [html.A("Worked example on your selection", href="#worked-example")], className="methods-nav")
    blocks = [html.Section([html.H3(title), dcc.Markdown(body, mathjax=True, className="methods-md")],
                           id=anchor, className="methods-section") for anchor, title, body in parts]
    example = html.Section([
        html.H3("Worked example on your selection"),
        html.P("The formulas above, evaluated on the disease and years chosen at the top of the page.", className="methods-note"),
        dcc.Loading(dcc.Markdown(id="methods-example", mathjax=True, className="methods-md"), type="dot"),
    ], id="worked-example", className="methods-section")
    return html.Div([contents, html.Div(blocks + [example], className="methods-body")], className="methods-layout")


# The disease menu is grouped: a disabled heading row precedes each group.
cause_options = []
for group_name, group_causes in DISEASE_GROUPS:
    cause_options.append({"label": group_name.upper(), "value": f"group:{group_name}", "disabled": True})
    cause_options.extend({"label": DISEASE_PATTERNS[code][0], "value": code} for code in group_causes)
initial_cause = DISEASE_GROUPS[0][1][0]
# Five-yearly ticks plus the current year, dropping ticks too close to it to read.
year_marks = {year: str(year) for year in range(FIRST_YEAR, CURRENT_YEAR - 2, 5)}
year_marks[CURRENT_YEAR] = str(CURRENT_YEAR)

app = Dash(__name__, title="Trial Equity Atlas", update_title="Recomputing…")
server = app.server

app.layout = html.Div(
    [
        dcc.Store(id="snapshot-token", data=_snapshot_token()),
        dcc.Interval(id="snapshot-poll", interval=POLL_MS, n_intervals=0),
        # Fixed control bar: what you are looking at is always in reach.
        html.Header([
            html.Div("Trial Equity Atlas", className="bar-brand"),
            html.Div([
                html.Label("Disease", htmlFor="cause", className="bar-label"),
                dcc.Dropdown(id="cause", options=cause_options, value=initial_cause, clearable=False, searchable=True,
                             maxHeight=460, optionHeight=34, placeholder="Search diseases", className="bar-select"),
            ], className="bar-field bar-disease"),
            html.Div([
                html.Div([html.Span("Years", className="bar-label"), html.Span(id="years-label", className="years-label")],
                         className="bar-field-head"),
                dcc.RangeSlider(id="years", min=FIRST_YEAR, max=CURRENT_YEAR, step=1, value=DEFAULT_YEARS,
                                marks=year_marks, allowCross=False, pushable=0, updatemode="mouseup",
                                tooltip={"placement": "bottom", "always_visible": False}, className="year-slider"),
            ], className="bar-field bar-years"),
            html.Div([
                html.Span(id="freshness-badge", className="freshness-badge"),
                html.Span(id="last-updated", className="bar-updated"),
                html.Button("Refresh", id="refresh-live", n_clicks=0, className="text-button"),
                html.Div(id="refresh-message", className="refresh-message"),
            ], className="bar-meta"),
        ], className="control-bar"),
        html.Section([
            html.H1(id="headline", className="headline"),
            html.P(id="dek", className="dek"),
        ], className="summary"),
        html.Section([
            ledger_item("Trial placements", "kpi-sites", "kpi-sites-detail"),
            ledger_item("Countries reached", "kpi-countries", "kpi-countries-detail"),
            ledger_item("Alignment", "kpi-alignment", "kpi-alignment-detail"),
            ledger_item("Burden underserved", "kpi-underserved", "kpi-underserved-detail"),
            ledger_item("Theil divergence", "kpi-theil", "kpi-theil-detail"),
        ], className="ledger ledger-5"),
        dcc.Tabs(id="workspace-tabs", value="overview", className="chapters", parent_className="chapters-parent", mobile_breakpoint=0, children=[
            chapter("Overview", "overview", [
                html.Div([
                    figure_block(1, "Where trials go, relative to where disease is", "Equity ratio: a country's share of trials divided by its share of burden. Vermilion countries host fewer trials than their burden implies; blue, more. Unshaded land has no burden estimate.", "equity-map"),
                    figure_block(2, "The largest gaps", "Additional placements each country would need to reach parity on its own. Each figure assumes no other country changes, so they do not add up.", "priority-bars"),
                ], className="spread spread-2-1"),
                html.Div([
                    figure_block(3, "Burden against trials, country by country", "Each circle is a country, sized by burden. On the dotted diagonal trials match burden exactly; below it, a country is under-researched. Both axes are logarithmic.", "alignment-scatter"),
                    figure_block(4, "Countries by equity band", "Count of countries in each band of the equity ratio, from critical (under half their fair share) to high (over double).", "er-distribution"),
                ], className="spread spread-1-1"),
            ]),
            chapter("Forecast", "forecast", [
                html.Div([
                    figure_block(5, "Global trial placements, observed and projected", f"Annual interventional study-country placements, with a {FORECAST_HORIZON}-year projection. The trend is used only if it beats a last-value benchmark in rolling validation; the shaded band uses empirical forecast errors.", "forecast-chart"),
                    html.Aside([html.Div("Model readout", className="aside-title"), html.Div(id="forecast-readout", className="narrative")], className="aside"),
                ], className="spread spread-2-1"),
                html.P("Forecasts extrapolate the recent global trend. They do not predict protocol feasibility, approvals, recruitment or causal effects. The Methods chapter sets out the model in full.", className="chapter-note"),
            ]),
            chapter("Scenarios", "scenario", [
                html.P("Add a number of new placements, choose how to share them out, and see how the distribution would change. Uncertainty comes from 500 simulations that vary burden and where the placements land.", className="chapter-intro"),
                html.Div([
                    html.Div([html.Label("Placements to add"), dcc.Slider(id="extra-sites", min=0, max=1000, step=25, value=250, marks={0: "0", 250: "250", 500: "500", 750: "750", 1000: "1,000"}, className="plain-slider")], className="instrument"),
                    html.Div([html.Label("Allocation rule"), dcc.RadioItems(id="strategy", options=["Equity gap", "Disease burden", "Hybrid"], value="Equity gap", inline=True, className="choice")], className="instrument"),
                ], className="instruments instruments-2"),
                html.Section([
                    ledger_item("Projected alignment", "scenario-alignment", "scenario-alignment-range"),
                    ledger_item("Projected Theil", "scenario-theil", "scenario-theil-range"),
                    ledger_item("Countries receiving placements", "scenario-recipients", detail="Whole-number allocation"),
                ], className="ledger ledger-3"),
                html.Div([
                    figure_block(6, "Where the new placements go", "Top recipients under the chosen rule.", "allocation-bars"),
                    figure_block(7, "Equity ratio before and after", "The twenty highest-burden countries. Grey: today. Blue: after the scenario. The dotted line is parity.", "scenario-comparison"),
                ], className="spread spread-1-1"),
            ]),
            chapter("Resource plan", "planner", [
                html.P("The equity view counts placements a country needs. The planner uses named facilities' historical registry activity as a prioritisation signal, then compares reuse with opening new sites. Registry history is not measured spare capacity, recruitment performance or operational feasibility.", className="chapter-intro"),
                planner_controls(),
                html.Section([
                    ledger_item("Alignment for this budget", "plan-alignment", "plan-alignment-detail"),
                    ledger_item("Sites involved", "plan-sites", "plan-sites-detail"),
                    ledger_item("Budget used", "plan-spent", "plan-spent-detail"),
                    ledger_item("Saving for the same equity", "plan-saving", "plan-saving-detail"),
                ], className="ledger ledger-4"),
                html.Div([
                    figure_block(8, "What each dollar buys", "Alignment against cumulative spend. Blue: the resource-aware plan. Grey dashed: one new site per placement. The dotted rule marks the budget.", "plan-frontier"),
                    figure_block(9, "Where the placements go", "Top recipients. Blue: studies assigned against historical activity headroom. Vermilion: studies assigned to assumed new sites.", "plan-countries"),
                ], className="spread spread-1-1"),
                html.Div([
                    figure_block(10, "Sites in the plan", "Blue circles: existing facilities taking extra studies, sized by studies added. Vermilion diamonds: new sites, placed at the country's most research-active city.", "plan-map"),
                    html.Aside([html.Div("If a new site costs more or less", className="aside-title"), html.Div(id="plan-sensitivity", className="sensitivity")], className="aside"),
                ], className="spread spread-2-1"),
                html.Section([
                    html.Div([html.Span("Table", className="fig-no"), html.H3("Country plan with cost and infrastructure")], className="fig-head"),
                    dash_table.DataTable(id="plan-table", page_size=15, filter_action="native", sort_action="native", export_format="csv", **TABLE_STYLE),
                    html.Div(id="plan-excluded", className="plan-excluded"),
                ], className="table-block"),
                html.Div(id="plan-method", className="chapter-note"),
            ]),
            chapter("Methods", "methods", [methods_tab_content()]),
            chapter("Data & sources", "quality", [
                html.Div([
                    html.Section([html.Div("Coverage", className="aside-title"), html.Div(id="quality-summary", className="quality-grid")], className="aside"),
                    html.Section([html.Div("Provenance", className="aside-title"), html.Div(id="provenance", className="provenance")], className="aside"),
                ], className="spread spread-1-1"),
                html.Section([
                    html.Div([html.Span("Table", className="fig-no"), html.H3("Country-level analytical table")], className="fig-head"),
                    dash_table.DataTable(id="country-table", page_size=15, filter_action="native", sort_action="native", export_format="csv", **TABLE_STYLE),
                ], className="table-block"),
            ]),
        ]),
        html.Footer([
            html.Span("Descriptive analytics and planning scenarios. Not causal evidence, and not medical advice."),
            html.Span("Sources: ClinicalTrials.gov · WHO Global Health Estimates · World Bank ICP and WDI"),
        ], className="colophon"),
    ],
    className="atlas",
)

@callback(
    Output("snapshot-token", "data"),
    Output("refresh-message", "children"),
    Input("snapshot-poll", "n_intervals"),
    Input("refresh-live", "n_clicks"),
    State("cause", "value"),
    prevent_initial_call=False,
)
def refresh_or_poll(_ticks: int, clicks: int, cause_code: str):
    if ctx.triggered_id == "refresh-live" and clicks:
        try:
            result = refresh(cause_code)
            return _snapshot_token(), f"Published {result['cause_name']} · {result['completed_at'][11:16]} UTC"
        except Exception as exc:
            return no_update, f"Refresh failed: {exc}"
    return _snapshot_token(), no_update


@callback(
    Output("years-label", "children"),
    Input("years", "value"),
    Input("cause", "value"),
    Input("snapshot-token", "data"),
)
def describe_years(years: list[int], cause_code: str, token: str):
    start, end = int(years[0]), int(years[1])
    span = end - start + 1
    _, selected = _scope(token, cause_code, start, end)
    # Compact enough for the control bar; the full detail sits in the tooltip.
    detail = [f"{span} year{'s' if span != 1 else ''} of trial starts"]
    if end == CURRENT_YEAR:
        detail.append(f"{CURRENT_YEAR} counts studies registered so far this year")
    if len(selected):
        detail.append(f"burden from WHO estimates for {selected['burden_years'].iloc[0]}")
    return html.Span([html.B(_years_label(start, end))], title="; ".join(detail).capitalize())


@callback(
    Output("kpi-sites", "children"), Output("kpi-sites-detail", "children"),
    Output("kpi-countries", "children"), Output("kpi-countries-detail", "children"),
    Output("kpi-alignment", "children"), Output("kpi-alignment-detail", "children"),
    Output("kpi-underserved", "children"), Output("kpi-underserved-detail", "children"),
    Output("kpi-theil", "children"), Output("kpi-theil-detail", "children"),
    Output("equity-map", "figure"), Output("priority-bars", "figure"),
    Output("alignment-scatter", "figure"), Output("er-distribution", "figure"),
    Output("forecast-chart", "figure"), Output("forecast-readout", "children"),
    Output("scenario-alignment", "children"), Output("scenario-alignment-range", "children"),
    Output("scenario-theil", "children"), Output("scenario-theil-range", "children"),
    Output("scenario-recipients", "children"), Output("allocation-bars", "figure"),
    Output("scenario-comparison", "figure"), Output("quality-summary", "children"),
    Output("provenance", "children"), Output("country-table", "data"), Output("country-table", "columns"),
    Output("freshness-badge", "children"), Output("freshness-badge", "className"), Output("last-updated", "children"),
    Output("headline", "children"), Output("dek", "children"),
    Input("cause", "value"), Input("years", "value"), Input("extra-sites", "value"),
    Input("strategy", "value"), Input("snapshot-token", "data"),
)
def render_dashboard(cause_code: str, years: list[int], extra_sites: int, strategy: str, token: str):
    start, end = int(years[0]), int(years[1])
    scope, selected = _scope(token, cause_code, start, end)
    total_sites = selected["trial_sites"].sum()
    covered = int((selected["trial_sites"] > 0).sum())
    all_countries = len(selected)
    align = alignment_score(selected["trial_share"], selected["burden_share"])
    underserved_burden = selected.loc[selected["equity_ratio"].fillna(0) < 0.8, "burden_share"].sum()
    theil = theil_index(selected["trial_share"], selected["burden_share"])
    parity = sites_to_parity(selected["trial_sites"], selected["burden_share"])
    selected["sites_to_parity"] = np.ceil(parity)
    selected["er_log"] = np.log2(selected["equity_ratio"].clip(lower=0.125, upper=8).fillna(0.125))

    names = _resources(token).set_index("country_iso3")["country_name"]
    selected["country_name"] = selected["country_iso3"].map(names).fillna(selected["country_iso3"])

    # Fig. 1: equity ratio map on a log2 scale, vermilion below parity.
    map_fig = go.Figure(go.Choropleth(
        locations=selected["country_iso3"], z=selected["er_log"], zmin=-3, zmax=3, colorscale=ER_SCALE,
        marker=dict(line=dict(color=PAPER, width=0.6)),
        customdata=np.column_stack([selected["country_name"], selected["equity_ratio"].fillna(0),
                                    selected["trial_sites"], selected["burden_share"]]),
        hovertemplate="<b>%{customdata[0]}</b><br>Equity ratio %{customdata[1]:.2f}<br>Placements %{customdata[2]:,.0f}"
                      "<br>Burden share %{customdata[3]:.2%}<extra></extra>",
        colorbar=dict(orientation="h", x=0, xanchor="left", y=-0.02, yanchor="top", len=0.42, thickness=7,
                      outlinewidth=0, tickvals=[-3, -2, -1, 0, 1, 2, 3],
                      ticktext=["0.125", "0.25", "0.5", "parity", "2", "4", "8"], tickfont=dict(size=10, color=INK2)),
    ))
    style_geo(map_fig, 460)

    # Fig. 2: parity gaps, one colour (they are all gaps).
    priority = selected.nlargest(12, "sites_to_parity").sort_values("sites_to_parity")
    priority_fig = _base_figure()
    priority_fig.add_trace(go.Bar(
        x=priority["sites_to_parity"], y=priority["country_name"], orientation="h", marker=dict(color=VERMILION),
        text=[f"{v:,.0f}" for v in priority["sites_to_parity"]], textposition="outside", cliponaxis=False,
        textfont=dict(family=MONO, size=10, color=INK2), customdata=priority[["equity_ratio", "burden_share"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} placements to parity<br>Equity ratio %{customdata[0]:.2f}"
                      "<br>Burden share %{customdata[1]:.2%}<extra></extra>"))
    priority_fig.update_xaxes(showticklabels=False, ticks="", showline=False).update_yaxes(showgrid=False, title=None)
    style_figure(priority_fig, 460).update_xaxes(showticklabels=False, ticks="", showline=False)

    # Fig. 3: burden share against trial share on log axes; countries with no
    # trials sit on a labelled floor so they stay visible.
    positive = selected.loc[selected["trial_share"] > 0, "trial_share"]
    # The "none" row sits half a decade below the lowest decade with data, so it
    # never collides with a percentage tick.
    floor = 10 ** (np.floor(np.log10(positive.min())) - 0.5) if len(positive) else 1e-4
    plotted = selected.assign(y=np.where(selected["trial_share"] > 0, selected["trial_share"], floor),
                              size=6 + 30 * np.sqrt(selected["burden_share"] / selected["burden_share"].max()))
    under = plotted["equity_ratio"].fillna(0) < 0.8
    scatter = _base_figure()
    low = min(floor, plotted["burden_share"].min()) * 0.7
    high = max(plotted["y"].max(), plotted["burden_share"].max()) * 1.4
    scatter.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Parity",
                                 line=dict(color=INK2, width=1, dash="dot"), hoverinfo="skip"))
    for mask, name, colour in ((~under, "At or above 0.8× fair share", PRUSSIAN), (under, "Below 0.8× fair share", VERMILION)):
        group = plotted[mask]
        scatter.add_trace(go.Scatter(
            x=group["burden_share"], y=group["y"], mode="markers", name=name,
            marker=dict(color=colour, size=group["size"], opacity=0.72, line=dict(color=PAPER, width=1)),
            customdata=np.column_stack([group["country_name"], group["trial_share"], group["equity_ratio"].fillna(0)]),
            hovertemplate="<b>%{customdata[0]}</b><br>Burden share %{x:.2%}<br>Trial share %{customdata[1]:.2%}"
                          "<br>Equity ratio %{customdata[2]:.2f}<extra></extra>"))
    span_low, span_high = np.log10(low), np.log10(high)
    for row in plotted.nlargest(6, "burden_share").itertuples():
        # Labels on the right half of the chart sit to the left of their circle.
        right_half = np.log10(row.burden_share) > span_low + 0.6 * (span_high - span_low)
        scatter.add_annotation(x=np.log10(row.burden_share), y=np.log10(row.y), text=row.country_name, showarrow=False,
                               xanchor="right" if right_half else "left",
                               xshift=-(row.size / 2 + 4) if right_half else row.size / 2 + 4,
                               font=dict(family=MONO, size=10, color=INK))
    ticks = [v for v in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1) if low <= v <= high]
    scatter.update_xaxes(type="log", title="Share of burden", tickvals=ticks, ticktext=[_pct_tick(v) for v in ticks],
                         range=[span_low, span_high])
    y_ticks = [v for v in ticks if v > floor * 2]
    scatter.update_yaxes(type="log", title="Share of trials", tickvals=[floor] + y_ticks,
                         ticktext=["none"] + [_pct_tick(v) for v in y_ticks], range=[span_low, span_high])
    style_figure(scatter, 420)

    # Fig. 4: countries per equity band, ordinal colours matching the map.
    band_labels = ["Critical <0.5", "Under 0.5–0.8", "Aligned 0.8–1.25", "Above 1.25–2", "High >2"]
    bands = pd.cut(selected["equity_ratio"].fillna(0), [-np.inf, .5, .8, 1.25, 2, np.inf], labels=band_labels, right=False)
    distribution = bands.value_counts(sort=False).reindex(band_labels).fillna(0)
    dist_fig = _base_figure()
    dist_fig.add_trace(go.Bar(x=band_labels, y=distribution.to_numpy(), marker=dict(color=BAND_COLOURS),
                              text=[f"{int(v)}" for v in distribution], textposition="outside", cliponaxis=False,
                              textfont=dict(family=SERIF, size=18, color=INK),
                              hovertemplate="%{x}<br>%{y} countries<extra></extra>"))
    dist_fig.update_yaxes(showticklabels=False, showgrid=False, title=None)
    style_figure(dist_fig, 420).update_yaxes(showticklabels=False, showgrid=False)

    # Fig. 5: forecast.
    forecast = forecast_trial_sites(_read_annual(), cause_code, FORECAST_HORIZON, FORECAST_LOOKBACK)
    forecast_fig = _base_figure()
    readout = "Insufficient yearly history for a governed forecast."
    if not forecast.empty:
        actual = forecast[forecast["kind"] == "Actual"]
        future = forecast[forecast["kind"] == "Forecast"]
        bridge = pd.concat([actual.tail(1), future])
        forecast_fig.add_trace(go.Scatter(x=bridge["year"], y=bridge["upper"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        forecast_fig.add_trace(go.Scatter(x=bridge["year"], y=bridge["lower"], mode="lines", fill="tonexty", fillcolor=PRUSSIAN_SOFT,
                                          line=dict(width=0), name="80% interval", hoverinfo="skip"))
        forecast_fig.add_trace(go.Scatter(x=actual["year"], y=actual["estimate"], mode="lines+markers", name="Observed",
                                          line=dict(color=INK, width=2), marker=dict(size=7, color=INK),
                                          hovertemplate="%{x}: %{y:,.0f} placements<extra>Observed</extra>"))
        forecast_fig.add_trace(go.Scatter(x=bridge["year"], y=bridge["estimate"], mode="lines+markers", name="Projected",
                                          line=dict(color=PRUSSIAN, width=2, dash="dash"), marker=dict(size=8, color=PRUSSIAN),
                                          hovertemplate="%{x}: %{y:,.0f} placements<extra>Projected</extra>"))
        forecast_fig.update_xaxes(title=None, dtick=1).update_yaxes(title="Placements per year", rangemode="tozero")
        style_figure(forecast_fig, 440)
        start_value, end_value = actual["estimate"].iloc[-1], future["estimate"].iloc[-1]
        change = (end_value / start_value - 1) if start_value else np.nan
        readout = [
            html.Div(f"{end_value:,.0f}", className="readout-big"),
            html.P(f"placements projected for {int(future['year'].iloc[-1])}"),
            html.Dl([
                html.Dt("Change from last complete year"), html.Dd(f"{change:+.1%}"),
                html.Dt("80% empirical interval"), html.Dd(f"{future['lower'].iloc[-1]:,.0f} to {future['upper'].iloc[-1]:,.0f}"),
                html.Dt("Selected model"), html.Dd(forecast.attrs["model"]["selected_model"]),
                html.Dt("Rolling WAPE"), html.Dd(f"{min(forecast.attrs['model']['loglinear_backtest_wape'], forecast.attrs['model']['naive_backtest_wape']):.1%}"),
            ]),
        ]

    # Figs. 6 and 7: scenario.
    scenario = allocate_sites(selected, int(extra_sites or 0), strategy)
    sim = simulate_allocation(selected, int(extra_sites or 0), strategy, SIMULATION_DRAWS)
    scenario_align = alignment_score(scenario["trial_share_after"], scenario["burden_share"])
    scenario_theil = theil_index(scenario["trial_share_after"], scenario["burden_share"])
    recipients = int((scenario["allocated_sites"] > 0).sum())
    allocations = scenario[scenario["allocated_sites"] > 0].nlargest(15, "allocated_sites").sort_values("allocated_sites")
    allocation_fig = _base_figure()
    allocation_fig.add_trace(go.Bar(
        x=allocations["allocated_sites"], y=allocations["country_name"], orientation="h", marker=dict(color=PRUSSIAN),
        text=[f"{v:,.0f}" for v in allocations["allocated_sites"]], textposition="outside", cliponaxis=False,
        textfont=dict(family=MONO, size=10, color=INK2), customdata=allocations[["equity_ratio", "equity_ratio_after"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>+%{x} placements<br>Equity ratio %{customdata[0]:.2f} → %{customdata[1]:.2f}<extra></extra>"))
    style_figure(allocation_fig, 440).update_xaxes(showticklabels=False, ticks="", showline=False).update_yaxes(showgrid=False)
    # Fig. 7 uses the map's log scale, so every country fits and halving reads
    # the same distance as doubling. Ratios below 1/16 (including none) sit on
    # the axis floor.
    er_floor = 1 / 16
    compare = scenario.nlargest(20, "burden_share").sort_values("burden_share").assign(
        before=lambda d: d["equity_ratio"].fillna(0).clip(lower=er_floor),
        after=lambda d: d["equity_ratio_after"].fillna(0).clip(lower=er_floor))
    compare_fig = _base_figure()
    for row in compare.itertuples():
        compare_fig.add_shape(type="line", x0=row.before, x1=row.after, y0=row.country_name, y1=row.country_name,
                              line=dict(color=RULE, width=2), layer="below")
    compare_fig.add_trace(go.Scatter(x=compare["before"], y=compare["country_name"], mode="markers", name="Today",
                                     marker=dict(color=FAINT, size=8), customdata=compare["equity_ratio"].fillna(0),
                                     hovertemplate="%{y}: %{customdata:.2f}<extra>Today</extra>"))
    compare_fig.add_trace(go.Scatter(x=compare["after"], y=compare["country_name"], mode="markers", name="After scenario",
                                     marker=dict(color=PRUSSIAN, size=10, line=dict(color=PAPER, width=1.5)),
                                     customdata=compare["equity_ratio_after"].fillna(0),
                                     hovertemplate="%{y}: %{customdata:.2f}<extra>After scenario</extra>"))
    compare_fig.add_vline(x=1, line=dict(color=INK, width=1, dash="dot"))
    er_top = max(2.0, float(compare[["before", "after"]].max().max()) * 1.3)
    er_ticks = [v for v in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32) if v <= er_top]
    compare_fig.update_xaxes(type="log", title="Equity ratio (log scale)", range=[np.log10(er_floor * 0.8), np.log10(er_top)],
                             tickvals=er_ticks, ticktext=["≈0" if v == er_floor else f"{v:g}" for v in er_ticks])
    style_figure(compare_fig, 440).update_yaxes(showgrid=False)

    zero_site = int((selected["trial_sites"] == 0).sum())
    quality = [
        html.Div([html.B(f"{all_countries}/{len(scope)}"), html.Span("Countries with a burden estimate")]),
        html.Div([html.B(f"{covered}/{all_countries}"), html.Span("Countries with any placement")]),
        html.Div([html.B(str(zero_site)), html.Span("Countries with none")]),
        html.Div([html.B("Weekdays"), html.Span("ClinicalTrials.gov refresh cadence")]),
    ]
    status = read_status()
    completed = status.get("completed_at")
    ct_timestamp = status.get("ctgov_data_timestamp")
    gold_time = datetime.fromtimestamp(GOLD_PATH.stat().st_mtime, tz=timezone.utc)
    shown_time = completed or gold_time.isoformat().replace("+00:00","Z")
    freshness_hours = (datetime.now(timezone.utc) - gold_time).total_seconds() / 3600
    badge = "Live snapshot" if completed and freshness_hours < FRESHNESS_WARNING_HOURS else "Repository snapshot"
    badge_class = "freshness-badge live" if badge == "Live snapshot" else "freshness-badge"
    provenance = [
        html.Div([html.B("Clinical trial activity"), html.Span("ClinicalTrials.gov API v2"), html.Small(f"Upstream dataset: {ct_timestamp or 'not recorded'}")]),
        html.Div([html.B("Disease burden"), html.Span("WHO Global Health Estimates 2023, DALYs by country"), html.Small(f"Absolute DALYs per year; {_years_label(start, end)} uses estimate year(s) {selected['burden_years'].iloc[0]}")]),
        html.Div([html.B("Costs and infrastructure"), html.Span("World Bank ICP 2021 health price levels; World Development Indicators"), html.Small("Used by the resource planner; no values are imputed")]),
        html.Div([html.B("Snapshot integrity"), html.Span(status.get("message", "Bundled repository snapshot")), html.Small(f"Dashboard file: {shown_time}")]),
        html.Div([html.B("Metric definition"), html.Span("ER = study-country activity share / burden indicator share"), html.Small("Alignment score = 1 minus total variation distance")]),
        html.Div([html.B("Activity unit"), html.Span("Unique NCT study-country placement"), html.Small("Multiple facilities in one country are counted once in the current historical extract")]),
    ]
    table = scenario[["country_iso3","trial_sites","burden_dalys","burden_rate_per_100k","trial_share","burden_share","equity_ratio","sites_to_parity"]].copy()
    table["burden_dalys"] = table["burden_dalys"].round(0)
    table["burden_rate_per_100k"] = table["burden_rate_per_100k"].round(1)
    table.columns = ["Country","Sites","DALYs per year","DALYs per 100k","Trial share","Burden share","Equity ratio","Sites to parity"]
    table["Trial share"] = table["Trial share"].map(lambda x: f"{x:.2%}")
    table["Burden share"] = table["Burden share"].map(lambda x: f"{x:.2%}")
    table["Equity ratio"] = table["Equity ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "Not available")
    table = table.sort_values("Sites to parity", ascending=False)
    columns = [{"name": column, "id": column} for column in table.columns]

    sentence_name = SENTENCE_NAMES.get(cause_code, DISEASE_PATTERNS[cause_code][0])
    if total_sites > 0:
        widest = selected.assign(gap=selected["burden_share"] - selected["trial_share"]).nlargest(1, "gap").iloc[0]
        headline = (f"{underserved_burden:.0%} of the world's {sentence_name} burden falls in countries that receive "
                    f"less than four-fifths of their fair share of trials.")
        dek = (f"{_years_label(start, end)}: {total_sites:,.0f} trial placements reached {covered} of {all_countries} countries. "
               f"Alignment with burden stands at {align:.0f} out of 100. The widest gap is {widest['country_name']}, "
               f"with {widest['burden_share']:.1%} of the burden and {widest['trial_share']:.1%} of the trials. "
               f"Burden uses WHO estimates for {selected['burden_years'].iloc[0]}.")
    else:
        headline = f"No {sentence_name} trials started in {_years_label(start, end)}."
        dek = "Widen the range of years to compare research activity with burden."

    return (
        f"{total_sites:,.0f}", "Study–country pairs",
        f"{covered}", f"of {all_countries} with a burden estimate",
        f"{align:.0f}", "of 100; 100 is proportional",
        f"{underserved_burden:.0%}", "in countries below 0.8× fair share",
        f"{theil:.2f}", "0 is proportional",
        map_fig, priority_fig, scatter, dist_fig, forecast_fig, readout,
        f"{scenario_align:.1f}", f"Assumption sensitivity P10 to P90: {sim.alignment_p10:.1f} to {sim.alignment_p90:.1f}",
        f"{scenario_theil:.3f}", f"Assumption sensitivity P10 to P90: {sim.theil_p10:.3f} to {sim.theil_p90:.3f}",
        str(recipients), allocation_fig, compare_fig, quality, provenance,
        table.to_dict("records"), columns, badge, badge_class, f"Updated {gold_time:%d %b %Y, %H:%M} UTC",
        headline, dek,
    )


@callback(
    Output("plan-alignment", "children"), Output("plan-alignment-detail", "children"),
    Output("plan-sites", "children"), Output("plan-sites-detail", "children"),
    Output("plan-spent", "children"), Output("plan-spent-detail", "children"),
    Output("plan-saving", "children"), Output("plan-saving-detail", "children"),
    Output("plan-frontier", "figure"), Output("plan-countries", "figure"), Output("plan-map", "figure"),
    Output("plan-sensitivity", "children"), Output("plan-table", "data"), Output("plan-table", "columns"),
    Output("plan-excluded", "children"), Output("plan-method", "children"),
    Input("cause", "value"), Input("years", "value"), Input("plan-budget", "value"),
    Input("plan-study-cost", "value"), Input("plan-site-cost", "value"), Input("plan-reference", "value"),
    Input("plan-scaled", "value"), Input("plan-capacity", "value"), Input("snapshot-token", "data"),
)
def render_planner(cause_code, years, budget, study_cost, site_cost, reference, scaled, capacity_choice, token):
    start, end = int(years[0]), int(years[1])
    span = end - start + 1
    budget = max(float(budget or 0), 0.0)
    study_cost = max(float(study_cost or 0), 0.0)
    site_cost = max(float(site_cost or 0), 0.0)
    percentile = CAPACITY_PERCENTILES.get(capacity_choice, 0.5)
    is_scaled = "scaled" in (scaled or [])
    portfolio, capacity, _, hubs, scale = _plan_inputs(token, cause_code, start, end, reference, percentile, is_scaled)
    smart, naive, smart_curve, naive_curve, sensitivity = _plans(
        token, cause_code, start, end, budget, study_cost, site_cost, reference, percentile, is_scaled)
    resources = _resources(token).set_index("country_iso3")

    # ---- headline numbers
    facilities_used = assign_to_facilities(capacity, smart.countries)
    n_facilities = int(len(facilities_used))
    gain, naive_gain = smart.alignment_after - smart.alignment_before, naive.alignment_after - naive.alignment_before
    cost_same = _cost_to_reach(smart_curve, naive.alignment_after) if naive.placements else np.nan
    saving = naive.spent - cost_same if np.isfinite(cost_same) else np.nan
    saving_text = _money(saving) if np.isfinite(saving) and saving > 0 else "None"
    saving_detail = (f"Reaches the one-site-per-placement result ({naive.alignment_after:.1f}) for {_money(cost_same)}, "
                     f"{saving / naive.spent:.0%} less" if np.isfinite(saving) and saving > 0 and naive.spent
                     else "No cheaper route to the same alignment in this scope")

    # ---- frontier
    frontier = _base_figure()
    for curve, name, colour, dash in ((naive_curve, "One new site per placement", FAINT, "dash"),
                                      (smart_curve, "Registry-proxy plan", PRUSSIAN, "solid")):
        x = np.concatenate([[0], curve["cumulative_cost"].to_numpy()])
        y = np.concatenate([[smart.alignment_before], curve["alignment"].to_numpy()])
        frontier.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=colour, width=2, dash=dash),
                                      hovertemplate="Spend %{x:$,.0f}<br>Alignment %{y:.1f}<extra>" + name + "</extra>"))
    frontier.add_vline(x=budget, line=dict(color=INK, width=1, dash="dot"),
                       annotation_text="budget", annotation_position="top left", annotation_font=dict(family=MONO, size=10, color=INK2))
    frontier.update_xaxes(title="Cumulative spend (US$)", tickprefix="$", rangemode="tozero")
    frontier.update_yaxes(title="Alignment score (0 to 100)")
    frontier.update_layout(hovermode="x unified")
    style_figure(frontier, 440)

    # ---- country bars
    plan = smart.countries.merge(hubs[["new_site_capacity", "capacity_basis"]], left_on="country_iso3", right_index=True, how="left")
    plan = plan.merge(scale[["cost_scale", "cost_basis"]], left_on="country_iso3", right_index=True, how="left")
    plan["country_name"] = plan["country_iso3"].map(resources["country_name"]).fillna(plan["country_iso3"])
    recipients = plan[plan["placements"] > 0].sort_values("placements", ascending=False)
    top = recipients.head(15).iloc[::-1]
    bars = _base_figure()
    hover = ("<b>%{customdata[0]}</b><br>%{x} placements %{meta}<br>Equity ratio %{customdata[1]:.2f} → %{customdata[2]:.2f}"
             "<br>New sites %{customdata[3]}<br>Cost %{customdata[4]:$,.0f}<extra></extra>")
    custom = top[["country_name", "equity_ratio_before", "equity_ratio_after", "new_sites", "cost"]].to_numpy()
    bars.add_trace(go.Bar(y=top["country_name"], x=top["placements_existing"], orientation="h", name="Against historical headroom",
                          marker=dict(color=PRUSSIAN, line=dict(color=PAPER, width=2)), customdata=custom,
                          meta="against historical headroom", hovertemplate=hover))
    bars.add_trace(go.Bar(y=top["country_name"], x=top["placements_new_sites"], orientation="h", name="At new sites",
                          marker=dict(color=VERMILION, line=dict(color=PAPER, width=2)), customdata=custom,
                          meta="at new sites", hovertemplate=hover,
                          text=[f"{n} new" if n else "" for n in top["new_sites"]], textposition="outside",
                          textfont=dict(family=MONO, color=INK2, size=10), cliponaxis=False))
    bars.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
    bars.update_xaxes(title="Additional study placements").update_yaxes(title=None, showgrid=False)
    style_figure(bars, 440)

    # ---- map of sites
    cities = _cities(token, start, end)
    best_city = cities.groupby("country_iso3").head(3).groupby("country_iso3").agg(
        candidate_cities=("city", lambda c: ", ".join(c)), lat=("lat", "first"), lon=("lon", "first"))
    opened = plan[plan["new_sites"] > 0].set_index("country_iso3")
    opened = opened.join(best_city, how="left")
    no_city = opened["lat"].isna()
    opened.loc[no_city, "lat"] = resources["capital_lat"].reindex(opened.index[no_city])
    opened.loc[no_city, "lon"] = resources["capital_lon"].reindex(opened.index[no_city])
    opened["location_note"] = np.where(no_city, "No recorded research facility; capital shown for reference",
                                       "Most research-active city: " + opened["candidate_cities"].fillna(""))
    site_map = go.Figure()
    if n_facilities:
        site_map.add_trace(go.Scattergeo(
            lon=facilities_used["lon"], lat=facilities_used["lat"], mode="markers", name="Facility with historical activity headroom",
            marker=dict(color=PRUSSIAN, size=np.clip(6 + 2.5 * facilities_used["assigned"], 8, 26), opacity=0.75, line=dict(color=PAPER, width=1)),
            customdata=facilities_used[["facility", "city", "assigned", "capacity", "used"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>+%{customdata[2]} studies"
                          f"<br>Historical peak %{{customdata[3]}} listings in {span} years, %{{customdata[4]}} in {_years_label(start, end)}<extra></extra>"))
    if len(opened):
        site_map.add_trace(go.Scattergeo(
            lon=opened["lon"], lat=opened["lat"], mode="markers", name="New sites (country total)",
            marker=dict(color=VERMILION, symbol="diamond", size=np.clip(8 + 2 * opened["new_sites"], 10, 30), line=dict(color=PAPER, width=1)),
            customdata=np.column_stack([opened["country_name"], opened["new_sites"], opened["new_site_capacity"], opened["location_note"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} new sites, up to %{customdata[2]} studies each"
                          "<br>%{customdata[3]}<extra></extra>"))
    style_geo(site_map, 460).update_geos(showcountries=True, countrycolor=PAPER, countrywidth=0.6)
    site_map.update_layout(legend=dict(y=1.02))

    # ---- sensitivity to the new-site cost
    rows = [html.Div([html.B("New-site cost"), html.B("Alignment"), html.B("New sites"), html.B("Historical headroom")], className="sens-row sens-head")]
    for factor, result in sensitivity:
        rows.append(html.Div([
            html.Span(f"{factor:g}× ({_money(site_cost * factor)})"), html.Span(f"{result.alignment_after:.1f}"),
            html.Span(f"{result.new_sites:,}"), html.Span(f"{result.from_existing:,} of {result.placements:,}"),
        ], className="sens-row" + (" sens-current" if factor == 1 else "")))
    rows.append(html.Small("Same budget and study cost. If the plan barely changes across rows, it does not depend much on the new-site estimate."))

    # ---- table
    plan = plan.join(best_city["candidate_cities"], on="country_iso3")
    for column in ("physicians_per_1000", "hospital_beds_per_1000", "electricity_access_pct",
                   "health_spend_per_capita_usd", "logistics_performance_index", "researchers_per_million"):
        plan[column] = plan["country_iso3"].map(resources[column])
    table = recipients[["country_iso3"]].merge(plan, on="country_iso3", how="left")
    table["candidate_cities"] = np.where(table["new_sites"] > 0, table["candidate_cities"].fillna("No recorded research facility"), "")
    out = pd.DataFrame({label: table[column] for label, column in PLAN_TABLE_COLUMNS})
    out["Burden share"] = out["Burden share"].map(lambda v: f"{v:.2%}")
    for label in ("Equity ratio now", "Equity ratio after", "Cost index vs reference"):
        out[label] = out[label].map(lambda v: f"{v:.2f}" if pd.notna(v) else "Not available")
    out["Cost (US$)"] = out["Cost (US$)"].round(0)
    money_columns = {"Cost (US$)", "Health spend per capita (US$)"}
    for label in ("Physicians per 1,000", "Hospital beds per 1,000", "Electricity access %",
                  "Health spend per capita (US$)", "Logistics index (1 to 5)", "Researchers per million"):
        out[label] = out[label].map(lambda v: round(float(v), 2) if pd.notna(v) else "Not reported")
    columns = [{"name": column, "id": column, "type": "numeric", "format": Format(group=Group.yes)}
               if column in money_columns else {"name": column, "id": column} for column in out.columns]

    excluded_share = portfolio.set_index("country_iso3").loc[smart.excluded, "burden_share"].sum() if smart.excluded else 0.0
    excluded_names = ", ".join(str(resources["country_name"].get(iso3, iso3)) for iso3 in smart.excluded)
    excluded = (html.Small(f"Not costed, so not allocated: {excluded_names}. "
                           f"They hold {excluded_share:.1%} of burden and have no ICP 2021 price level.")
                if smart.excluded else html.Small("Every burden-observed country has a measured price level."))
    listings = _facilities(token)
    listings = listings[listings["cause_code"] == cause_code]
    named_share = 1 - listings["anonymised"].mean() if len(listings) else np.nan
    method = [html.B("Method and limits. "), html.Span(
        f" Historical activity proxy: {int((capacity['headroom'] > 0).sum()):,} named facilities are below their prior activity ceiling for this disease "
        f"(their peak studies in any {span}-year window up to {end}, minus studies already in {_years_label(start, end)}). "
        f"{named_share:.0%} of this disease's site listings name a facility; anonymised sponsor listings cannot be tracked and are excluded. "
        f"The new-site throughput assumption is the {capacity_choice.lower()} of comparable facilities' historical listing activity. "
        f"Costs: your inputs × ICP 2021 health price level relative to the reference country. "
        f"The deterministic heuristic greedily selects the available action with the largest alignment gain per dollar; "
        f"it stops at the budget ({smart.stopped_by.lower()}) and does not guarantee a global optimum. "
        "It does not model recruitment rates, eligible participants, regulatory timelines, competing trials or sponsor strategy. "
        "Use protocol-specific, finance-approved costs before operational decisions.")]

    return (
        f"{smart.alignment_after:.1f}", f"From {smart.alignment_before:.1f} now (+{gain:.1f}); one site per placement reaches {naive.alignment_after:.1f} (+{naive_gain:.1f})",
        f"{n_facilities + smart.new_sites:,}", f"{n_facilities:,} existing facilities + {smart.new_sites:,} new sites, versus {naive.new_sites:,} new sites if each placement opened one",
        _money(smart.spent), f"{smart.placements:,} placements; {smart.from_existing:,} use historical activity headroom",
        saving_text, saving_detail,
        frontier, bars, site_map, rows, out.to_dict("records"), columns, excluded, method,
    )


def _fmt_share(value: float, math: bool = False) -> str:
    return f"{100 * value:.2f}" + ("\\%" if math else "%")


def _num(value: float, decimals: int = 0) -> str:
    """A number for use inside MathJax: braces keep the thousands comma tight."""
    return f"{value:,.{decimals}f}".replace(",", "{,}")


@callback(
    Output("methods-example", "children"),
    Input("cause", "value"), Input("years", "value"), Input("snapshot-token", "data"),
)
def render_methods_example(cause_code: str, years: list[int], token: str):
    start, end = int(years[0]), int(years[1])
    _, selected = _scope(token, cause_code, start, end)
    cause_name = DISEASE_PATTERNS[cause_code][0]
    n, total = len(selected), selected["trial_sites"].sum()
    p_share, b_share = selected["trial_share"].to_numpy(), selected["burden_share"].to_numpy()
    tvd = 0.5 * np.abs(p_share - b_share).sum()
    theil = theil_index(p_share, b_share)
    under = selected.loc[selected["equity_ratio"].fillna(0) < 0.8, "burden_share"].sum()
    selected = selected.assign(parity=sites_to_parity(selected["trial_sites"], selected["burden_share"]))
    top = selected.sort_values("burden_share", ascending=False).head(6)
    rows = "\n".join(
        f"| {row.country_iso3} | {row.trial_sites:,.0f} | {_fmt_share(row.trial_share)} | {_fmt_share(row.burden_share)} "
        f"| {row.equity_ratio:.2f} | {row.parity:,.1f} |"
        for row in top.itertuples()
    )
    lead = top.iloc[0]

    text = [f"""
**Scope.** {cause_name}, {_years_label(start, end)}. The cohort has $N = {n}$ countries with measured burden and
$S = {_num(total)}$ study–country placements. Burden uses WHO estimate year(s) {selected['burden_years'].iloc[0]}.

**Largest burden countries.**

| Country | $s_c$ | $p_c$ | $b_c$ | $\\text{{ER}}_c$ | $x_c$ (parity need) |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

For {lead.country_iso3}: $\\text{{ER}} = {lead.trial_share:.4f} / {lead.burden_share:.4f} = {lead.equity_ratio:.2f}$, and
$x = \\max\\big(({lead.burden_share:.4f} \\times {_num(total)} - {_num(lead.trial_sites)}) / (1 - {lead.burden_share:.4f}),\\, 0\\big) = {_num(lead.parity, 1)}$
additional placements for {lead.country_iso3} alone.

**Distribution measures.**

$$\\tfrac{{1}}{{2}}\\sum_c |p_c - b_c| = {tvd:.4f} \\quad\\Rightarrow\\quad A = 100\\,(1 - {tvd:.4f}) = {100 * (1 - tvd):.1f}$$

$$T = \\sum_c p_c \\ln(p_c / b_c) = {theil:.4f}, \\qquad U = {_fmt_share(under, math=True)} \\text{{ of burden in countries with ER}} < 0.8$$
"""]

    forecast = forecast_trial_sites(_read_annual(), cause_code, FORECAST_HORIZON, FORECAST_LOOKBACK)
    model = forecast.attrs.get("model")
    if model:
        future = forecast[forecast["kind"] == "Forecast"].iloc[-1]
        slope = model["slope"]
        sign = "+" if slope >= 0 else "-"
        r_squared = model["r_squared"]
        if r_squared >= 0.6:
            fit_note = "The trend explains most of the year-to-year variation, so its direction is well supported."
        elif r_squared >= 0.3:
            fit_note = "The trend explains part of the year-to-year variation; read its direction as moderate evidence."
        else:
            fit_note = "The trend explains little of the year-to-year variation, so treat its direction as weak evidence."
        selected = model["selected_model"]
        fit_note += (f" Rolling-origin validation selected the **{selected}**: log-linear WAPE "
                     f"{model['loglinear_backtest_wape']:.1%}, last-value WAPE {model['naive_backtest_wape']:.1%}. "
                     f"The interval uses {model['interval_method']}.")
        text.append(f"""
**Forecast fit** (global interventional study-country activity, {model['first_year']} to {model['last_year']}, $n = {model['n_years']}$ complete years).

$$\\ln(1 + Y_t) = {model['intercept']:.3f} {sign} {abs(slope):.4f}\\,(t - {model['first_year']}) + \\varepsilon_t$$

The implied annual growth is $e^{{{slope:.4f}}} - 1 = {100 * (np.exp(slope) - 1):+.1f}\\%$,
with residual standard deviation {model['residual_sd']:.3f} on the log scale and $R^2 = {model['r_squared']:.2f}$.
{fit_note}
The governed {selected} forecast for {int(future['year'])} is {future['estimate']:,.0f}
(80% interval {future['lower']:,.0f} to {future['upper']:,.0f}).
""")
    else:
        text.append("**Forecast fit.** Fewer than four complete years of history, so no model is fitted.")

    portfolio, capacity, headroom, hubs, scale = _plan_inputs(
        token, cause_code, start, end, PLANNER.get("reference_country", "USA"), 0.5, True)
    with_room = int((capacity["headroom"] > 0).sum()) if len(capacity) else 0
    basis = hubs["capacity_basis"].str.split(" ").str[0].value_counts()
    uncosted = int(scale["cost_scale"].isna().sum())
    text.append(f"""
**Planner inputs** (median assumed new-site throughput, {PLANNER.get('reference_country', 'USA')} prices).
{len(capacity):,} named facilities have listed {cause_name} studies up to {end}; {with_room:,} are active below their historical activity ceiling,
totalling {int(headroom.sum()):,} placements of activity headroom ($\\sum_f H_f$). The new-site proxy is measured at country level for {int(basis.get('Country', 0))}
countries, income-group level for {int(basis.get('Income', 0))} and global level for {int(basis.get('All', 0))}.
{uncosted} cohort countries lack an ICP price level and cannot receive placements.
""")
    return display_math("\n".join(text))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("DASH_DEBUG", "0") == "1")
