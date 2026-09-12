"""Global Clinical Trial Equity Intelligence dashboard.

Run locally with ``python app.py``. The application reads atomic parquet
snapshots produced by the Python pipeline and can refresh one disease directly
from ClinicalTrials.gov.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html, no_update
import numpy as np
import pandas as pd
import plotly.express as px
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
from py.live import read_status, refresh


ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "data" / "gold" / "country_interval_disease.parquet"
CT_PATH = ROOT / "data" / "clean" / "ctgov_sites.parquet"
CFG = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
POLL_MS = int(CFG.get("live", {}).get("dashboard_poll_seconds", 300)) * 1000
FRESHNESS_WARNING_HOURS = int(CFG.get("live", {}).get("freshness_warning_hours", 36))
FORECAST_HORIZON = int(CFG.get("live", {}).get("forecast_horizon_years", 3))
FORECAST_LOOKBACK = int(CFG.get("live", {}).get("forecast_lookback_years", 8))
SIMULATION_DRAWS = int(CFG.get("live", {}).get("simulation_draws", 500))
CURRENT_YEAR = datetime.now(timezone.utc).year

INK = "#10233F"
NAVY = "#0B1930"
BLUE = "#2F6BFF"
CYAN = "#36C5F0"
RED = "#E25555"
GREEN = "#20B486"
MUTED = "#6F7C91"
GRID = "#E8EDF5"
BG = "#F4F7FB"


def _read_gold() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        raise FileNotFoundError("Run `python -m py.indices` to create the gold dataset.")
    frame = pd.read_parquet(GOLD_PATH)
    numeric = [
        "trial_sites", "burden_dalys", "trial_share", "burden_share",
        "equity_ratio", "trial_density_per_k_dalys", "gini_er_weighted", "theil_T",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trial_sites"] = frame["trial_sites"].fillna(0)
    frame["period_key"] = frame["interval_start"].astype(str) + "-" + frame["interval_end"].astype(str)
    frame["period_display"] = frame.apply(
        lambda row: (
            f"{int(row.interval_start)} to {int(row.interval_end)} (to date)"
            if int(row.interval_start) <= CURRENT_YEAR <= int(row.interval_end)
            else f"{int(row.interval_start)} to {int(row.interval_end)}"
        ), axis=1,
    )
    return frame


def _read_annual() -> pd.DataFrame:
    if not CT_PATH.exists():
        return pd.DataFrame(columns=["year", "cause_code", "trial_sites"])
    frame = pd.read_parquet(CT_PATH)
    frame["trial_sites"] = pd.to_numeric(frame["trial_sites"], errors="coerce").fillna(0)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    return frame.dropna(subset=["year"])


def _snapshot_token() -> str:
    stamps = [path.stat().st_mtime_ns for path in (GOLD_PATH, CT_PATH) if path.exists()]
    return str(max(stamps, default=0))


def _available_periods(frame: pd.DataFrame, cause_code: str) -> pd.DataFrame:
    scoped = frame[frame["cause_code"] == cause_code]
    active = scoped.groupby("period_key")["trial_sites"].sum()
    active_keys = active[active > 0].index
    periods = scoped[scoped["period_key"].isin(active_keys)][
        ["period_key", "period_display", "interval_start"]
    ].drop_duplicates().sort_values("interval_start")
    if periods.empty:  # Preserve a diagnosable empty period instead of breaking UI.
        periods = scoped[["period_key", "period_display", "interval_start"]].drop_duplicates().sort_values("interval_start").tail(1)
    return periods


def _base_figure(title: str | None = None) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title)
    return style_figure(figure)


def style_figure(figure: go.Figure, height: int = 390) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=28, r=22, t=42, b=32),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color=INK, size=12),
        hoverlabel=dict(bgcolor=NAVY, font_color="white", bordercolor=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False)
    figure.update_yaxes(gridcolor=GRID, zeroline=False)
    return figure


def kpi_card(label: str, value_id: str, detail_id: str, accent: str) -> html.Div:
    return html.Div(
        [
            html.Div(className="kpi-accent", style={"backgroundColor": accent}),
            html.Div(label, className="kpi-label"),
            html.Div(id=value_id, className="kpi-value"),
            html.Div(id=detail_id, className="kpi-detail"),
        ],
        className="kpi-card",
    )


gold_initial = _read_gold()
causes = gold_initial[["cause_code", "cause_name"]].drop_duplicates().sort_values("cause_name")
cause_options = [{"label": row.cause_name, "value": row.cause_code} for row in causes.itertuples()]
initial_cause = cause_options[0]["value"]
initial_period_frame = _available_periods(gold_initial, initial_cause)
initial_period_options = [{"label": row.period_display, "value": row.period_key} for row in initial_period_frame.itertuples()]
initial_period = initial_period_options[-1]["value"]

app = Dash(__name__, title="Trial Equity Intelligence", update_title="Updating intelligence…")
server = app.server

app.layout = html.Div(
    [
        dcc.Store(id="snapshot-token", data=_snapshot_token()),
        dcc.Interval(id="snapshot-poll", interval=POLL_MS, n_intervals=0),
        html.Aside(
            [
                html.Div([html.Div("E", className="brand-mark"), html.Div([html.B("EQUITY"), html.Span("INTELLIGENCE")], className="brand-copy")], className="brand"),
                html.Div("ANALYSIS SCOPE", className="sidebar-eyebrow"),
                html.Label("Disease area"),
                dcc.Dropdown(id="cause", options=cause_options, value=initial_cause, clearable=False, className="control"),
                html.Label("Analysis period"),
                dcc.Dropdown(id="period", options=initial_period_options, value=initial_period, clearable=False, className="control"),
                html.Hr(),
                html.Div("LIVE DATA", className="sidebar-eyebrow"),
                html.Button([html.Span("↻"), " Refresh selected disease"], id="refresh-live", n_clicks=0, className="refresh-button"),
                html.Div(id="refresh-message", className="refresh-message"),
                html.Div(
                    [html.Div(className="pulse"), html.Div([html.B("Near-real-time feed"), html.Span("ClinicalTrials.gov publishes weekdays")])],
                    className="live-card",
                ),
                html.Div(className="sidebar-spacer"),
                html.Div("DECISION SUPPORT", className="sidebar-foot-title"),
                html.P("Descriptive analytics and planning scenarios. This is not causal evidence or medical advice."),
            ], className="sidebar",
        ),
        html.Main(
            [
                html.Header(
                    [
                        html.Div([html.Div("GLOBAL EVIDENCE COVERAGE", className="eyebrow"), html.H1("Clinical Trial Equity Intelligence Centre"), html.P("Where research activity diverges from disease burden, and where the next site could have the greatest equity impact.")]),
                        html.Div([html.Div(id="freshness-badge", className="freshness-badge"), html.Div(id="last-updated", className="last-updated")], className="freshness-wrap"),
                    ], className="topbar",
                ),
                html.Section(
                    [
                        kpi_card("TRIAL-COUNTRY ACTIVITY", "kpi-sites", "kpi-sites-detail", BLUE),
                        kpi_card("MARKET COVERAGE", "kpi-countries", "kpi-countries-detail", CYAN),
                        kpi_card("ALIGNMENT SCORE", "kpi-alignment", "kpi-alignment-detail", GREEN),
                        kpi_card("BURDEN UNDERSERVED", "kpi-underserved", "kpi-underserved-detail", RED),
                        kpi_card("THEIL DIVERGENCE", "kpi-theil", "kpi-theil-detail", "#876CFF"),
                    ], className="kpi-grid",
                ),
                dcc.Tabs(
                    id="workspace-tabs", value="overview", className="tabs",
                    children=[
                        dcc.Tab(label="Executive overview", value="overview", children=html.Div([
                            html.Div([html.Div([html.Div([html.H3("Global equity ratio"), html.Span("ER = trial share ÷ burden share", className="chart-meta")], className="chart-head"), dcc.Graph(id="equity-map", config={"displayModeBar": False})], className="panel panel-wide"),
                                      html.Div([html.Div([html.H3("Priority markets"), html.Span("Largest absolute site deficits", className="chart-meta")], className="chart-head"), dcc.Graph(id="priority-bars", config={"displayModeBar": False})], className="panel")], className="grid-2-1"),
                            html.Div([html.Div([html.Div([html.H3("Burden vs research activity"), html.Span("Bubble size = disease burden", className="chart-meta")], className="chart-head"), dcc.Graph(id="alignment-scatter", config={"displayModeBar": False})], className="panel"),
                                      html.Div([html.Div([html.H3("Equity distribution"), html.Span("Country portfolio by ER band", className="chart-meta")], className="chart-head"), dcc.Graph(id="er-distribution", config={"displayModeBar": False})], className="panel")], className="grid-1-1"),
                        ], className="tab-page")),
                        dcc.Tab(label="Forecast", value="forecast", children=html.Div([
                            html.Div([html.Div([html.Div([html.H3("Trial-site outlook"), html.Span(f"{FORECAST_HORIZON}-year residual-bootstrap forecast", className="chart-meta")], className="chart-head"), dcc.Graph(id="forecast-chart", config={"displayModeBar": False})], className="panel panel-wide"),
                                      html.Div([html.H3("Model readout"), html.Div(id="forecast-readout", className="narrative")], className="panel")], className="grid-2-1"),
                            html.Div([html.B("Model governance"), html.Span(" Forecasts extrapolate the recent global log-linear trend and show 80% empirical intervals. They do not predict protocol feasibility, approvals, recruitment, or causal effects.")], className="method-note"),
                        ], className="tab-page")),
                        dcc.Tab(label="Scenario lab", value="scenario", children=html.Div([
                            html.Div([html.Div([html.Label("New sites to allocate"), dcc.Slider(id="extra-sites", min=0, max=1000, step=25, value=250, marks={0:"0",250:"250",500:"500",750:"750",1000:"1,000"})], className="scenario-control"),
                                      html.Div([html.Label("Allocation strategy"), dcc.RadioItems(id="strategy", options=["Equity gap", "Disease burden", "Hybrid"], value="Equity gap", inline=True, className="radio")], className="scenario-control")], className="scenario-controls"),
                            html.Div([html.Div([html.Span("PROJECTED ALIGNMENT"), html.B(id="scenario-alignment"), html.Small(id="scenario-alignment-range")], className="scenario-stat"),
                                      html.Div([html.Span("PROJECTED THEIL"), html.B(id="scenario-theil"), html.Small(id="scenario-theil-range")], className="scenario-stat"),
                                      html.Div([html.Span("COUNTRIES RECEIVING SITES"), html.B(id="scenario-recipients"), html.Small("Integer portfolio allocation")], className="scenario-stat")], className="scenario-stats"),
                            html.Div([html.Div([html.Div([html.H3("Recommended portfolio"), html.Span("Top allocations under selected rule", className="chart-meta")], className="chart-head"), dcc.Graph(id="allocation-bars", config={"displayModeBar": False})], className="panel"),
                                      html.Div([html.Div([html.H3("Before vs scenario"), html.Span("Equity ratio by country", className="chart-meta")], className="chart-head"), dcc.Graph(id="scenario-comparison", config={"displayModeBar": False})], className="panel")], className="grid-1-1"),
                        ], className="tab-page")),
                        dcc.Tab(label="Data and methodology", value="quality", children=html.Div([
                            html.Div([html.Div([html.H3("Coverage and quality"), html.Div(id="quality-summary", className="quality-grid")], className="panel"),
                                      html.Div([html.H3("Source provenance"), html.Div(id="provenance", className="provenance")], className="panel")], className="grid-1-1"),
                            html.Div([html.Div([html.H3("Country-level analytical table"), html.Span("Sort, filter, and export the active cohort", className="chart-meta")], className="chart-head"), dash_table.DataTable(id="country-table", page_size=15, filter_action="native", sort_action="native", export_format="csv", style_table={"overflowX":"auto"}, style_cell={"fontFamily":"Inter, Arial", "fontSize":12, "padding":"11px", "border":"none"}, style_header={"backgroundColor":"#EFF3F9", "fontWeight":700, "color":INK}, style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#F9FBFD"}])], className="panel"),
                        ], className="tab-page")),
                    ],
                ),
            ], className="main",
        ),
    ], className="shell",
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
    Output("period", "options"),
    Output("period", "value"),
    Input("cause", "value"),
    Input("snapshot-token", "data"),
)
def update_periods(cause_code: str, _token: str):
    periods = _available_periods(_read_gold(), cause_code)
    options = [{"label": row.period_display, "value": row.period_key} for row in periods.itertuples()]
    return options, options[-1]["value"] if options else None


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
    Input("cause", "value"), Input("period", "value"), Input("extra-sites", "value"),
    Input("strategy", "value"), Input("snapshot-token", "data"),
)
def render_dashboard(cause_code: str, period_key: str, extra_sites: int, strategy: str, _token: str):
    gold = _read_gold()
    scope = gold[(gold["cause_code"] == cause_code) & (gold["period_key"] == period_key)].copy()
    if scope.empty:
        raise ValueError("No data are available for the selected scope")
    selected = scope[scope["burden_dalys"].fillna(0) > 0].copy()
    selected["trial_share"] = selected["trial_sites"] / selected["trial_sites"].sum()
    selected["burden_share"] = selected["burden_dalys"] / selected["burden_dalys"].sum()
    selected["equity_ratio"] = selected["trial_share"] / selected["burden_share"]
    total_sites = selected["trial_sites"].sum()
    covered = int((selected["trial_sites"] > 0).sum())
    all_countries = len(selected)
    align = alignment_score(selected["trial_share"], selected["burden_share"])
    underserved_burden = selected.loc[selected["equity_ratio"].fillna(0) < 0.8, "burden_share"].sum()
    theil = theil_index(selected["trial_share"], selected["burden_share"])
    parity = sites_to_parity(selected["trial_sites"], selected["burden_share"])
    selected["sites_to_parity"] = np.ceil(parity)
    selected["er_log"] = np.log2(selected["equity_ratio"].clip(lower=0.125, upper=8).fillna(0.125))

    map_fig = px.choropleth(selected, locations="country_iso3", color="er_log", hover_name="country_iso3",
        hover_data={"er_log":False, "equity_ratio":":.2f", "trial_sites":":,.0f", "burden_share":":.2%"},
        color_continuous_scale=[[0,RED],[0.42,"#F4C7A1"],[0.5,"#E9EEF5"],[0.58,"#A7DBCF"],[1,GREEN]], range_color=(-3,3))
    map_fig.update_coloraxes(colorbar=dict(title="Equity ratio", tickvals=[-3,-2,-1,0,1,2,3], ticktext=["≤0.125","0.25","0.5","1","2","4","≥8"], thickness=10))
    map_fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor="white")
    style_figure(map_fig, 430).update_layout(margin=dict(l=0,r=0,t=10,b=0))

    priority = selected.sort_values("sites_to_parity", ascending=False).head(12).sort_values("sites_to_parity")
    priority_fig = px.bar(priority, x="sites_to_parity", y="country_iso3", orientation="h", color="burden_share", color_continuous_scale=["#B9C9FF",BLUE], hover_data={"equity_ratio":":.2f", "burden_share":":.2%"})
    priority_fig.update_layout(coloraxis_showscale=False).update_xaxes(title="Sites needed for individual parity").update_yaxes(title=None)
    style_figure(priority_fig, 430)

    scatter = px.scatter(selected, x="burden_share", y="trial_share", size="burden_dalys", color="equity_ratio", hover_name="country_iso3", color_continuous_scale=[RED,"#E9EEF5",GREEN], range_color=(0,2), hover_data={"equity_ratio":":.2f", "trial_sites":":,.0f"})
    maximum = max(selected["burden_share"].max(), selected["trial_share"].max()) * 1.08
    scatter.add_trace(go.Scatter(x=[0,maximum], y=[0,maximum], mode="lines", line=dict(color=MUTED,dash="dash"), name="Perfect alignment", hoverinfo="skip"))
    scatter.update_xaxes(title="Share of global burden", tickformat=".1%").update_yaxes(title="Share of trial sites", tickformat=".1%")
    scatter.update_layout(coloraxis_showscale=False)
    style_figure(scatter)

    bands = pd.cut(selected["equity_ratio"].fillna(0), [-np.inf,.5,.8,1.25,2,np.inf], labels=["Critical <0.5","Under 0.5 to 0.8","Aligned 0.8 to 1.25","Above 1.25 to 2","High >2"])
    distribution = bands.value_counts(sort=False).rename_axis("band").reset_index(name="countries")
    dist_fig = px.bar(distribution, x="band", y="countries", color="band", color_discrete_sequence=[RED,"#F49B67","#7BC8B4","#79A7FF",BLUE])
    dist_fig.update_layout(showlegend=False).update_xaxes(title=None, tickangle=-18).update_yaxes(title="Countries")
    style_figure(dist_fig)

    forecast = forecast_trial_sites(_read_annual(), cause_code, FORECAST_HORIZON, FORECAST_LOOKBACK)
    forecast_fig = _base_figure()
    readout = "Insufficient yearly history for a governed forecast."
    if not forecast.empty:
        actual = forecast[forecast["kind"] == "Actual"]
        future = forecast[forecast["kind"] == "Forecast"]
        forecast_fig.add_trace(go.Scatter(x=future["year"], y=future["upper"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        forecast_fig.add_trace(go.Scatter(x=future["year"], y=future["lower"], mode="lines", fill="tonexty", fillcolor="rgba(47,107,255,.14)", line=dict(width=0), name="80% interval"))
        forecast_fig.add_trace(go.Scatter(x=actual["year"], y=actual["estimate"], mode="lines+markers", line=dict(color=INK,width=3), name="Actual"))
        forecast_fig.add_trace(go.Scatter(x=future["year"], y=future["estimate"], mode="lines+markers", line=dict(color=BLUE,width=3,dash="dash"), name="Forecast"))
        forecast_fig.update_xaxes(title=None, dtick=1).update_yaxes(title="Registered trial sites")
        style_figure(forecast_fig, 430)
        start_value, end_value = actual["estimate"].iloc[-1], future["estimate"].iloc[-1]
        change = (end_value / start_value - 1) if start_value else np.nan
        readout = [html.Div(f"{end_value:,.0f}", className="forecast-big"), html.P(f"projected annual sites by {int(future['year'].iloc[-1])}"), html.Hr(), html.P(f"Implied change from the last complete year: {change:+.1%}."), html.P(f"80% interval: {future['lower'].iloc[-1]:,.0f} to {future['upper'].iloc[-1]:,.0f} sites."), html.Small(f"Trained on the latest {len(actual)} complete years; fixed seed for reproducibility.")]

    scenario = allocate_sites(selected, int(extra_sites or 0), strategy)
    sim = simulate_allocation(selected, int(extra_sites or 0), strategy, SIMULATION_DRAWS)
    scenario_align = alignment_score(scenario["trial_share_after"], scenario["burden_share"])
    scenario_theil = theil_index(scenario["trial_share_after"], scenario["burden_share"])
    recipients = int((scenario["allocated_sites"] > 0).sum())
    allocations = scenario.sort_values("allocated_sites", ascending=False).head(15).sort_values("allocated_sites")
    allocation_fig = px.bar(allocations, x="allocated_sites", y="country_iso3", orientation="h", color="equity_ratio", color_continuous_scale=[RED,"#E9EEF5",GREEN], range_color=(0,2), hover_data={"trial_sites_after":":,.0f"})
    allocation_fig.update_layout(coloraxis_showscale=False).update_xaxes(title="New sites").update_yaxes(title=None)
    style_figure(allocation_fig)
    compare = scenario.nlargest(20, "burden_share").sort_values("burden_share")
    compare_fig = go.Figure()
    compare_fig.add_trace(go.Scatter(x=compare["equity_ratio"], y=compare["country_iso3"], mode="markers", marker=dict(color=MUTED,size=8), name="Current"))
    compare_fig.add_trace(go.Scatter(x=compare["equity_ratio_after"], y=compare["country_iso3"], mode="markers", marker=dict(color=BLUE,size=10), name="Scenario"))
    for row in compare.itertuples():
        compare_fig.add_shape(type="line", x0=max(row.equity_ratio,0), x1=max(row.equity_ratio_after,0), y0=row.country_iso3, y1=row.country_iso3, line=dict(color="#C9D2E3",width=2))
    compare_fig.add_vline(x=1, line_dash="dash", line_color=GREEN)
    compare_fig.update_xaxes(title="Equity ratio", range=[0, min(3, max(1.5, np.nanpercentile(compare["equity_ratio_after"],95)*1.15))]).update_yaxes(title=None)
    style_figure(compare_fig)

    zero_site = int((selected["trial_sites"] == 0).sum())
    quality = [
        html.Div([html.B(f"{all_countries}/{len(scope)}"), html.Span("Burden records")]),
        html.Div([html.B(f"{covered}/{all_countries}"), html.Span("Countries with sites")]),
        html.Div([html.B(str(zero_site)), html.Span("Zero-site countries")]),
        html.Div([html.B("Daily weekdays"), html.Span("CT.gov source cadence")]),
    ]
    status = read_status()
    completed = status.get("completed_at")
    ct_timestamp = status.get("ctgov_data_timestamp")
    gold_time = datetime.fromtimestamp(GOLD_PATH.stat().st_mtime, tz=timezone.utc)
    shown_time = completed or gold_time.isoformat().replace("+00:00","Z")
    freshness_hours = (datetime.now(timezone.utc) - gold_time).total_seconds() / 3600
    badge = "LIVE SNAPSHOT" if completed and freshness_hours < FRESHNESS_WARNING_HOURS else "REPOSITORY SNAPSHOT"
    badge_class = "freshness-badge live" if badge == "LIVE SNAPSHOT" else "freshness-badge"
    provenance = [
        html.Div([html.B("Clinical trial activity"), html.Span("ClinicalTrials.gov API v2"), html.Small(f"Upstream dataset: {ct_timestamp or 'not recorded'}")]),
        html.Div([html.B("Disease burden"), html.Span("WHO Global Health Observatory"), html.Small("Published estimates; slower release cadence than registrations")]),
        html.Div([html.B("Snapshot integrity"), html.Span(status.get("message", "Bundled repository snapshot")), html.Small(f"Dashboard file: {shown_time}")]),
        html.Div([html.B("Metric definition"), html.Span("ER = study-country activity share / burden indicator share"), html.Small("Alignment score = 1 minus total variation distance")]),
        html.Div([html.B("Activity unit"), html.Span("Unique NCT study-country placement"), html.Small("Multiple facilities in one country are counted once in the current historical extract")]),
    ]
    table = scenario[["country_iso3","trial_sites","burden_dalys","trial_share","burden_share","equity_ratio","sites_to_parity"]].copy()
    table.columns = ["Country","Sites","DALYs / 100k","Trial share","Burden share","Equity ratio","Sites to parity"]
    table["Trial share"] = table["Trial share"].map(lambda x: f"{x:.2%}")
    table["Burden share"] = table["Burden share"].map(lambda x: f"{x:.2%}")
    table["Equity ratio"] = table["Equity ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "Not available")
    table = table.sort_values("Sites to parity", ascending=False)
    columns = [{"name": column, "id": column} for column in table.columns]

    return (
        f"{total_sites:,.0f}", "Unique study-country activity units",
        f"{covered}/{all_countries}", f"{covered/all_countries:.0%} of burden-observed markets",
        f"{align:.1f}", "0 = divergent · 100 = aligned",
        f"{underserved_burden:.1%}", "Burden in countries with ER < 0.8",
        f"{theil:.3f}", "Lower is better · 0 is aligned",
        map_fig, priority_fig, scatter, dist_fig, forecast_fig, readout,
        f"{scenario_align:.1f}", f"Monte Carlo P10 to P90: {sim.alignment_p10:.1f} to {sim.alignment_p90:.1f}",
        f"{scenario_theil:.3f}", f"Monte Carlo P10 to P90: {sim.theil_p10:.3f} to {sim.theil_p90:.3f}",
        str(recipients), allocation_fig, compare_fig, quality, provenance,
        table.to_dict("records"), columns, badge, badge_class, f"Updated {gold_time:%d %b %Y at %H:%M UTC}",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("DASH_DEBUG", "0") == "1")
