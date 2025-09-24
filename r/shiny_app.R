suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(plotly)
  library(DT)
  library(scales)
  library(stringr)
})

# ----------------------------- Paths & load -----------------------------
gold_dir   <- file.path("..", "data", "gold")
gold_main  <- file.path(gold_dir, "country_interval_disease.parquet")
gold_fall  <- file.path(gold_dir, "country_year_disease.parquet")

GOLD_PATH <- if (file.exists(gold_main)) gold_main else gold_fall
if (!file.exists(GOLD_PATH)) {
  stop(paste0(
    "Gold dataset not found. Expected either:\n  ",
    normalizePath(gold_main, winslash = "/"),
    "\n  or\n  ",
    normalizePath(gold_fall, winslash = "/"),
    "\nPlease run the Python pipeline to generate the gold parquet."
  ))
}

df_gold <- read_parquet(GOLD_PATH) |>
  mutate(across(where(is.character), as.character))

# Harmonise schema (interval-based preferred)
is_interval <- all(c("interval_start","interval_end","interval_label") %in% names(df_gold))
if (!is_interval) {
  df_gold <- df_gold |>
    mutate(
      interval_start = pmax(2000L, floor(as.integer(year)/5)*5),
      interval_end   = interval_start + 5L,
      interval_label = paste0(interval_start, "\u2013", interval_end)
    )
}

# Coerce numerics
df_gold <- df_gold |>
  mutate(
    trial_sites  = as.numeric(trial_sites),
    burden_dalys = as.numeric(burden_dalys),
    trial_share  = as.numeric(trial_share),
    burden_share = as.numeric(burden_share),
    equity_ratio = as.numeric(equity_ratio),
    gini_er_weighted = as.numeric(gini_er_weighted),
    theil_T = as.numeric(theil_T)
  ) |>
  # ---- ADDITION: treat missing site counts as zero so filtering works cleanly
  mutate(trial_sites = ifelse(is.na(trial_sites), 0, trial_sites))

# ----------------------------- Helpers -----------------------------
ER_CAP <- 10

poster_palette <- list(
  under_deep = "#2C6FB7",
  under      = "#5C90CA",
  neutral    = "#E5E7EB",
  over       = "#E67E73",
  over_deep  = "#C0392B"
)

er_bin <- function(er) {
  cut(er,
      breaks = c(-Inf, 0.5, 0.8, 1.25, 2.0, Inf),
      labels = c("< 0.5", "0.5–0.8", "0.8–1.25", "1.25–2.0", "> 2.0"),
      right = TRUE)
}

fmt_pct   <- function(x) percent(x, accuracy = 0.01)
fmt_num   <- function(x) comma(x, accuracy = 1)
fmt_ratio <- function(x) ifelse(is.na(x), "—", sprintf("%.2f", x))

weighted_gini <- function(values, weights) {
  v <- as.numeric(values); w <- as.numeric(weights)
  m <- is.finite(v) & is.finite(w) & w >= 0
  v <- v[m]; w <- w[m]
  if (!length(v) || sum(w) <= 0) return(NA_real_)
  mu <- sum(v*w)/sum(w); if (!is.finite(mu) || mu <= 0) return(NA_real_)
  o <- order(v); v <- v[o]; w <- w[o]
  vw <- v*w; c_vw <- cumsum(vw); cw <- cumsum(w); W <- cw[length(cw)]
  s <- sum(w * (c(0, c_vw[-length(c_vw)]) + c_vw))
  1 - s/(mu*W*W)
}

theil_T_fun <- function(p, b) {
  p <- as.numeric(p); b <- as.numeric(b)
  m <- is.finite(p) & is.finite(b) & p >= 0 & b > 0
  p <- p[m]; b <- b[m]
  if (!length(p)) return(NA_real_)
  p <- p/sum(p); b <- b/sum(b)
  sum(ifelse(p > 0, p*log(p/b), 0))
}

policy_allocate <- function(d_group, extra_sites) {
  d <- d_group
  if (!nrow(d) || extra_sites <= 0) {
    return(list(
      after = d |> mutate(alloc_sites = 0, trial_sites_after = trial_sites,
                          trial_share_after = trial_share,
                          equity_ratio_after = equity_ratio),
      theil_before = unique(d$theil_T),
      gini_before  = unique(d$gini_er_weighted),
      theil_after  = unique(d$theil_T),
      gini_after   = unique(d$gini_er_weighted)
    ))
  }
  gap <- pmax(d$burden_share - d$trial_share, 0)
  alloc <- if (sum(gap) > 0) extra_sites * (gap/sum(gap)) else rep(0, nrow(d))
  d_after <- d |>
    mutate(alloc_sites = alloc,
           trial_sites_after = trial_sites + alloc_sites)
  tot_after <- sum(d_after$trial_sites_after)
  d_after <- d_after |>
    mutate(trial_share_after  = ifelse(tot_after > 0, trial_sites_after/tot_after, NA_real_),
           equity_ratio_after = trial_share_after/burden_share)
  theil_after <- theil_T_fun(d_after$trial_share_after, d_after$burden_share)
  gini_after  <- weighted_gini(d_after$equity_ratio_after, d_after$burden_share)
  list(
    after = d_after,
    theil_before = unique(d$theil_T),
    gini_before  = unique(d$gini_er_weighted),
    theil_after  = theil_after,
    gini_after   = gini_after
  )
}

# ----------------------------- UI -----------------------------
periods <- df_gold |>
  distinct(interval_start, interval_end, interval_label) |>
  arrange(interval_start)

causes <- df_gold |>
  distinct(cause_code, cause_name) |>
  arrange(cause_name)

theme <- bs_theme(
  bootswatch = "flatly",
  base_font = font_google("Inter", local = TRUE)
)

ui <- fluidPage(
  theme = theme,
  tags$head(
    tags$script(src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"),
    # ----------------- STYLING ONLY (content untouched) -----------------
    tags$style(HTML(
      "
      body { margin: 10px 12px; background:#FFFFFF; color:#111827; }
      .container-fluid { padding-left: 6px; padding-right: 6px; max-width: 1400px; }

      /* Tile (main page title) — blue */
      h1 {
  margin-top: 12px;
  margin-bottom: 8px;
  color: #FFFFFF;
  background: #0A192F;   /* <- dark navy */
  padding: 10px 12px;
  border-radius: 10px;
}

      /* Section headings (Key Figures, Formulae, Policy Planner, Country table) — red */
      h2{
        margin-top: 5px; margin-bottom: 8px;
        color:#FFFFFF; background:#0A192F; padding:8px 10px; border-radius:8px;
        font-size:15px; font-weight:500;
      }

      /* Figure headings (Figure 1... etc) — red and same size */
      .figure-title{
        font-size:15px; font-weight:500; margin-bottom:6px;
        color:#FFFFFF; background:#B91C1C; padding:8px 10px; border-radius:8px;
      }

      /* Supporting copy for readability on light bg */
      .lead { color:#374151; font-size:15px; margin-top:-6px; margin-bottom:6px; }
      .muted { color:#6b7280; font-size:12px; }
      .tight { margin-top:4px; margin-bottom:6px; }
      .subtle, .figure-sub { color:#374151; }
      .figure-sub { margin-bottom:8px; }

      /* Red separators */
      .hr { border-top: 2px solid #B91C1C; margin: 12px 0 16px 0; }

      .kpi td, .kpi th { text-align:center; vertical-align:middle; }

      /* Make all MathJax equations red (highlight) */
      .mjx-container, .mjx-math { color:#B91C1C !important; }

      /* Make bold numbers in Abstract red for emphasis only in that block */
      #abstract_text b { color:#B91C1C; }

      /* DataTables text color */
      table, .dataTables_wrapper .dataTables_length label,
      .dataTables_wrapper .dataTables_filter label,
      .dataTables_wrapper .dataTables_info { color:#111827; }
      "
    ))
  ),
  
  # Header row
  fluidRow(
    column(
      width = 8,
      h1("Global Equity in Clinical Trial Site Placement: Assessing Alignment Between Research Activity and Disease Burden"),
      div(
        class = "lead",
        "Are clinical trial sites placed where disease burden lies? We investigate this by comparing ",
        "clinical trial site counts from CT.gov with global disease burden measured through WHO DALYs ",
        "(per-100k, cumulative), aggregated into fixed multi-year periods. ",
        strong("Equity Ratio (ER) = trial_share / burden_share"),
        " serves as the central measure of alignment, where values near 1 indicate equity between trial activity ",
        "and disease need. Group-level inequality is further summarised by Theil T and the burden-weighted ",
        "Gini of ER, highlighting global disparities in research investment. This dashboard provides an ",
        "intuitive policy-oriented view to support recruitment planning, trial placement, and equitable ",
        "distribution of research resources across diseases and regions."
      )
    ),
    column(
      width = 4,
      div(
        h4("Disease"),
        selectInput("cause", NULL,
                    choices = setNames(causes$cause_code, causes$cause_name),
                    selected = causes$cause_code[[1]]),
        h4("Period"),
        sliderInput("period", NULL,
                    min = 1, max = nrow(periods), value = nrow(periods),
                    step = 1, ticks = FALSE,
                    animate = animationOptions(interval = 900, loop = FALSE)),
        div(class="muted tight",
            textOutput("period_label", inline = TRUE)),
        checkboxInput("logx", "Log X (Burden share)", FALSE),
        checkboxInput("logy", "Log Y (Trial share)", FALSE)
      )
    )
  ),
  
  # Abstract
  fluidRow(
    column(
      width = 12,
      uiOutput("abstract_text")
    )
  ),
  
  # Figure 1: Map
  fluidRow(
    column(
      width = 12,
      div(class="figure-title", "Figure 1. World equity map (ER)"),
      div(class="figure-sub subtle",
          "Countries shaded by Equity Ratio. Hover to see sites, DALYs and shares. ",
          "Colourbar shows ER with a visual cap at ", ER_CAP, " (anything higher is shown as >", ER_CAP, "). ",
          "Blue <1 (underserved), grey ≈1 (aligned), red >1 (over-represented)."),
      plotlyOutput("map_er", height = 480)
    )
  ),
  
  # ---- ADDITION: Figure 1b — zero-site countries colored by burden ----
  fluidRow(
    column(
      width = 12,
      div(class="figure-title", "Figure 1b. Countries with burden but zero trial sites"),
      div(class="figure-sub subtle",
          "Countries highlighted using a red gradient carry non-zero disease burden in the selected period and disease ",
          "but hosted no clinical trial sites. Legend reflects burden intensity (log10 DALYs)."),
      plotlyOutput("map_zero_sites", height = 380)
    )
  ),
  
  # Figure 2 + Figure 3
  fluidRow(
    column(
      width = 7,
      div(class="figure-title", "Figure 2. Burden vs Trial Share (each point is a country)"),
      div(class="figure-sub subtle",
          "The x-axis shows each country’s share of the total disease burden; the y-axis shows its share of all trial sites. ",
          "The dashed diagonal is perfect alignment (ER = 1). Points ",
          strong("above"),
          " the line have ",
          strong("more trial sites than their share of burden"),
          " (ER > 1); points ",
          strong("below"),
          " have ",
          strong("fewer trial sites than their share of burden"),
          " (ER < 1). Bubble size reflects total DALYs, so larger circles indicate higher underlying burden. ",
          "Use the log toggles to emphasise smaller shares and reveal patterns among lower-burden countries. ",
          "Hover to see the country code and its exact ER."
      ),
      plotlyOutput("scatter_bvts", height = 420)
    ),
    column(
      width = 5,
      div(class="figure-title", "Figure 3. Top 10 countries"),
      div(class="figure-sub subtle",
          "Left: lowest ER (underserved). Right: highest ER (over-represented; axis capped at ", ER_CAP, "). ",
          "Hover a bar to see the country and its ER."),
      plotlyOutput("rank_under", height = 330),
      plotlyOutput("rank_over",  height = 330)
    )
  ),
  
  # Key Figures + Formulae
  fluidRow(
    column(
      width = 12,
      h2("Key Figures"),
      tableOutput("key_figs"),
      div(class="muted tight",
          "‘Top-10 share of sites’ is the proportion of all sites placed in the ten highest-ER countries in the selected disease-period.")
    )
  ),
  
  # --- Formulae (two columns) ---
  fluidRow(
    column(width = 12, h2("Formulae"))
  ),
  fluidRow(
    column(width = 6, uiOutput("formulae_left")),
    column(width = 6, uiOutput("formulae_right"))
  ),
  
  # ---------------- Executive Views (NEW) ----------------
  fluidRow(
    column(
      width = 6,
      div(class="figure-title", "Executive View A — Impact: Burden Gap vs Sites Needed"),
      div(class="figure-sub subtle",
          "x: burden share; y: positive gap (burden_share − trial_share). Bubble size = DALYs; colour = sites needed to reach ER = 1.",
          " Use this to prioritise countries where both burden and gap are large."),
      plotlyOutput("impact_gap_sites", height = 380)
    ),
    column(
      width = 6,
      div(class="figure-title", "Executive View B — Trial Density vs Equity Ratio"),
      div(class="figure-sub subtle",
          "x: Equity Ratio (ER). y: trial sites per 1k DALYs (log scale). Vertical line = ER = 1.",
          " Distinguishes efficiency (high density) from inequity (high ER)."),
      plotlyOutput("density_vs_er", height = 380)
    )
  ),
  
  # Policy Planner
  fluidRow(
    column(
      width = 12,
      h2("Policy Planner — Allocate extra sites where gaps are largest"),
      div(class="subtle",
          "This pragmatic rule places additional sites in proportion to the positive share gap (burden_share − trial_share). ",
          "Use the slider to simulate a global uplift and observe the change in inequality. ",
          "This is not a recommendation per se, but a transparent starting point for planning discussions.")
    )
  ),
  fluidRow(
    column(
      width = 4,
      sliderInput("extra_sites", "Add N new sites (global):", min = 0, max = 500, value = 0, step = 10)
    )
  ),
  fluidRow(
    column(width = 5, tableOutput("planner_metrics")),
    column(width = 7, plotlyOutput("planner_alloc", height = 280))
  ),
  fluidRow(
    column(
      width = 12,
      div(class="figure-sub subtle","Before vs after allocation — ER distribution (countries, relative density; log-scale)"),
      plotlyOutput("planner_density", height = 260)
    )
  ),
  
  # Country table
  fluidRow(
    column(
      width = 12,
      h2("Country table"),
      div(class="subtle", "Filtered to the selected disease and period. Tables keep the true ER (uncapped)."),
      DTOutput("table_all")
    )
  )
)

# ----------------------------- Server -----------------------------
server <- function(input, output, session) {
  
  output$period_label <- renderText({
    row <- periods[input$period, ]
    paste0(row$interval_start, "–", row$interval_end)
  })
  
  cur_period <- reactive({
    periods[input$period, ]
  })
  
  df_group <- reactive({
    req(input$cause)
    p <- cur_period()
    df_gold |>
      filter(cause_code == input$cause,
             interval_start == p$interval_start,
             interval_end   == p$interval_end) |>
      mutate(
        er_bin  = er_bin(equity_ratio),
        er_plot = pmin(equity_ratio, ER_CAP)
      )
  })
  
  # ---- ADDITION: zero-site set (has burden > 0, sites == 0) ----
  df_zero <- reactive({
    req(input$cause)
    p <- cur_period()
    df_gold |>
      filter(
        cause_code == input$cause,
        interval_start == p$interval_start,
        interval_end   == p$interval_end,
        trial_sites == 0,
        burden_dalys > 0
      )
  })
  
  # ---------------- Abstract (dynamic text) ----------------
  output$abstract_text <- renderUI({
    d <- df_group() |> filter(trial_sites >= 1)  # <-- ADDITION: use only countries with ≥1 site
    if (!nrow(d)) return(HTML("<p>No rows for this disease-period.</p>"))
    tot_sites <- sum(d$trial_sites, na.rm = TRUE)
    n_cty     <- n_distinct(d$country_iso3)
    med_sites <- median(d$trial_sites, na.rm = TRUE)
    theil     <- d$theil_T[[1]]
    giniw     <- d$gini_er_weighted[[1]]
    
    under <- d |> arrange(equity_ratio) |> slice_head(n = 3) |> pull(country_iso3) |> paste(collapse=", ")
    over  <- d |> arrange(desc(equity_ratio)) |> slice_head(n = 3) |> pull(country_iso3) |> paste(collapse=", ")
    
    lbl <- cur_period()$interval_label
    cname <- df_gold |>
      filter(cause_code == input$cause) |>
      distinct(cause_name) |>
      slice_head(n = 1) |>
      pull(cause_name)
    
    HTML(paste0(
      "Trial evidence should be generalisable to those living with the disease. ",
      "We examine whether site placement for <i>", cname, "</i> matches burden during <b>", lbl, "</b>, ",
      "using WHO DALYs (per-100k, cumulative) and CT.gov site counts.</p>",
      "ER compares each country’s trial share to its burden share. ",
      "Group-level misalignment is summarised by Theil T and the burden-weighted Gini of ER.</p>",
      "Countries analysed: <b>", n_cty, "</b>. Total sites: <b>", fmt_num(tot_sites),
      "</b> (median <b>", fmt_num(med_sites), "</b> per country). ",
      "Inequality: Theil T = <b>", sprintf('%.3f', theil), "</b>, Gini(ER|burden) = <b>",
      sprintf('%.3f', giniw), "</b>. ",
      "Under-represented (lowest ER): <b>", under, "</b>. Over-represented (highest ER): <b>", over, "</b>.</p>",
      "ER ≈ 1 suggests alignment; ER < 1 implies likely under-serving relative to burden. ",
      "These figures support recruitment planning and prioritisation of site placement.</p>"
    ))
  })
  
  # ---------------- Figure 1: Map ----------------
  output$map_er <- renderPlotly({
    d <- df_group() |> filter(trial_sites >= 1)  # <-- ADDITION
    if (!nrow(d)) return(NULL)
    
    hover <- paste0(
      "<b>", d$country_iso3, "</b><br>",
      "ER (true): ", fmt_ratio(d$equity_ratio), "<br>",
      "Sites: ", fmt_num(d$trial_sites), "<br>",
      "DALYs: ", fmt_num(d$burden_dalys), "<br>",
      "Trial share: ", fmt_pct(d$trial_share), "<br>",
      "Burden share: ", fmt_pct(d$burden_share)
    )
    
    plot_ly(
      type = "choropleth",
      locations    = d$country_iso3,
      locationmode = "ISO-3",
      z            = d$er_plot,
      text         = hover,
      hoverinfo    = "text",
      colorscale   = list(
        c(0, 0.5, 1),
        c("#2C6FB7", "#E5E7EB", "#C0392B")
      ),
      zmin = 0,
      zmax = ER_CAP,
      colorbar = list(
        title = "Equity Ratio (ER)",
        tickvals = c(0, 0.5, 1, 2, 5, ER_CAP),
        ticktext = c("0","0.5","1","2","5", paste0(">", ER_CAP))
      )
    ) |>
      layout(
        geo = list(showframe = FALSE, showcoastlines = TRUE,
                   projection = list(type = "natural earth"))
      )
  })
  
  # ---- ADDITION: Figure 1b (zero-site countries; legend = burden) ----
  output$map_zero_sites <- renderPlotly({
    d0 <- df_zero()
    if (!nrow(d0)) return(NULL)
    
    z_log <- log10(pmax(d0$burden_dalys, 1))  # avoid -Inf
    
    red_seq <- list(
      c(0.00, 0.25, 0.50, 0.75, 1.00),
      c("#FEE5D9", "#FCAE91", "#FB6A4A", "#DE2D26", "#A50F15")
    )
    
    hover <- paste0(
      "<b>", d0$country_iso3, "</b><br>",
      "Sites: 0<br>",
      "DALYs: ", fmt_num(d0$burden_dalys), "<br>",
      "Burden share: ", fmt_pct(d0$burden_share)
    )
    
    plot_ly(
      type = "choropleth",
      locations    = d0$country_iso3,
      locationmode = "ISO-3",
      z            = z_log,
      text         = hover,
      hoverinfo    = "text",
      colorscale   = red_seq,
      zmin = min(z_log, na.rm = TRUE),
      zmax = max(z_log, na.rm = TRUE),
      colorbar = list(
        title = "Burden (DALYs, log10)"
      )
    ) |>
      layout(
        geo = list(showframe = FALSE, showcoastlines = TRUE,
                   projection = list(type = "natural earth"))
      )
  })
  
  # ---------------- Figure 2: Scatter ----------------
  output$scatter_bvts <- renderPlotly({
    d <- df_group() |> filter(trial_sites >= 1)  # <-- ADDITION
    if (!nrow(d)) return(NULL)
    p <- ggplot(
      d,
      aes(
        x = burden_share,
        y = trial_share,
        label = country_iso3,
        text = paste0("Country: ", country_iso3, "<br>ER: ", sprintf("%.2f", equity_ratio))
      )
    ) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "#9CA3AF") +
      geom_point(aes(size = burden_dalys, fill = er_plot),
                 shape = 21, colour = "white", alpha = 0.9, stroke = 0.3) +
      scale_fill_gradient2(low = poster_palette$under_deep, mid = poster_palette$neutral,
                           high = poster_palette$over_deep, midpoint = 1, name = "ER (capped)") +
      scale_x_continuous("Burden share", labels = percent) +
      scale_y_continuous("Trial share", labels = percent) +
      guides(size = "none") +
      theme_minimal(base_size = 12) +
      theme(legend.position = "right")
    if (isTRUE(input$logx)) p <- p + scale_x_log10(labels = percent)
    if (isTRUE(input$logy)) p <- p + scale_y_log10(labels = percent)
    ggplotly(p, tooltip = c("text","x","y")) |>
      layout(legend = list(orientation = "v"))
  })
  
  # ---------------- Figure 3: Ranks (hover fixed) ----------------
  output$rank_under <- renderPlotly({
    d <- df_group() |> filter(trial_sites >= 1) |> arrange(equity_ratio) |> slice_head(n = 10)  # <-- ADDITION
    if (!nrow(d)) return(NULL)
    p <- ggplot(
      d,
      aes(
        x = reorder(country_iso3, equity_ratio),
        y = pmin(equity_ratio, ER_CAP),
        text = paste0("Country: ", country_iso3, "<br>ER: ", sprintf("%.2f", equity_ratio))
      )
    ) +
      geom_col(fill = poster_palette$under_deep) +
      coord_flip() +
      labs(x = NULL, y = "ER (capped at 10)") +
      theme_minimal(base_size = 12)
    ggplotly(p, tooltip = "text")
  })
  output$rank_over <- renderPlotly({
    d <- df_group() |> filter(trial_sites >= 1) |> arrange(desc(equity_ratio)) |> slice_head(n = 10)  # <-- ADDITION
    if (!nrow(d)) return(NULL)
    p <- ggplot(
      d,
      aes(
        x = reorder(country_iso3, -equity_ratio),
        y = pmin(equity_ratio, ER_CAP),
        text = paste0("Country: ", country_iso3, "<br>ER: ", sprintf("%.2f", equity_ratio))
      )
    ) +
      geom_col(fill = poster_palette$over_deep) +
      coord_flip() +
      labs(x = NULL, y = "ER (capped at 10)") +
      theme_minimal(base_size = 12)
    ggplotly(p, tooltip = "text")
  })
  
  # ---------------- Key Figures ----------------
  output$key_figs <- renderTable({
    d <- df_group() |> filter(trial_sites >= 1)  # <-- ADDITION
    if (!nrow(d)) return(NULL)
    tot_sites <- sum(d$trial_sites, na.rm = TRUE)
    n_cty     <- n_distinct(d$country_iso3)
    med_sites <- median(d$trial_sites, na.rm = TRUE)
    med_er    <- median(d$equity_ratio, na.rm = TRUE)
    top10_sum <- d |> arrange(desc(equity_ratio)) |> slice_head(n=10) |> summarise(s = sum(trial_sites, na.rm=TRUE)) |> pull()
    top10_shr <- if (tot_sites>0) top10_sum/tot_sites else NA_real_
    theil     <- d$theil_T[[1]]
    giniw     <- d$gini_er_weighted[[1]]
    tibble::tibble(
      Countries = n_cty,
      `Total trial sites` = round(tot_sites, 2),
      `Median sites\n(per country)` = round(med_sites, 2),
      `Top-10 share\nof sites` = percent(top10_shr, 0.01),
      `Median\nER` = round(med_er, 2),
      `Theil\nT` = round(theil, 2),
      `Gini(ER|burden)` = round(giniw, 2)
    )
  }, striped = TRUE, hover = TRUE, spacing = "s", digits = 2)
  
  # ---------------- Formulae blocks (two columns) ----------------
  output$formulae_left <- renderUI({
    withMathJax(HTML("
    <p class='subtle'>
      Within a selected period and disease, we compare how much of the global <i>burden</i> a country carries
      to how much of the global <i>trial activity</i> it hosts.
    </p>
    <p>
      $$\\text{trial_share}_c = \\frac{\\text{sites}_c}{\\sum_i \\text{sites}_i}, 
      \\qquad 
      \\text{burden_share}_c = \\frac{\\text{DALY}_c}{\\sum_i \\text{DALY}_i}$$
    </p>
    <p>
      The central alignment measure is the Equity Ratio:
    </p>
    <p>
      $$ER_c = \\frac{\\text{trial_share}_c}{\\text{burden_share}_c}$$
    </p>
    <p class='subtle'>
      Here: <i>c</i> indexes a country; <i>i</i> runs over all countries; 
      <i>sites</i> = number of clinical trial sites; <i>DALY</i> = disease burden (DALYs, cumulative in the chosen period).
      Shares each sum to 1 across countries.
    </p>
    <p class='subtle'>
      <b>Interpretation cheat-sheet.</b> ER≈1 = aligned. ER&lt;1 = underserved (too few sites for burden).
      ER&gt;1 = over-represented. For visual consistency some charts cap ER at ", ER_CAP, 
                     " (tables and tooltips always show the true value).
    </p>
    "))
  })
  
  output$formulae_right <- renderUI({
    withMathJax(HTML("
    <p>
      We summarise group-level misalignment using inequality statistics:
    </p>
    <p>
      $$T = \\sum_i p_i \\ln\\!\\left(\\frac{p_i}{b_i}\\right), \\qquad
      \\text{Gini}(ER)\\;\\text{weighted by}\\; b_i$$
    </p>
    <p class='subtle'>
      \\(p_i\\) is country <i>i</i>'s trial share and \\(b_i\\) its burden share (weights).
      The <i>Theil T</i> index is 0 when trial shares perfectly match burden shares and increases as they diverge.
      The <i>burden-weighted Gini</i> condenses inequality in ER while giving more emphasis to countries carrying higher burden \\(b_i\\).
    </p>
    <p class='subtle'>
      <b>Practical reading.</b> A lower Theil/Gini means trial activity is better distributed relative to need.
      Policy scenarios (below) simulate adding sites where the positive gap \\((b_i - p_i)^+\\) is largest.
    </p>
    "))
  })
  
  # ---------------- Executive Views (NEW) ----------------
  output$impact_gap_sites <- renderPlotly({
    d <- df_group() |> filter(trial_sites >= 1)  # <-- ADDITION
    if (!nrow(d)) return(NULL)
    total_sites <- sum(d$trial_sites, na.rm = TRUE)
    dd <- d |>
      mutate(
        gap = pmax(burden_share - trial_share, 0),
        sites_to_parity = gap * total_sites
      )
    p <- ggplot(
      dd,
      aes(
        x = burden_share,
        y = gap,
        size = burden_dalys,
        colour = sites_to_parity,
        text = paste0(
          "<b>", country_iso3, "</b>",
          "<br>ER: ", sprintf("%.2f", equity_ratio),
          "<br>Burden share: ", fmt_pct(burden_share),
          "<br>Trial share: ", fmt_pct(trial_share),
          "<br>Gap: ", fmt_pct(gap),
          "<br>Sites to ER=1: ", fmt_num(sites_to_parity),
          "<br>DALYs: ", fmt_num(burden_dalys)
        )
      )
    ) +
      geom_point(alpha = 0.9, stroke = 0) +
      scale_x_continuous("Burden share", labels = percent) +
      scale_y_continuous("Positive gap (burden − trials)", labels = percent) +
      scale_colour_gradient(name = "Sites to parity") +
      guides(size = "none") +
      theme_minimal(base_size = 12) +
      theme(legend.position = "right")
    ggplotly(p, tooltip = "text")
  })
  
  output$density_vs_er <- renderPlotly({
    d <- df_group() |> filter(trial_sites >= 1)  # <-- ADDITION
    if (!nrow(d)) return(NULL)
    p <- ggplot(
      d,
      aes(
        x = equity_ratio,
        y = trial_density_per_k_dalys,
        size = burden_dalys,
        fill = er_plot,
        text = paste0(
          "<b>", country_iso3, "</b>",
          "<br>ER: ", sprintf("%.2f", equity_ratio),
          "<br>Trial density (sites per 1k DALYs): ", signif(trial_density_per_k_dalys, 4),
          "<br>Burden share: ", fmt_pct(burden_share),
          "<br>Trial share: ", fmt_pct(trial_share)
        )
      )
    ) +
      geom_vline(xintercept = 1, linetype = "dashed", colour = "#9CA3AF") +
      geom_point(shape = 21, colour = "white", alpha = 0.9, stroke = 0.3) +
      scale_y_continuous("Sites per 1k DALYs (log scale)", trans = "log10", labels = comma) +
      scale_x_continuous("Equity Ratio (ER)") +
      scale_fill_gradient2(low = poster_palette$under_deep, mid = poster_palette$neutral,
                           high = poster_palette$over_deep, midpoint = 1, name = "ER (capped)") +
      guides(size = "none") +
      theme_minimal(base_size = 12) +
      theme(legend.position = "right")
    ggplotly(p, tooltip = "text")
  })
  
  # ---------------- Policy planner ----------------
  planner_result <- reactive({
    d <- df_group() |> filter(trial_sites >= 1)  # <-- ADDITION
    policy_allocate(d, extra_sites = input$extra_sites)
  })
  
  output$planner_metrics <- renderTable({
    res <- planner_result()
    tibble::tibble(
      Metric = c("Theil T", "Gini(ER|burden)"),
      Current = c(round(res$theil_before, 3), round(res$gini_before, 3)),
      After_Reallocation = c(round(res$theil_after, 3), round(res$gini_after, 3)),
      Change = c(round(res$theil_after - res$theil_before, 3),
                 round(res$gini_after  - res$gini_before, 3))
    )
  }, striped = TRUE, hover = TRUE, spacing = "s", digits = 3)
  
  output$planner_alloc <- renderPlotly({
    res <- planner_result()
    d <- res$after |> arrange(desc(alloc_sites)) |> slice_head(n = 12)
    if (!nrow(d)) return(NULL)
    p <- ggplot(d, aes(x = reorder(country_iso3, alloc_sites), y = alloc_sites)) +
      geom_col(fill = "#0A192F") + coord_flip() +
      labs(x = NULL, y = "Allocated sites", title = "Suggested additions (top 12)") +
      theme_minimal(base_size = 12)
    ggplotly(p, tooltip = c("x","y"))
  })
  
  output$planner_density <- renderPlotly({
    res <- planner_result()
    d0 <- df_group() |> filter(trial_sites >= 1) |> mutate(kind = "Before", er = equity_ratio)  # <-- ADDITION
    d1 <- res$after     |> mutate(kind = "After",  er = equity_ratio_after)
    dd <- bind_rows(d0, d1)
    p <- ggplot(dd, aes(x = er, colour = kind)) +
      geom_density(adjust = 1.2) +
      scale_x_log10("Equity Ratio (ER, log scale)") +
      scale_colour_manual(values = c("Before"="#ef4444","After"="#0A192F"), name = "") +
      labs(y = "Countries (relative density)") +
      theme_minimal(base_size = 12) +
      theme(legend.position = "top")
    ggplotly(p)
  })
  
  # ---------------- Country table ----------------
  output$table_all <- renderDT({
    d <- df_group() |> filter(trial_sites >= 1)  # <-- ADDITION
    d |>
      transmute(
        ISO3 = country_iso3,
        Sites = trial_sites,
        DALYs = burden_dalys,
        `Trial share` = trial_share,
        `Burden share` = burden_share,
        ER = equity_ratio
      ) |>
      datatable(options = list(pageLength = 20), rownames = FALSE) |>
      formatPercentage(c("Trial share","Burden share"), 2) |>
      formatRound(c("ER"), 3) |>
      formatRound(c("Sites","DALYs"), 0)
  })
}

# ----------------------------- Run -----------------------------
shinyApp(ui, server)
