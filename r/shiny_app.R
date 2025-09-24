# r/shiny_app.R
# Global Clinical Trial Equity Dashboard — poster-style, single-file Shiny app
# Reads: data/gold/country_year_disease.parquet

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

# ----------------------------- Paths & load ---------------------------------
GOLD_PATH <- file.path("..", "data", "gold", "country_year_disease.parquet")


if (!file.exists(GOLD_PATH)) {
  stop(paste0(
    "Gold dataset not found at ", normalizePath(GOLD_PATH, winslash = "/"),
    "\nRun your Python pipeline (make gold) to write data/gold/country_year_disease.parquet."
  ))
}

df_gold <- read_parquet(GOLD_PATH) |>
  mutate(
    country_iso3 = as.character(country_iso3),
    cause_code   = as.character(cause_code),
    cause_name   = as.character(cause_name),
    year         = as.integer(year),
    trial_sites  = as.numeric(trial_sites),
    burden_dalys = as.numeric(burden_dalys),
    trial_share  = as.numeric(trial_share),
    burden_share = as.numeric(burden_share),
    equity_ratio = as.numeric(equity_ratio),
    trial_density_per_k_dalys = as.numeric(trial_density_per_k_dalys),
    gini_er_weighted = as.numeric(gini_er_weighted),
    theil_T          = as.numeric(theil_T)
  )

# ----------------------------- Helpers --------------------------------------
poster_palette <- list(
  under_deep = "#2C6FB7",  # deep blue
  under      = "#5C90CA",
  neutral    = "#E5E7EB",  # light gray
  over       = "#E67E73",  # soft red
  over_deep  = "#C0392B"   # deep red
)

er_bin <- function(er) {
  cut(
    er,
    breaks = c(-Inf, 0.5, 0.8, 1.25, 2.0, Inf),
    labels = c("< 0.5", "0.5–0.8", "0.8–1.25", "1.25–2.0", "> 2.0"),
    right = TRUE
  )
}

er_colors <- setNames(
  c(poster_palette$under_deep,
    poster_palette$under,
    poster_palette$neutral,
    poster_palette$over,
    poster_palette$over_deep),
  c("< 0.5","0.5–0.8","0.8–1.25","1.25–2.0","> 2.0")
)

fmt_pct   <- function(x) percent(x, accuracy = 0.01)
fmt_num   <- function(x) comma(x, accuracy = 1)
fmt_ratio <- function(x) ifelse(is.na(x), "—", sprintf("%.2f", x))

compute_ineq_by_year <- function(d) {
  d |>
    group_by(year, cause_code, cause_name) |>
    summarise(
      theil_T = first(theil_T),
      gini    = first(gini_er_weighted),
      .groups = "drop"
    )
}

# Policy planner: allocate S extra sites over positive gaps
# Input: data frame for one disease-year with trial_sites, trial_share, burden_share, etc.
# Returns: list(data_new, metrics_before, metrics_after, alloc_table)
policy_allocate <- function(d_group, extra_sites) {
  d <- d_group
  if (nrow(d) == 0 || extra_sites <= 0) {
    return(list(
      data_new = d,
      metrics_before = c(theil = unique(d$theil_T), gini = unique(d$gini_er_weighted)),
      metrics_after  = c(theil = unique(d$theil_T), gini = unique(d$gini_er_weighted)),
      alloc_table = dplyr::mutate(d, alloc_sites = 0)
    ))
  }
  
  # current totals
  ts_total <- sum(d$trial_sites, na.rm = TRUE)
  # positive share gaps
  gap <- pmax(d$burden_share - d$trial_share, 0)
  gap_sum <- sum(gap, na.rm = TRUE)
  if (gap_sum <= 0) {
    alloc <- rep(0, nrow(d))
  } else {
    alloc <- extra_sites * (gap / gap_sum)
  }
  
  d_after <- d |>
    mutate(
      alloc_sites = alloc,
      trial_sites_after = trial_sites + alloc_sites
    )
  
  # recompute shares & ER after allocation within group
  tot_after <- sum(d_after$trial_sites_after, na.rm = TRUE)
  d_after <- d_after |>
    mutate(
      trial_share_after  = ifelse(tot_after > 0, trial_sites_after / tot_after, NA_real_),
      equity_ratio_after = trial_share_after / burden_share
    )
  
  # recompute Theil T and burden-weighted Gini (ER)
  theil_after <- {
    p <- d_after$trial_share_after
    b <- d_after$burden_share
    mask <- is.finite(p) & is.finite(b) & p >= 0 & b > 0
    p <- p[mask]; b <- b[mask]
    if (length(p) == 0) NA_real_ else {
      p <- p / sum(p); b <- b / sum(b)
      sum(ifelse(p > 0, p * log(p / b), 0))
    }
  }
  
  gini_after <- {
    v <- d_after$equity_ratio_after
    w <- d_after$burden_share
    mask <- is.finite(v) & is.finite(w) & w >= 0
    v <- v[mask]; w <- w[mask]
    if (length(v) == 0 || sum(w) <= 0) NA_real_ else {
      mu <- sum(v * w) / sum(w)
      if (!is.finite(mu) || mu <= 0) NA_real_ else {
        o <- order(v); v <- v[o]; w <- w[o]
        vw <- v * w
        c_vw <- cumsum(vw)
        cw <- cumsum(w)
        W <- cw[length(cw)]
        s <- sum(w * (c(0, c_vw[-length(c_vw)]) + c_vw))
        1 - s / (mu * W * W)
      }
    }
  }
  
  list(
    data_new = d_after,
    metrics_before = c(theil = unique(d$theil_T), gini = unique(d$gini_er_weighted)),
    metrics_after  = c(theil = theil_after, gini = gini_after),
    alloc_table = d_after |>
      transmute(country_iso3, alloc_sites = round(alloc_sites, 1),
                trial_sites_after = round(trial_sites_after, 1),
                equity_ratio_after = equity_ratio_after)
  )
}

# ----------------------------- UI -------------------------------------------
years   <- sort(unique(df_gold$year))
causes  <- df_gold |>
  distinct(cause_code, cause_name) |>
  arrange(cause_name)

app_theme <- bs_theme(
  bootswatch = "flatly",
  base_font = font_google("Inter", local = TRUE)
)

ui <- fluidPage(
  theme = app_theme,
  tags$head(tags$script(HTML("MathJax = {tex: {inlineMath: [['$','$'], ['\\\\(','\\\\)']]}};"))),
  tags$script(src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"),
  tags$style(HTML("
    .panel-card { background: #ffffff; border-radius: 16px; padding: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); margin-bottom: 16px; }
    .metric { font-size: 24px; font-weight: 700; }
    .muted { color: #6b7280; font-size: 12px; }
    .poster-title { font-size: 26px; font-weight: 800; letter-spacing: 0.3px; }
  ")),
  
  fluidRow(
    column(
      width = 8,
      div(class="poster-title", "Global Clinical Trial Equity Dashboard — Alignment of Trial Sites with Disease Burden"),
      div(class="muted", "ER = trial_share / burden_share • Data: CT.gov v2 (sites), WHO DALYs (counts) • Inequality: Theil T & burden-weighted Gini(ER)")
    ),
    column(
      width = 4,
      div(class="panel-card",
          selectInput("cause", "Disease", choices = setNames(causes$cause_code, causes$cause_name)),
          sliderInput("year", "Year", min(min(years), na.rm=TRUE), max(max(years), na.rm=TRUE),
                      value = max(years), step = 1, sep = ""),
          checkboxInput("logx", "Log X (Burden Share)", FALSE),
          checkboxInput("logy", "Log Y (Trial Share)", FALSE)
      )
    )
  ),
  
  fluidRow(
    column(
      width = 8,
      div(class="panel-card",
          h4("Figure 1. World Equity Map (ER)"),
          plotlyOutput("map_er", height = 480),
          div(class="muted", "Bins: < 0.5 (deep blue), 0.5–0.8 (blue), 0.8–1.25 (neutral), 1.25–2.0 (orange), > 2.0 (deep red)")
      )
    ),
    column(
      width = 4,
      div(class="panel-card",
          h4("Formulae"),
          HTML(paste0(
            "$$\\textbf{ER}_{c,y,d} = \\frac{\\text{trial\\_share}_{c,y,d}}{\\text{burden\\_share}_{c,y,d}}$$",
            "<div class='muted'>", 
            "with trial_share = sites / total sites (within disease–year) and burden_share = DALYs / total DALYs.",
            "</div>",
            "$$T_y = \\sum_i p_{i,y}\\,\\ln\\!\\left(\\frac{p_{i,y}}{b_{i,y}}\\right),\\quad \\text{Gini}(ER)\\text{ weighted by }b_i$$"
          )),
          hr(),
          h4("Group Inequality (selected disease–year)"),
          div("Theil T: ", span(textOutput("theil_cur", inline=TRUE), class="metric")),
          div("Gini(ER|burden): ", span(textOutput("gini_cur", inline=TRUE), class="metric")),
          div(class="muted", "0 = perfect alignment; higher = more misalignment.")
      )
    )
  ),
  
  fluidRow(
    column(
      width = 7,
      div(class="panel-card",
          h4("Figure 2. Burden vs Trial Share (each point = country)"),
          plotlyOutput("scatter_bvts", height = 420)
      )
    ),
    column(
      width = 5,
      div(class="panel-card",
          h4("Figure 3. Rankings"),
          fluidRow(
            column(6, plotlyOutput("rank_under", height = 330)),
            column(6, plotlyOutput("rank_over", height = 330))
          )
      )
    )
  ),
  
  fluidRow(
    column(
      width = 6,
      div(class="panel-card",
          h4("Figure 4. Inequality Over Time"),
          plotlyOutput("ineq_time", height = 350)
      )
    ),
    column(
      width = 6,
      div(class="panel-card",
          h4("Figure 5. Policy Planner — Reallocate Extra Sites"),
          sliderInput("extra_sites", "Add N new sites (global):", min = 0, max = 300, value = 0, step = 10),
          tableOutput("planner_metrics"),
          DTOutput("planner_table")
      )
    )
  ),
  
  fluidRow(
    column(
      width = 12,
      div(class="panel-card",
          h4("Country Table (current disease–year)"),
          DTOutput("table_all")
      )
    )
  )
)

# ----------------------------- Server ---------------------------------------
server <- function(input, output, session) {
  
  # Ensure a disease default
  observeEvent(TRUE, {
    if (is.null(input$cause)) {
      updateSelectInput(session, "cause", selected = causes$cause_code[[1]])
    }
  }, once = TRUE)
  
  df_group <- reactive({
    req(input$cause, input$year)
    df_gold |>
      filter(cause_code == input$cause, year == input$year) |>
      mutate(er_bin = er_bin(equity_ratio))
  })
  
  # --- Group-level metrics (duplicated in rows)
  output$theil_cur <- renderText({
    d <- df_group(); if (nrow(d) == 0) return("NA")
    sprintf("%.4f", d$theil_T[[1]])
  })
  output$gini_cur <- renderText({
    d <- df_group(); if (nrow(d) == 0) return("NA")
    sprintf("%.4f", d$gini_er_weighted[[1]])
  })
  
  # --- Map (Plotly choropleth with ISO-3 codes)
  output$map_er <- renderPlotly({
    d <- df_group()
    if (nrow(d) == 0) return(NULL)
    
    bins <- er_bin(d$equity_ratio)
    colors <- er_colors[as.character(bins)]
    hover <- paste0(
      "<b>", d$country_iso3, "</b>",
      "<br>ER: ", fmt_ratio(d$equity_ratio),
      "<br>Sites: ", fmt_num(d$trial_sites),
      "<br>DALYs: ", fmt_num(d$burden_dalys),
      "<br>Trial share: ", fmt_pct(d$trial_share),
      "<br>Burden share: ", fmt_pct(d$burden_share)
    )
    
    plot_ly(
      type = "choropleth",
      locations = d$country_iso3,
      z = as.numeric(bins),
      text = hover,
      hoverinfo = "text",
      colorscale = list(
        c(0, 0.25, 0.5, 0.75, 1),
        unname(unlist(er_colors))
      ),
      showscale = FALSE,
      locationmode = "ISO-3"
    ) |>
      colorbar(title = "ER bins") |>
      layout(
        geo = list(
          showframe = FALSE, showcoastlines = TRUE, projection = list(type = "natural earth")
        )
      )
  })
  
  # --- Scatter: burden share vs trial share
  output$scatter_bvts <- renderPlotly({
    d <- df_group()
    if (nrow(d) == 0) return(NULL)
    p <- ggplot(d, aes(x = burden_share, y = trial_share, label = country_iso3)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#9CA3AF") +
      geom_point(aes(size = burden_dalys, fill = equity_ratio),
                 shape = 21, color = "white", alpha = 0.9, stroke = 0.3) +
      scale_fill_gradient2(low = poster_palette$under_deep, mid = poster_palette$neutral,
                           high = poster_palette$over_deep, midpoint = 1, name = "ER") +
      scale_x_continuous("Burden share", labels = percent) +
      scale_y_continuous("Trial share", labels = percent) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "right")
    if (isTRUE(input$logx)) p <- p + scale_x_log10(labels = percent)
    if (isTRUE(input$logy)) p <- p + scale_y_log10(labels = percent)
    ggplotly(p, tooltip = c("label","x","y","fill")) |> layout(legend = list(orientation = "v"))
  })
  
  # --- Rankers
  output$rank_under <- renderPlotly({
    d <- df_group() |>
      arrange(equity_ratio) |>
      slice_head(n = 10)
    if (nrow(d) == 0) return(NULL)
    p <- ggplot(d, aes(x = reorder(country_iso3, equity_ratio), y = equity_ratio)) +
      geom_col(fill = poster_palette$under_deep) +
      coord_flip() +
      labs(x = NULL, y = "ER", title = "Under-represented (lowest ER)") +
      theme_minimal(base_size = 12)
    ggplotly(p, tooltip = c("x","y"))
  })
  
  output$rank_over <- renderPlotly({
    d <- df_group() |>
      arrange(desc(equity_ratio)) |>
      slice_head(n = 10)
    if (nrow(d) == 0) return(NULL)
    p <- ggplot(d, aes(x = reorder(country_iso3, -equity_ratio), y = equity_ratio)) +
      geom_col(fill = poster_palette$over_deep) +
      coord_flip() +
      labs(x = NULL, y = "ER", title = "Over-represented (highest ER)") +
      theme_minimal(base_size = 12)
    ggplotly(p, tooltip = c("x","y"))
  })
  
  # --- Inequality over time (for selected disease)
  output$ineq_time <- renderPlotly({
    d <- df_gold |>
      filter(cause_code == input$cause) |>
      compute_ineq_by_year()
    
    p <- ggplot(d, aes(x = year)) +
      geom_line(aes(y = theil_T, color = "Theil T"), linewidth = 0.9) +
      geom_point(aes(y = theil_T, color = "Theil T"), size = 1.8) +
      geom_line(aes(y = gini, color = "Gini(ER|burden)"), linewidth = 0.9) +
      geom_point(aes(y = gini, color = "Gini(ER|burden)"), size = 1.8) +
      scale_color_manual(values = c("Theil T" = "#0F766E", "Gini(ER|burden)" = "#7C3AED"), name = NULL) +
      labs(y = "Index", x = NULL) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom")
    ggplotly(p)
  })
  
  # --- Policy planner
  planner_result <- reactive({
    d <- df_group()
    policy_allocate(d, extra_sites = input$extra_sites)
  })
  
  output$planner_metrics <- renderTable({
    res <- planner_result()
    before <- res$metrics_before; after <- res$metrics_after
    tibble::tibble(
      Metric = c("Theil T", "Gini(ER|burden)"),
      Current = c(round(before["theil"], 4), round(before["gini"], 4)),
      After_Reallocation = c(round(after["theil"], 4), round(after["gini"], 4)),
      Change = c(round(after["theil"] - before["theil"], 4),
                 round(after["gini"]  - before["gini"], 4))
    )
  })
  
  output$planner_table <- renderDT({
    res <- planner_result()
    res$alloc_table |>
      arrange(desc(alloc_sites)) |>
      rename(`ISO3` = country_iso3,
             `Allocated sites` = alloc_sites,
             `Sites after` = trial_sites_after,
             `ER after` = equity_ratio_after) |>
      datatable(options = list(pageLength = 10))
  })
  
  # --- Full table
  output$table_all <- renderDT({
    d <- df_group()
    d |>
      transmute(
        ISO3 = country_iso3,
        Sites = trial_sites,
        DALYs = burden_dalys,
        `Trial share` = trial_share,
        `Burden share` = burden_share,
        ER = equity_ratio,
        `Sites per 1k DALYs` = trial_density_per_k_dalys
      ) |>
      datatable(
        options = list(pageLength = 20),
        rownames = FALSE
      ) |>
      formatPercentage(c("Trial share","Burden share"), 2) |>
      formatRound(c("ER","Sites per 1k DALYs"), 3) |>
      formatRound(c("Sites","DALYs"), 0)
  })
}

# ----------------------------- Run ------------------------------------------
shinyApp(ui, server)
