# R/renv_setup.R
# Generates renv.lock with a curated, dashboard-ready stack

suppressPackageStartupMessages({
  options(repos = c(CRAN = "https://cran.rstudio.com"))
  if (.Platform$OS.type == "windows") options(pkgType = "binary")
})

pkgs <- c(
  # Shiny + UI
  "shiny", "bslib", "htmltools", "DT",
  # Data wrangling & plotting
  "arrow", "dplyr", "tidyr", "readr", "ggplot2", "scales", "glue",
  # Maps
  "sf", "rnaturalearth", "rnaturalearthdata",
  # Optional interactivity
  "plotly"
)

ensure_pkg <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message(sprintf("[renv_setup] installing %s ...", p))
    install.packages(p)
  }
}

# 1) Ensure renv is available
ensure_pkg("renv")

# 2) Initialize or activate the project
if (!dir.exists("renv") || !file.exists(file.path("renv", "settings.dcf"))) {
  message("[renv_setup] initializing renv project (bare=TRUE)")
  renv::init(bare = TRUE)
} else {
  message("[renv_setup] renv already initialized; activating")
  renv::activate()
}

# 3) Prefer cache (faster installs), keep default snapshot type
try(renv::settings$use.cache(TRUE), silent = TRUE)

# 4) Install dashboard stack (idempotent)
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) {
  message("[renv_setup] installing packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install)
} else {
  message("[renv_setup] all dashboard packages already installed")
}

# 5) Sanity checks (some heavy pkgs)
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop(sprintf("[renv_setup] package %s failed to install. Check output above.", p))
  }
}

# 6) Snapshot exact versions to renv.lock
message("[renv_setup] snapshotting lockfile")
renv::snapshot(prompt = FALSE)

# 7) Print summary
message("[renv_setup] lockfile: ", renv::paths$lockfile())
print(renv::status())

message("\n[renv_setup] Done. Commit renv.lock and renv/settings.dcf to your repo.")
