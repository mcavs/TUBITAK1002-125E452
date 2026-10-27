# TÜBİTAK 1002 - Main Execution Script
# Based on Cavus et al. (2025) experimental design
library(mlr3)
library(mlr3learners)
library(data.table)

# Load core experiment script
source("src/run_experiments.R")

# Create results folder
dir.create("results", showWarnings = FALSE)

# Models to test (Table A1)
models_to_test = c("glmnet", "rpart", "kknn", "svm", "ranger", "xgboost")

# Grid resolution: number of equally spaced values per hyperparameter
# Adjust this to control total number of configurations
RESOLUTION = 2

# 1. Classification Pilot Run
cat("\n=========================================\n")
cat("Starting Classification Pilot on 'sonar'\n")
cat("=========================================\n")
task_classif = tsk("sonar")

classif_results = list()
for (m in models_to_test) {
  t0 = Sys.time()
  res = run_single_experiment(m, task_classif, resolution = RESOLUTION, seed = 123, parallel = TRUE)
  t1 = Sys.time()
  cat(sprintf("[LOG] Model %s completed in %.2f seconds\n", m, as.numeric(t1 - t0, units = "secs")))
  classif_results[[m]] = res
}
saveRDS(classif_results, "results/pilot_classif_sonar.rds")
cat("[LOG] Classification pilot saved to results/pilot_classif_sonar.rds\n")

# 2. Regression Pilot Run
cat("\n=========================================\n")
cat("Starting Regression Pilot on 'mtcars'\n")
cat("=========================================\n")
task_regr = tsk("mtcars")

regr_results = list()
for (m in models_to_test) {
  t0 = Sys.time()
  res = run_single_experiment(m, task_regr, resolution = RESOLUTION, seed = 123, parallel = TRUE)
  t1 = Sys.time()
  cat(sprintf("[LOG] Model %s completed in %.2f seconds\n", m, as.numeric(t1 - t0, units = "secs")))
  regr_results[[m]] = res
}
saveRDS(regr_results, "results/pilot_regr_mtcars.rds")
cat("[LOG] Regression pilot saved to results/pilot_regr_mtcars.rds\n")