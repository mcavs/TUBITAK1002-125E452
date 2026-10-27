# TÜBİTAK 1002 - Run kknn experiments on ALL datasets
# Saves each dataset result as individual CSV:
#   kknn_{classif|regr}_{datasetID}_res{resolution}.csv
# Uses run_and_save() from run_experiments.R

library(mlr3)
library(data.table)
library(progressr)

source("src/run_experiments.R")

# Show a progress bar in the console for each grid search
handlers(global = TRUE)
handlers("txtprogressbar")

# Sort dataset files by number of rows (smallest first)
sort_files_by_size = function(files, decreasing = FALSE) {
  sizes = sapply(files, function(f) {
    obj = tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(obj)) return(Inf)
    obj$task$nrow
  })
  files[order(sizes, decreasing = decreasing)]
}

RESOLUTION = 30
MODEL = "kknn"
SEED = 123
OUTPUT_DIR = "results/kknn"
USE_SUBSET = TRUE
SUBSET_RATIO = 0.05

# future'ın global obje boyut limitini kaldır (büyük datasetlerde gerekli)
options(future.globals.maxSize = Inf)

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==========================================
# 1. Classification datasets
# ==========================================
classif_files = sort_files_by_size(list.files("datasets/classification", pattern = "\\.rds$", full.names = TRUE))
cat(sprintf("\n========================================\n"))
cat(sprintf("  kknn Experiments: %d classification datasets\n", length(classif_files)))
cat(sprintf("  Resolution: %d | 10-fold CV\n", RESOLUTION))
cat(sprintf("  Output: %s/\n", OUTPUT_DIR))
cat(sprintf("========================================\n\n"))

classif_summaries = list()
for (i in seq_along(classif_files)) {
  fpath = classif_files[i]
  fname = basename(fpath)
  
  # Extract dataset ID from filename (e.g., task_37_diabetes.rds -> 37)
  dataset_id = sub("task_([0-9]+)_.*", "\\1", fname)
  
  # Build CSV filename: kknn_classif_{datasetID}_res{resolution}.csv
  csv_name = sprintf("%s_classif_%s_res%d.csv", MODEL, dataset_id, RESOLUTION)
  csv_path = file.path(OUTPUT_DIR, csv_name)
  
  # Skip if already completed
  if (file.exists(csv_path)) {
    cat(sprintf("[%d/%d] SKIP (exists): %s\n", i, length(classif_files), csv_name))
    next
  }
  
  cat(sprintf("[%d/%d] %s ...\n", i, length(classif_files), fname))
  
  obj = tryCatch(readRDS(fpath), error = function(e) { cat(sprintf("  [ERROR] Cannot read: %s\n", e$message)); NULL })
  if (is.null(obj)) next
  
  task = obj$task
  if (USE_SUBSET) {
    n_sub = max(10L, ceiling(task$nrow * SUBSET_RATIO))
    task$filter(seq_len(n_sub))
    cat(sprintf("  [SUBSET] İlk %d satır kullanılıyor (toplam %d)\n", n_sub, obj$task$nrow))
  }
  cat(sprintf("  Task: %s | %d rows x %d cols\n", task$id, task$nrow, task$ncol))
  
  t0 = Sys.time()
  res = tryCatch(
    run_single_experiment(MODEL, task, resolution = RESOLUTION, seed = SEED, parallel = FALSE),
    error = function(e) { cat(sprintf("  [ERROR] %s\n", e$message)); NULL }
  )
  t1 = Sys.time()
  elapsed = as.numeric(t1 - t0, units = "secs")
  
  if (!is.null(res)) {
    # Save CSV
    dt = copy(res$results)
    
    # Remove tunability_ columns
    tun_cols = grep("^tunability_", names(dt), value = TRUE)
    if (length(tun_cols) > 0) dt[, (tun_cols) := NULL]
    
    # Add dataset_id as integer first column
    dt[, dataset_id := as.integer(dataset_id)]
    data.table::setcolorder(dt, c("dataset_id", setdiff(names(dt), "dataset_id")))
    
    # Add is_default: TRUE for the default hyperparameter configuration
    def_params = res$default_params
    param_names = names(def_params)
    dt[, is_default := Reduce(`&`, lapply(param_names, function(p) dt[[p]] == def_params[[p]]))]
    
    # Convert list columns to semicolon-joined strings for CSV compatibility
    if ("fold1_predictions" %in% names(dt))
      dt[, fold1_predictions := sapply(fold1_predictions, function(x) if (is.null(x)) NA else paste(x, collapse = ";"))]
    if ("fold1_truth" %in% names(dt))
      dt[, fold1_truth := sapply(fold1_truth, function(x) if (is.null(x)) NA else paste(x, collapse = ";"))]
    if ("fold1_positive_prob" %in% names(dt))
      dt[, fold1_positive_prob := sapply(fold1_positive_prob, function(x) if (is.null(x)) NA else paste(round(as.numeric(x), 6), collapse = ";"))]
    
    fwrite(dt, csv_path)
    
    valid = res$results[is.na(error)]
    cat(sprintf("  Done in %.1fs | Valid: %d/%d | Default F1: %.4f | Best F1: %.4f\n",
                elapsed, nrow(valid), nrow(res$results),
                res$default_performance$perf_f1,
                max(valid$perf_f1, na.rm = TRUE)))
    cat(sprintf("  Saved: %s\n", csv_name))
    
    # Store summary
    classif_summaries[[dataset_id]] = data.table(
      dataset_id = as.integer(dataset_id),
      task_name = task$id,
      n_rows = task$nrow,
      n_cols = task$ncol,
      default_f1 = round(res$default_performance$perf_f1, 4),
      best_f1 = round(max(valid$perf_f1, na.rm = TRUE), 4),
      n_valid = nrow(valid),
      time_sec = round(elapsed, 1),
      csv_file = csv_name
    )
  } else {
    cat(sprintf("  Failed after %.1fs\n", elapsed))
  }
  cat("\n")
}

# Save classification summary
if (length(classif_summaries) > 0) {
  summary_classif = rbindlist(classif_summaries)
  fwrite(summary_classif, file.path(OUTPUT_DIR, "summary_classif.csv"))
  cat(sprintf("[LOG] Classification summary saved: %d datasets\n\n", nrow(summary_classif)))
}

# ==========================================
# 2. Regression datasets
# ==========================================
regr_files = sort_files_by_size(list.files("datasets/regression", pattern = "\\.rds$", full.names = TRUE))
cat(sprintf("\n========================================\n"))
cat(sprintf("  kknn Experiments: %d regression datasets\n", length(regr_files)))
cat(sprintf("  Resolution: %d | 10-fold CV\n", RESOLUTION))
cat(sprintf("  Output: %s/\n", OUTPUT_DIR))
cat(sprintf("========================================\n\n"))

regr_summaries = list()
for (i in seq_along(regr_files)) {
  fpath = regr_files[i]
  fname = basename(fpath)
  
  dataset_id = sub("task_([0-9]+)_.*", "\\1", fname)
  csv_name = sprintf("%s_regr_%s_res%d.csv", MODEL, dataset_id, RESOLUTION)
  csv_path = file.path(OUTPUT_DIR, csv_name)
  
  # Skip if already completed
  if (file.exists(csv_path)) {
    cat(sprintf("[%d/%d] SKIP (exists): %s\n", i, length(regr_files), csv_name))
    next
  }
  
  cat(sprintf("[%d/%d] %s ...\n", i, length(regr_files), fname))
  
  obj = tryCatch(readRDS(fpath), error = function(e) { cat(sprintf("  [ERROR] Cannot read: %s\n", e$message)); NULL })
  if (is.null(obj)) next
  
  task = obj$task
  if (USE_SUBSET) {
    n_sub = max(10L, ceiling(task$nrow * SUBSET_RATIO))
    task$filter(seq_len(n_sub))
    cat(sprintf("  [SUBSET] İlk %d satır kullanılıyor (toplam %d)\n", n_sub, obj$task$nrow))
  }
  cat(sprintf("  Task: %s | %d rows x %d cols\n", task$id, task$nrow, task$ncol))
  
  t0 = Sys.time()
  res = tryCatch(
    run_single_experiment(MODEL, task, resolution = RESOLUTION, seed = SEED, parallel = FALSE),
    error = function(e) { cat(sprintf("  [ERROR] %s\n", e$message)); NULL }
  )
  t1 = Sys.time()
  elapsed = as.numeric(t1 - t0, units = "secs")
  
  if (!is.null(res)) {
    dt = copy(res$results)
    
    # Remove tunability_ columns
    tun_cols = grep("^tunability_", names(dt), value = TRUE)
    if (length(tun_cols) > 0) dt[, (tun_cols) := NULL]
    
    # Add dataset_id as integer first column
    dt[, dataset_id := as.integer(dataset_id)]
    data.table::setcolorder(dt, c("dataset_id", setdiff(names(dt), "dataset_id")))
    
    # Add is_default: TRUE for the default hyperparameter configuration
    def_params = res$default_params
    param_names = names(def_params)
    dt[, is_default := Reduce(`&`, lapply(param_names, function(p) dt[[p]] == def_params[[p]]))]
    
    if ("fold1_predictions" %in% names(dt))
      dt[, fold1_predictions := sapply(fold1_predictions, function(x) if (is.null(x)) NA else paste(x, collapse = ";"))]
    if ("fold1_truth" %in% names(dt))
      dt[, fold1_truth := sapply(fold1_truth, function(x) if (is.null(x)) NA else paste(x, collapse = ";"))]
    if ("fold1_positive_prob" %in% names(dt))
      dt[, fold1_positive_prob := sapply(fold1_positive_prob, function(x) if (is.null(x)) NA else paste(round(as.numeric(x), 6), collapse = ";"))]
    
    fwrite(dt, csv_path)
    
    valid = res$results[is.na(error)]
    cat(sprintf("  Done in %.1fs | Valid: %d/%d | Default R2: %.4f | Best R2: %.4f\n",
                elapsed, nrow(valid), nrow(res$results),
                res$default_performance$perf_r2,
                max(valid$perf_r2, na.rm = TRUE)))
    cat(sprintf("  Saved: %s\n", csv_name))
    
    regr_summaries[[dataset_id]] = data.table(
      dataset_id = as.integer(dataset_id),
      task_name = task$id,
      n_rows = task$nrow,
      n_cols = task$ncol,
      default_r2 = round(res$default_performance$perf_r2, 4),
      best_r2 = round(max(valid$perf_r2, na.rm = TRUE), 4),
      n_valid = nrow(valid),
      time_sec = round(elapsed, 1),
      csv_file = csv_name
    )
  } else {
    cat(sprintf("  Failed after %.1fs\n", elapsed))
  }
  cat("\n")
}

# Save regression summary
if (length(regr_summaries) > 0) {
  summary_regr = rbindlist(regr_summaries)
  fwrite(summary_regr, file.path(OUTPUT_DIR, "summary_regr.csv"))
  cat(sprintf("[LOG] Regression summary saved: %d datasets\n", nrow(summary_regr)))
}

cat("\n========================================\n")
cat("  ALL kknn EXPERIMENTS COMPLETED\n")
cat(sprintf("  Classification: %d datasets\n", length(classif_summaries)))
cat(sprintf("  Regression: %d datasets\n", length(regr_summaries)))
cat(sprintf("  Output dir: %s/\n", OUTPUT_DIR))
cat("========================================\n")
