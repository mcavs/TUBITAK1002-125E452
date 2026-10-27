# TÜBİTAK 1002 - Run rpart experiments on ALL datasets
# Saves each dataset result as individual CSV:
#   rpart_{classif|regr}_{datasetID}_res{resolution}.csv
# Uses run_and_save() from run_experiments.R

library(mlr3)
library(data.table)
library(progressr)

source("src/run_experiments.R")
plan(multisession, workers = 12)  # GCP e2-custom-12-49152: 12 vCPU

# Show a progress bar in the console for each grid search
handlers(global = TRUE)
handlers("txtprogressbar")

# Sort dataset files by overall size factor (rows * features) (smallest first)
sort_files_by_size = function(files, decreasing = FALSE) {
  sizes = sapply(files, function(f) {
    obj = tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(obj)) return(Inf)
    obj$task$nrow * length(obj$task$feature_names)
  })
  files[order(sizes, decreasing = decreasing)]
}

RESOLUTION = 20
MODEL = "rpart"
SEED = 123
OUTPUT_DIR = "results/rpart"

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==========================================
# 1. Classification datasets
# ==========================================
classif_files = sort_files_by_size(list.files("datasets/classification", pattern = "\\.rds$", full.names = TRUE))
cat(sprintf("\n========================================\n"))
cat(sprintf("  rpart Experiments: %d classification datasets\n", length(classif_files)))
cat(sprintf("  Resolution: %d | 10-fold CV\n", RESOLUTION))
cat(sprintf("  Output: %s/\n", OUTPUT_DIR))
cat(sprintf("========================================\n\n"))

classif_summaries = list()
for (i in seq_along(classif_files)) {
  fpath = classif_files[i]
  fname = basename(fpath)
  
  # Extract dataset ID from filename (e.g., task_37_diabetes.rds -> 37)
  dataset_id = sub("task_([0-9]+)_.*", "\\1", fname)
  
  # Skip Internet-Advertisements (167125) and Bioresponse (9910) as requested by user
  if (dataset_id %in% c("167125", "9910")) {
    cat(sprintf("[%d/%d] SKIP (user requested): %s (ID: %s)\n", i, length(classif_files), fname, dataset_id))
    next
  }
  
  # ONLY run the missing datasets (failed or not run)
  allowed_classif_ids = c("9976")
  if (!(dataset_id %in% allowed_classif_ids)) {
    cat(sprintf("[%d/%d] SKIP (already completed locally): %s\n", i, length(classif_files), fname))
    next
  }
  
  # Build CSV filename: rpart_classif_{datasetID}_res{resolution}.csv
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
  
  # Filter to the first 10% of rows for nomao (9977) and all subsequent datasets
  # (which includes bank-marketing, electricity, adult, and numerai28.6)
  is_nomao_or_after_classif = (dataset_id %in% c("9977", "14965", "219", "7592", "167120"))
  if (is_nomao_or_after_classif) {
    row_ids = task$row_ids
    subset_size = max(10, round(length(row_ids) * 0.10))
    subset_ids = row_ids[1:min(length(row_ids), subset_size)]
    task$filter(subset_ids)
    cat(sprintf("  [SUBSET 10%%] Filtered task to %d/%d rows\n", task$nrow, length(row_ids)))
    
    # Save results as sub10 csv name instead
    csv_name = sprintf("%s_classif_%s_res%d_sub10.csv", MODEL, dataset_id, RESOLUTION)
    csv_path = file.path(OUTPUT_DIR, csv_name)
    
    # Skip if already completed (sub10 file)
    if (file.exists(csv_path)) {
      cat(sprintf("[%d/%d] SKIP (exists sub10): %s\n", i, length(classif_files), csv_name))
      next
    }
  }
  
  cat(sprintf("  Task: %s | %d rows x %d cols\n", task$id, task$nrow, task$ncol))
  
  t0 = Sys.time()
  res = tryCatch(
    run_single_experiment(MODEL, task, resolution = RESOLUTION, seed = SEED, parallel = TRUE),
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
cat(sprintf("  rpart Experiments: %d regression datasets\n", length(regr_files)))
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
  
  # Skip Mercedes_Benz_Greener_Manufacturing (233215) and Santander_transaction_value (233214) to save time as requested by user
  if (dataset_id %in% c("233215", "233214")) {
    cat(sprintf("[%d/%d] SKIP (user requested): %s (ID: %s)\n", i, length(regr_files), fname, dataset_id))
    next
  }

  # ONLY run the missing datasets (failed or not run)
  allowed_regr_ids = c("233211", "359937")
  if (!(dataset_id %in% allowed_regr_ids)) {
    cat(sprintf("[%d/%d] SKIP (already completed locally or not targeted): %s\n", i, length(regr_files), fname))
    next
  }

  # Skip if already completed
  if (file.exists(csv_path)) {
    cat(sprintf("[%d/%d] SKIP (exists): %s\n", i, length(regr_files), csv_name))
    next
  }
  
  cat(sprintf("[%d/%d] %s ...\n", i, length(regr_files), fname))
  
  obj = tryCatch(readRDS(fpath), error = function(e) { cat(sprintf("  [ERROR] Cannot read: %s\n", e$message)); NULL })
  if (is.null(obj)) next
  
  task = obj$task
  
  # Filter to the first 10% of rows for diamonds (233211) and black_friday (359937)
  is_subset_regr = (dataset_id %in% c("233211", "359937"))
  if (is_subset_regr) {
    row_ids = task$row_ids
    subset_size = max(10, round(length(row_ids) * 0.10))
    subset_ids = row_ids[1:min(length(row_ids), subset_size)]
    task$filter(subset_ids)
    cat(sprintf("  [SUBSET 10%%] Filtered task to %d/%d rows\n", task$nrow, length(row_ids)))
    
    # Save results as sub10 csv name instead
    csv_name = sprintf("%s_regr_%s_res%d_sub10.csv", MODEL, dataset_id, RESOLUTION)
    csv_path = file.path(OUTPUT_DIR, csv_name)
    
    # Skip if already completed (sub10 file)
    if (file.exists(csv_path)) {
      cat(sprintf("[%d/%d] SKIP (exists sub10): %s\n", i, length(regr_files), csv_name))
      next
    }
  }
  
  cat(sprintf("  Task: %s | %d rows x %d cols\n", task$id, task$nrow, task$ncol))
  
  t0 = Sys.time()
  res = tryCatch(
    run_single_experiment(MODEL, task, resolution = RESOLUTION, seed = SEED, parallel = TRUE),
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
cat("  ALL rpart EXPERIMENTS COMPLETED\n")
cat(sprintf("  Classification: %d datasets\n", length(classif_summaries)))
cat(sprintf("  Regression: %d datasets\n", length(regr_summaries)))
cat(sprintf("  Output dir: %s/\n", OUTPUT_DIR))
cat("========================================\n")
