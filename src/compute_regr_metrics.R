# Script to compute Regression metrics across all models per Serce & Cavus (2026)
library(data.table)

cat("=================================================================\n")
cat("Starting Regression Metric Calculation (Serce & Cavus 2026)\n")
cat("=================================================================\n")

# 1. Load Regression dataset metadata (y_range, p, n)
regr_rds = list.files("datasets/regression", pattern = "[.]rds$", full.names = TRUE)
dataset_meta = list()
for (f in regr_rds) {
  obj = tryCatch(readRDS(f), error = function(e) NULL)
  if (!is.null(obj)) {
    ds_id = as.character(obj$task$id)
    p = length(obj$task$feature_names)
    n = obj$task$nrow
    y = obj$task$truth()
    y_range = max(y, na.rm = TRUE) - min(y, na.rm = TRUE)
    dataset_meta[[ds_id]] = list(p = p, n = n, name = obj$task$id, y_range = y_range)
  }
}
cat(sprintf("Loaded metadata for %d regression datasets.\n", length(dataset_meta)))

get_regr_defaults = function(l, p = 10, n = 1000) {
  if (l == "glmnet") {
    list(alpha = 1, lambda = 0)
  } else if (l == "rpart") {
    list(cp = 0.1, maxdepth = 30, minbucket = 7, minsplit = 20)
  } else if (l == "kknn") {
    list(k = 7)
  } else if (l == "svm") {
    list(cost = 1, gamma = 1/p)
  } else if (l == "ranger") {
    list(num.trees = 500, sample.fraction = 1, mtry = max(1L, floor(p/3)), min.node.size = 5)
  } else if (l == "xgboost") {
    list(booster = "gbtree", nrounds = 500, eta = 0.3, subsample = 1, max_depth = 6, 
         min_child_weight = 1, colsample_bytree = 1, colsample_bylevel = 1, lambda = 1, alpha = 0)
  }
}

learners = c("glmnet", "rpart", "kknn", "svm", "ranger", "xgboost")
model_regr_list = list()
hp_regr_list = list()

for (l in learners) {
  files = list.files(file.path("results", l), pattern = "regr.*[.]csv$", full.names = TRUE)
  files = files[!grepl("summary", files)]
  cat(sprintf("\nProcessing regression learner: %s (%d datasets)\n", l, length(files)))
  
  for (f in files) {
    bn = basename(f)
    ds_id = gsub(".*regr_([0-9]+).*", "\\1", bn)
    
    dt = tryCatch(fread(f), error = function(e) NULL)
    if (is.null(dt) || nrow(dt) == 0) next
    if (!("fold1_predictions" %in% names(dt))) next
    
    p_meta = if (!is.null(dataset_meta[[ds_id]])) dataset_meta[[ds_id]]$p else 10
    n_meta = if (!is.null(dataset_meta[[ds_id]])) dataset_meta[[ds_id]]$n else 1000
    y_range = if (!is.null(dataset_meta[[ds_id]])) dataset_meta[[ds_id]]$y_range else NA_real_
    
    defs = get_regr_defaults(l, p = p_meta, n = n_meta)
    
    # Identify default row
    def_idx = integer(0)
    if ("is_default" %in% names(dt)) {
      def_idx = which(dt$is_default == TRUE)
    }
    
    if (length(def_idx) == 0) {
      if (l == "ranger") {
        def_idx = which(dt$num.trees == 500 & abs(dt$sample.fraction - 1) < 1e-4 & 
                        dt$mtry == defs$mtry & dt$min.node.size == defs$min.node.size)
        if (length(def_idx) == 0) def_idx = which(dt$num.trees == 500 & abs(dt$sample.fraction - 1) < 1e-4)
      } else if (l == "svm") {
        def_idx = which(abs(dt$cost - 1) < 1e-4 & abs(dt$gamma - defs$gamma) < 1e-4)
        if (length(def_idx) == 0) def_idx = which(abs(dt$cost - 1) < 1e-4)[1]
      } else if (l == "xgboost") {
        match_scores = rep(0, nrow(dt))
        if ("nrounds" %in% names(dt)) match_scores = match_scores + (abs(dt$nrounds - defs$nrounds) < 1e-4)
        if ("eta" %in% names(dt)) match_scores = match_scores + (abs(dt$eta - defs$eta) < 1e-4)
        if ("subsample" %in% names(dt)) match_scores = match_scores + (abs(dt$subsample - defs$subsample) < 1e-4)
        if ("alpha" %in% names(dt)) match_scores = match_scores + (abs(dt$alpha - defs$alpha) < 1e-4)
        if ("lambda" %in% names(dt)) match_scores = match_scores + (abs(dt$lambda - defs$lambda) < 1e-4)
        def_idx = which.max(match_scores)
      }
    }
    
    if (length(def_idx) == 0 || is.na(def_idx[1])) def_idx = 1
    else def_idx = def_idx[1]
    
    def_row = dt[def_idx]
    def_preds_str = def_row$fold1_predictions
    if (is.na(def_preds_str) || def_preds_str == "") next
    def_preds = as.numeric(strsplit(def_preds_str, ";")[[1]])
    n_inst = length(def_preds)
    if (n_inst == 0) next
    
    # Calculate y_range from fold1_truth if not already known
    if (is.na(y_range) || y_range == 0) {
      if ("fold1_truth" %in% names(def_row)) {
        t_vals = as.numeric(strsplit(def_row$fold1_truth, ";")[[1]])
        y_range = max(t_vals, na.rm = TRUE) - min(t_vals, na.rm = TRUE)
      }
    }
    if (is.na(y_range) || y_range == 0) y_range = max(def_preds, na.rm = TRUE) - min(def_preds, na.rm = TRUE)
    if (is.na(y_range) || y_range == 0) y_range = 1.0 # fallback
    
    def_rmse = if ("perf_rmse" %in% names(def_row)) def_row$perf_rmse else NA_real_
    def_r2   = if ("perf_r2" %in% names(def_row)) def_row$perf_r2 else NA_real_
    
    valid_dt = dt[!is.na(fold1_predictions) & fold1_predictions != ""]
    
    # Calculate normalized regression predictive multiplicity:
    # Serce & Cavus (2026) Eq 1: mean( |f_j - f_def| / (y_max - y_min) )
    # delta-Discrepancy (Eq 3) with delta = 0.05: mean( (|f_j - f_def| / y_range) >= 0.05 )
    norm_mults = vapply(valid_dt$fold1_predictions, function(p_str) {
      p_vec = as.numeric(strsplit(p_str, ";")[[1]])
      if (length(p_vec) != n_inst) return(NA_real_)
      mean(abs(p_vec - def_preds) / y_range, na.rm = TRUE)
    }, FUN.VALUE = numeric(1))
    
    delta_discs = vapply(valid_dt$fold1_predictions, function(p_str) {
      p_vec = as.numeric(strsplit(p_str, ";")[[1]])
      if (length(p_vec) != n_inst) return(NA_real_)
      mean((abs(p_vec - def_preds) / y_range) >= 0.05, na.rm = TRUE)
    }, FUN.VALUE = numeric(1))
    
    max_multiplicity = max(norm_mults, na.rm = TRUE)
    max_delta_disc   = max(delta_discs, na.rm = TRUE)
    mean_multiplicity = mean(norm_mults, na.rm = TRUE)
    
    best_rmse = if ("perf_rmse" %in% names(valid_dt)) min(valid_dt$perf_rmse, na.rm = TRUE) else NA_real_
    best_r2   = if ("perf_r2" %in% names(valid_dt)) max(valid_dt$perf_r2, na.rm = TRUE) else NA_real_
    
    # Tunability in RMSE (relative reduction) or R2 gain
    tunability_r2 = if (!is.na(best_r2) && !is.na(def_r2)) max(0, best_r2 - def_r2) else NA_real_
    tunability_rmse_rel = if (!is.na(best_rmse) && !is.na(def_rmse) && def_rmse > 0) max(0, (def_rmse - best_rmse) / def_rmse) else NA_real_
    
    model_regr_list[[length(model_regr_list) + 1]] = data.table(
      learner             = l,
      dataset_id          = ds_id,
      multiplicity        = max_multiplicity,
      delta_discrepancy   = max_delta_disc,
      mean_multiplicity   = mean_multiplicity,
      default_rmse        = def_rmse,
      best_rmse           = best_rmse,
      tunability_rmse_rel = tunability_rmse_rel,
      default_r2          = def_r2,
      best_r2             = best_r2,
      tunability_r2       = tunability_r2,
      n_inst              = n_inst
    )
    
    # HP Discrepancy for regression
    param_names = names(defs)
    param_names = intersect(param_names, names(dt))
    for (hp in param_names) {
      other_hps = setdiff(param_names, hp)
      cond = rep(TRUE, nrow(valid_dt))
      for (ohp in other_hps) {
        ohp_def = defs[[ohp]]
        if (is.numeric(ohp_def)) {
          cond = cond & (abs(valid_dt[[ohp]] - ohp_def) < 1e-4)
        } else {
          cond = cond & (valid_dt[[ohp]] == ohp_def)
        }
      }
      idx_hp = which(cond)
      if (length(idx_hp) > 0) {
        hp_mult = max(norm_mults[idx_hp], na.rm = TRUE)
        hp_regr_list[[length(hp_regr_list) + 1]] = data.table(
          learner        = l,
          dataset_id     = ds_id,
          hyperparameter = hp,
          multiplicity   = hp_mult
        )
      }
    }
  }
}

dt_regr_model = rbindlist(model_regr_list)
dt_regr_hp    = rbindlist(hp_regr_list)

fwrite(dt_regr_model, "results/summary_metrics/regr_model_discrepancy.csv")
fwrite(dt_regr_hp, "results/summary_metrics/regr_hp_discrepancy.csv")

cat("\n=== Regression Summary Table (Serce & Cavus 2026) ===\n")
regr_table = dt_regr_model[, .(
  mean_multiplicity   = mean(multiplicity, na.rm = TRUE),
  sd_multiplicity     = sd(multiplicity, na.rm = TRUE),
  mean_delta_disc     = mean(delta_discrepancy, na.rm = TRUE),
  sd_delta_disc       = sd(delta_discrepancy, na.rm = TRUE),
  mean_tunability_r2  = mean(tunability_r2, na.rm = TRUE),
  sd_tunability_r2    = sd(tunability_r2, na.rm = TRUE),
  n_datasets          = .N
), by = learner]

print(regr_table)
fwrite(regr_table, "results/summary_metrics/table_regr_model_summary.csv")
cat("\nRegression calculation finished successfully!\n")
