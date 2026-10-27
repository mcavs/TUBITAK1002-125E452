# Script to compute Cavus et al. (2025) metrics across all models and datasets
library(data.table)

cat("=================================================================\n")
cat("Starting Metric Calculation (Cavus et al. 2025 & Serce & Cavus 2026)\n")
cat("=================================================================\n")

# 1. Classification metadata for p and n
classif_rds = list.files("datasets/classification", pattern = "[.]rds$", full.names = TRUE)
dataset_meta = list()
for (f in classif_rds) {
  obj = tryCatch(readRDS(f), error = function(e) NULL)
  if (!is.null(obj)) {
    ds_id = as.character(obj$task$id)
    p = length(obj$task$feature_names)
    n = obj$task$nrow
    dataset_meta[[ds_id]] = list(p = p, n = n, name = obj$task$id)
  }
}
cat(sprintf("Loaded metadata for %d classification datasets.\n", length(dataset_meta)))

dir.create("results/summary_metrics", showWarnings = FALSE, recursive = TRUE)
dir.create("figures", showWarnings = FALSE, recursive = TRUE)

# Default parameter rules per Table A1
get_defaults_for_learner = function(l, p = 10) {
  if (l == "glmnet") {
    list(alpha = 1, lambda = 0)
  } else if (l == "rpart") {
    list(cp = 0.1, maxdepth = 30, minbucket = 7, minsplit = 20)
  } else if (l == "kknn") {
    list(k = 7)
  } else if (l == "svm") {
    list(cost = 1, gamma = 1/p)
  } else if (l == "ranger") {
    list(num.trees = 500, sample.fraction = 1, mtry = max(1L, floor(sqrt(p))), min.node.size = 1)
  } else if (l == "xgboost") {
    list(booster = "gbtree", nrounds = 500, eta = 0.3, subsample = 1, max_depth = 6, 
         min_child_weight = 1, colsample_bytree = 1, colsample_bylevel = 1, lambda = 1, alpha = 0)
  }
}

learners = c("glmnet", "rpart", "kknn", "svm", "ranger", "xgboost")
model_discrepancy_list = list()
hp_discrepancy_list = list()

for (l in learners) {
  files = list.files(file.path("results", l), pattern = "classif.*[.]csv$", full.names = TRUE)
  files = files[!grepl("summary", files)]
  cat(sprintf("\nProcessing learner: %s (%d datasets)\n", l, length(files)))
  
  for (f in files) {
    bn = basename(f)
    ds_id = gsub(".*classif_([0-9]+).*", "\\1", bn)
    
    # Read CSV
    dt = tryCatch(fread(f), error = function(e) NULL)
    if (is.null(dt) || nrow(dt) == 0) next
    
    # Check fold1_predictions
    if (!("fold1_predictions" %in% names(dt))) next
    
    # Identify default row
    def_idx = integer(0)
    if ("is_default" %in% names(dt)) {
      def_idx = which(dt$is_default == TRUE)
    }
    
    p_meta = if (!is.null(dataset_meta[[ds_id]])) dataset_meta[[ds_id]]$p else 10
    defs = get_defaults_for_learner(l, p = p_meta)
    
    if (length(def_idx) == 0) {
      if (l == "ranger") {
        def_idx = which(dt$num.trees == 500 & abs(dt$sample.fraction - 1) < 1e-4 & 
                        dt$mtry == defs$mtry & dt$min.node.size == 1)
      } else if (l == "svm") {
        def_idx = which(abs(dt$cost - 1) < 1e-4 & abs(dt$gamma - defs$gamma) < 1e-4)
        if (length(def_idx) == 0) def_idx = which(abs(dt$cost - 1) < 1e-4)[1]
      } else if (l == "xgboost") {
        # Check closest match to defaults
        match_scores = rep(0, nrow(dt))
        if ("nrounds" %in% names(dt)) match_scores = match_scores + (abs(dt$nrounds - defs$nrounds) < 1e-4)
        if ("eta" %in% names(dt)) match_scores = match_scores + (abs(dt$eta - defs$eta) < 1e-4)
        if ("subsample" %in% names(dt)) match_scores = match_scores + (abs(dt$subsample - defs$subsample) < 1e-4)
        if ("max_depth" %in% names(dt)) match_scores = match_scores + (abs(dt$max_depth - defs$max_depth) < 1e-4)
        if ("alpha" %in% names(dt)) match_scores = match_scores + (abs(dt$alpha - defs$alpha) < 1e-4)
        if ("lambda" %in% names(dt)) match_scores = match_scores + (abs(dt$lambda - defs$lambda) < 1e-4)
        best_match = which.max(match_scores)
        def_idx = best_match
      }
    }
    
    if (length(def_idx) == 0 || is.na(def_idx[1])) {
      # Fallback: take median or default if not matched
      def_idx = 1
    } else {
      def_idx = def_idx[1]
    }
    
    def_row = dt[def_idx]
    def_preds_str = def_row$fold1_predictions
    if (is.na(def_preds_str) || def_preds_str == "") next
    def_preds = strsplit(def_preds_str, ";")[[1]]
    n_inst = length(def_preds)
    if (n_inst == 0) next
    
    def_f1 = if ("perf_f1" %in% names(def_row)) def_row$perf_f1 else NA_real_
    valid_dt = dt[!is.na(fold1_predictions) & fold1_predictions != ""]
    
    # Fast discrepancy calculation
    discs = vapply(valid_dt$fold1_predictions, function(p_str) {
      p_vec = strsplit(p_str, ";")[[1]]
      if (length(p_vec) != n_inst) return(NA_real_)
      mean(p_vec != def_preds)
    }, FUN.VALUE = numeric(1))
    
    # Model Discrepancy (Eq 2)
    model_disc = max(discs, na.rm = TRUE)
    mean_disc  = mean(discs, na.rm = TRUE)
    best_f1    = max(valid_dt$perf_f1, na.rm = TRUE)
    tunability = if (!is.na(best_f1) && !is.na(def_f1)) max(0, best_f1 - def_f1) else NA_real_
    
    model_discrepancy_list[[length(model_discrepancy_list) + 1]] = data.table(
      learner     = l,
      dataset_id  = ds_id,
      discrepancy = model_disc,
      mean_discrepancy = mean_disc,
      default_f1  = def_f1,
      best_f1     = best_f1,
      tunability  = tunability,
      n_inst      = n_inst
    )
    
    # Hyperparameter Discrepancy (Eq 4)
    # Vary 1 HP while others fixed to default
    param_names = names(defs)
    param_names = intersect(param_names, names(dt))
    
    for (hp in param_names) {
      other_hps = setdiff(param_names, hp)
      # Filter rows where other HPs are default
      cond = rep(TRUE, nrow(valid_dt))
      for (ohp in other_hps) {
        # Prefer actual default value present in def_row if valid
        ohp_def = if (ohp %in% names(def_row) && !is.na(def_row[[ohp]])) def_row[[ohp]] else defs[[ohp]]
        if (is.numeric(ohp_def)) {
          cond = cond & (abs(valid_dt[[ohp]] - ohp_def) < 1e-4)
        } else {
          cond = cond & (valid_dt[[ohp]] == ohp_def)
        }
      }
      
      idx_hp = which(cond)
      if (length(idx_hp) > 0) {
        hp_disc = max(discs[idx_hp], na.rm = TRUE)
      } else {
        # If strict defaults not available across all combinations, use overall marginal variance
        hp_disc = NA_real_
      }
      
      if (!is.na(hp_disc)) {
        hp_discrepancy_list[[length(hp_discrepancy_list) + 1]] = data.table(
          learner     = l,
          dataset_id  = ds_id,
          hyperparameter = hp,
          discrepancy = hp_disc
        )
      }
    }
  }
}

dt_model_disc = rbindlist(model_discrepancy_list)
dt_hp_disc    = rbindlist(hp_discrepancy_list)

fwrite(dt_model_disc, "results/summary_metrics/model_discrepancy.csv")
fwrite(dt_hp_disc, "results/summary_metrics/hp_discrepancy.csv")

cat("\n=== Summary Table: Model Discrepancy & Tunability (Cavus et al. Table 1) ===\n")
table1 = dt_model_disc[, .(
  mean_discrepancy = mean(discrepancy, na.rm = TRUE),
  sd_discrepancy   = sd(discrepancy, na.rm = TRUE),
  mean_tunability  = mean(tunability, na.rm = TRUE),
  sd_tunability    = sd(tunability, na.rm = TRUE),
  n_datasets       = .N
), by = learner]

print(table1)
fwrite(table1, "results/summary_metrics/table1_model_summary.csv")
cat("\nResults successfully saved in results/summary_metrics/!\n")
