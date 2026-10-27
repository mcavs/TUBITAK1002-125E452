# TÜBİTAK 1002 - Utility Functions for Model Performance and Tunability

library(mlr3)
library(mlr3learners)
library(paradox)

# ==========================================
# 1. Classification Performance Metrics (Vectorized)
# ==========================================

# Helper: compute confusion matrix once, derive all per-class metrics vectorized
compute_confusion_matrix = function(truth, response) {
  truth = factor(truth)
  response = factor(response)
  lvls = levels(truth)
  
  table(factor(truth, levels = lvls), factor(response, levels = lvls))
}

# Accuracy
calculate_accuracy = function(truth, response) {
  mean(truth == response, na.rm = TRUE)
}

# Balanced Accuracy (Mean of Recall for each class) - VECTORIZED
calculate_balanced_accuracy = function(truth, response) {
  conf_matrix = compute_confusion_matrix(truth, response)
  
  # tp = diagonal, fn = row sums - diagonal
  tp = diag(conf_matrix)
  fn = rowSums(conf_matrix) - tp
  
  recalls = tp / (tp + fn)
  recalls[is.nan(recalls)] = 0
  mean(recalls)
}

# Macro Precision - VECTORIZED
calculate_macro_precision = function(truth, response) {
  conf_matrix = compute_confusion_matrix(truth, response)
  
  # tp = diagonal, fp = column sums - diagonal
  tp = diag(conf_matrix)
  fp = colSums(conf_matrix) - tp
  
  precisions = tp / (tp + fp)
  precisions[is.nan(precisions)] = 0
  mean(precisions)
}

# Macro Recall - VECTORIZED
calculate_macro_recall = function(truth, response) {
  conf_matrix = compute_confusion_matrix(truth, response)
  
  tp = diag(conf_matrix)
  fn = rowSums(conf_matrix) - tp
  
  recalls = tp / (tp + fn)
  recalls[is.nan(recalls)] = 0
  mean(recalls)
}

# Macro F1 - VECTORIZED
calculate_macro_f1 = function(truth, response) {
  conf_matrix = compute_confusion_matrix(truth, response)
  
  tp = diag(conf_matrix)
  fp = colSums(conf_matrix) - tp
  fn = rowSums(conf_matrix) - tp
  
  precisions = tp / (tp + fp)
  precisions[is.nan(precisions)] = 0
  
  recalls = tp / (tp + fn)
  recalls[is.nan(recalls)] = 0
  
  f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
  f1_scores[is.nan(f1_scores)] = 0
  
  mean(f1_scores)
}

# Geometric Mean (G-Mean) of recalls - VECTORIZED
calculate_gmean = function(truth, response) {
  conf_matrix = compute_confusion_matrix(truth, response)
  
  tp = diag(conf_matrix)
  fn = rowSums(conf_matrix) - tp
  
  recalls = tp / (tp + fn)
  recalls[is.nan(recalls)] = 0
  
  if (any(recalls == 0)) return(0)
  exp(mean(log(recalls)))
}

# Cohen's Kappa (Agreement between predictions and ground truth)
calculate_cohen_kappa = function(truth, response) {
  truth = as.character(truth)
  response = as.character(response)
  lvls = unique(c(truth, response))
  
  tab = table(factor(truth, levels = lvls), factor(response, levels = lvls))
  n = sum(tab)
  if (n == 0) return(0)
  po = sum(diag(tab)) / n
  pe = sum(rowSums(tab) * colSums(tab)) / (n^2)
  if (pe == 1) return(1)
  kappa = (po - pe) / (1 - pe)
  return(kappa)
}

# AUC (Area Under the ROC Curve) for binary classification
# truth: binary labels (0/1), probs: positive class probabilities [0,1]
calculate_auc = function(truth, probs) {
  if (is.null(probs) || length(probs) == 0) return(NA_real_)
  if (length(truth) != length(probs)) return(NA_real_)
  
  truth = as.numeric(as.character(truth))
  probs = as.numeric(probs)
  
  # Check if binary
  if (length(unique(truth)) != 2) return(NA_real_)
  
  # Use rank-based AUC calculation (efficient, no ties issue)
  n_pos = sum(truth == 1)
  n_neg = sum(truth == 0)
  if (n_pos == 0 || n_neg == 0) return(NA_real_)
  
  # Rank probabilities in ascending order (required by Wilcoxon formula)
  rank_all = rank(probs, ties.method = "average")
  rank_pos = rank_all[truth == 1]
  
  # AUC = (sum of positive ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
  auc = (sum(rank_pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
  return(auc)
}

# Matthews Correlation Coefficient (MCC) - highly balanced metric for binary/multi-class
calculate_mcc = function(truth, response) {
  truth = as.numeric(truth)
  response = as.numeric(response)
  
  conf_matrix = compute_confusion_matrix(truth, response)
  
  # For binary: MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
  # For multi-class: use formula per Gorodkin (2004)
  tp = diag(conf_matrix)
  fp = colSums(conf_matrix) - tp
  fn = rowSums(conf_matrix) - tp
  tn = sum(conf_matrix) - tp - fp - fn
  
  cov_yhat_y = sum(tp) * sum(tn) - sum(fp) * sum(fn)
  sqrt_term = sum(tp + fp) * sum(tp + fn) * sum(tn + fp) * sum(tn + fn)
  
  if (sqrt_term == 0) return(0)
  mcc = cov_yhat_y / sqrt(sqrt_term)
  return(mcc)
}

# Specificity (True Negative Rate, TNR) for binary classification
calculate_specificity = function(truth, response) {
  conf_matrix = compute_confusion_matrix(truth, response)
  
  # For binary (class 0 = negative, class 1 = positive)
  # Specificity = TN / (TN + FP)
  if (nrow(conf_matrix) != 2) {
    # Multi-class: return macro-specificity
    tn_vec = numeric(nrow(conf_matrix))
    fp_vec = numeric(nrow(conf_matrix))
    for (i in seq_len(nrow(conf_matrix))) {
      tn_vec[i] = sum(conf_matrix[-i, -i])
      fp_vec[i] = colSums(conf_matrix)[-i][i] - conf_matrix[i, i]
    }
    specs = tn_vec / (tn_vec + fp_vec)
    specs[is.nan(specs)] = 0
    return(mean(specs))
  }
  
  tn = conf_matrix[1, 1]
  fp = conf_matrix[1, 2]
  return(tn / (tn + fp))
}

# RMSLE (Root Mean Squared Log Error) for regression
calculate_rmsle = function(truth, response) {
  # RMSLE = sqrt(mean((log(truth+1) - log(response+1))^2))
  # Used for positive target values (e.g., price, count)
  if (any(truth < 0) || any(response < 0)) {
    return(NA_real_)  # RMSLE requires non-negative values
  }
  sqrt(mean((log(truth + 1) - log(response + 1))^2, na.rm = TRUE))
}

# SMAPE (Symmetric Mean Absolute Percentage Error) for regression
calculate_smape = function(truth, response) {
  # SMAPE = mean(2 * |truth - response| / (|truth| + |response|)) * 100
  diffs = abs(truth - response)
  sums = abs(truth) + abs(response)
  
  valid = sums > 0  # exclude division by zero
  if (sum(valid) == 0) return(NA_real_)
  
  mean(2 * diffs[valid] / sums[valid], na.rm = TRUE) * 100
}

# ==========================================
# 2. Regression Performance Metrics
# ==========================================

# Mean Squared Error (MSE)
calculate_mse = function(truth, response) {
  mean((truth - response)^2, na.rm = TRUE)
}

# Root Mean Squared Error (RMSE)
calculate_rmse = function(truth, response) {
  sqrt(mean((truth - response)^2, na.rm = TRUE))
}

# Mean Absolute Error (MAE)
calculate_mae = function(truth, response) {
  mean(abs(truth - response), na.rm = TRUE)
}

# Mean Absolute Percentage Error (MAPE)
calculate_mape = function(truth, response) {
  # Avoid division by zero by filtering out truth == 0
  valid = (truth != 0)
  if (sum(valid) == 0) return(NA_real_)
  mean(abs((truth[valid] - response[valid]) / truth[valid]), na.rm = TRUE) * 100
}

# R-squared (R2)
calculate_r2 = function(truth, response) {
  rss = sum((truth - response)^2, na.rm = TRUE)
  tss = sum((truth - mean(truth, na.rm = TRUE))^2, na.rm = TRUE)
  if (tss == 0) return(0)
  1 - (rss / tss)
}

# Adjusted R-squared
calculate_adjusted_r2 = function(truth, response, p) {
  r2 = calculate_r2(truth, response)
  n = length(truth)
  if (n - p - 1 <= 0) return(r2)
  1 - (1 - r2) * (n - 1) / (n - p - 1)
}

# Maximum Absolute Error (MaxAE)
calculate_max_ae = function(truth, response) {
  max(abs(truth - response), na.rm = TRUE)
}

# Median Absolute Error (MedAE)
calculate_med_ae = function(truth, response) {
  median(abs(truth - response), na.rm = TRUE)
}

# ==========================================
# 3. Dynamic Search Spaces (Table A1 - Cavus et al. 2025)
# ==========================================
get_search_space = function(learner_id, task) {
  p = length(task$feature_names)
  n = task$nrow
  
  if (learner_id == "glmnet") {
    ps(
      alpha = p_dbl(lower = 0, upper = 1),
      lambda = p_dbl(lower = 2^-10, upper = 2^10)
    )
  } else if (learner_id == "rpart") {
    ps(
      cp = p_dbl(lower = 0, upper = 1),
      maxdepth = p_int(lower = 1, upper = 30),
      minbucket = p_int(lower = 1, upper = 60),
      minsplit = p_int(lower = 1, upper = 60)
    )
  } else if (learner_id == "kknn") {
    ps(
      k = p_int(lower = 1, upper = 30)
    )
  } else if (learner_id == "svm") {
    # kernel = "radial" olarak sabitlendi; degree yalnızca polynomial kernel
    # için anlamlı olduğundan arama uzayına dahil edilmedi.
    ps(
      cost  = p_dbl(lower = 2^-10, upper = 2^10),
      gamma = p_dbl(lower = 2^-10, upper = 2^10)
    )
  } else if (learner_id == "ranger") {
    ps(
      num.trees = p_int(lower = 1, upper = 2000),
      sample.fraction = p_dbl(lower = 0.1, upper = 1),
      mtry = p_int(lower = 1, upper = max(1, p)),
      min.node.size = p_int(lower = 1, upper = n)
    )
  } else if (learner_id == "xgboost") {
    ps(
      nrounds = p_int(lower = 1, upper = 5000),
      eta = p_dbl(lower = 0, upper = 1),
      subsample = p_dbl(lower = 0.1, upper = 1),
      min_child_weight = p_dbl(lower = 1, upper = 14),
      lambda = p_dbl(lower = 2^-10, upper = 2^10),
      alpha = p_dbl(lower = 2^-10, upper = 2^10)
    )
  } else {
    stop("Unknown learner_id")
  }
}

# ==========================================
# 4. Default Configurations (Table A1 - Cavus et al. 2025)
# ==========================================
get_default_values = function(learner_id, task, type = "classif") {
  p = length(task$feature_names)
  n = task$nrow
  
  if (learner_id == "glmnet") {
    list(alpha = 1, lambda = 0)
  } else if (learner_id == "rpart") {
    list(cp = 0.1, maxdepth = 30, minbucket = 7, minsplit = 20)
  } else if (learner_id == "kknn") {
    list(k = 7)
  } else if (learner_id == "svm") {
    # gamma = 1/p: veriye bağlı default (ranger'daki mtry gibi)
    list(cost = 1, gamma = 1/p)
  } else if (learner_id == "ranger") {
    mtry_val = if (type == "classif") floor(sqrt(p)) else floor(p/3)
    mtry_val = max(1, mtry_val)
    min_node = if (type == "classif") 1 else 5
    list(num.trees = 500, sample.fraction = 1, mtry = mtry_val, min.node.size = min_node)
  } else if (learner_id == "xgboost") {
    list(booster = "gbtree", nrounds = 500, eta = 0.3, subsample = 1, max_depth = 6, min_child_weight = 1,
         colsample_bytree = 1, colsample_bylevel = 1, lambda = 1, alpha = 0)
  } else {
    stop("Unknown learner_id")
  }
}

# ==========================================
# 5. Grid Design Generator
# ==========================================
# Generate a full factorial grid design by dividing each hyperparameter's
# [lower, upper] range into equally spaced values.
generate_grid_design = function(search_space, resolution = 10, default_params = list()) {
  # Generate a full factorial grid but ensure each parameter's DEFAULT value is included
  # Cap per-parameter candidate set size to `resolution`
  param_ids = search_space$ids()
  grid_list = list()
  
  # In paradox >= 1.0, $lower, $upper, $class are named vectors
  lower_vec = search_space$lower
  upper_vec = search_space$upper
  class_vec = search_space$class
  
  for (pid in param_ids) {
    lower_val = lower_vec[[pid]]
    upper_val = upper_vec[[pid]]
    default_val = if (!is.null(default_params[[pid]])) default_params[[pid]] else NA
    
    if (class_vec[[pid]] == "ParamInt") {
      # Integer: generate resolution evenly spaced integers (unique)
      seq_vals = unique(as.integer(round(seq(lower_val, upper_val, length.out = resolution))))
      vals = seq_vals
      # Ensure default included
      if (!is.na(default_val) && !(default_val %in% vals)) {
        vals = unique(c(vals, as.integer(default_val)))
      }
      # If over budget, remove the value farthest from default (or farthest from midpoint if default NA)
      if (length(vals) > resolution) {
        if (!is.na(default_val)) {
          dists = abs(vals - as.integer(default_val))
        } else {
          mid = (min(seq_vals) + max(seq_vals)) / 2
          dists = abs(vals - mid)
        }
        # remove the one with largest distance (prefer removing duplicates at ends)
        remove_idx = which.max(dists)
        vals = vals[-remove_idx]
      }
    } else if (class_vec[[pid]] == "ParamDbl") {
      seq_vals = seq(lower_val, upper_val, length.out = resolution)
      vals = seq_vals
      # Ensure default included (within tolerance)
      tol = .Machine$double.eps^0.5 * max(1, abs(lower_val), abs(upper_val))
      if (!is.na(default_val) && all(abs(vals - as.numeric(default_val)) > tol)) {
        vals = unique(c(vals, as.numeric(default_val)))
      }
      # If over budget, remove the value farthest from default (or midpoint if default NA)
      if (length(vals) > resolution) {
        if (!is.na(default_val)) {
          dists = abs(vals - as.numeric(default_val))
        } else {
          mid = (min(seq_vals) + max(seq_vals)) / 2
          dists = abs(vals - mid)
        }
        remove_idx = which.max(dists)
        vals = vals[-remove_idx]
      }
    } else {
      # Categorical/nominal
      vals = NULL
      if (!is.null(search_space$values) && !is.null(search_space$values[[pid]])) {
        vals = as.character(search_space$values[[pid]])
      } else if (!is.null(search_space$levels) && !is.null(search_space$levels[[pid]])) {
        vals = as.character(search_space$levels[[pid]])
      }
      if (is.null(vals)) {
        # fallback to numeric sequence
        vals = as.character(seq(lower_val, upper_val, length.out = resolution))
      }
      # Ensure default included
      if (!is.na(default_val)) vals = unique(c(as.character(default_val), vals))
      # If more than resolution levels, truncate but keep default
      if (length(vals) > resolution) {
        if (!is.na(default_val)) {
          other = setdiff(vals, as.character(default_val))
          vals = c(as.character(default_val), head(other, resolution - 1))
        } else {
          vals = head(vals, resolution)
        }
      }
    }
    grid_list[[pid]] = vals
  }
  
  # Build full factorial grid
  # expand.grid in older R versions may not accept stringsAsFactors via do.call; set globally if needed
  if (getRversion() >= '4.0.0') {
    grid = do.call(expand.grid, c(grid_list, stringsAsFactors = FALSE))
  } else {
    grid = do.call(expand.grid, grid_list)
  }
  # Ensure proper typing for integer parameters
  for (pid in param_ids) {
    if (class_vec[[pid]] == "ParamInt") {
      grid[[pid]] = as.integer(round(grid[[pid]]))
    }
  }
  
  grid_dt = data.table::as.data.table(grid)
  
  # Inject optimal defaults for xgboost parameters removed from search space
  # Based on Probst et al. (2019) Table 3
  if ("nrounds" %in% names(grid_dt) && "eta" %in% names(grid_dt)) {
    # If this is xgboost (identified by its parameters)
    grid_dt[, booster := "gbtree"]
    grid_dt[, max_depth := 13]
    grid_dt[, colsample_bytree := 0.752]
    grid_dt[, colsample_bylevel := 0.585]
  }

  if ("booster" %in% names(grid_dt)) {
    # If booster is gblinear, tree-specific parameters are ignored.
    # Set them to NA and keep only unique rows to avoid redundant model training.
    tree_params = c("subsample", "max_depth", "min_child_weight", "colsample_bytree", "colsample_bylevel")
    gblinear_rows = which(grid_dt$booster == "gblinear")
    if (length(gblinear_rows) > 0) {
      for (tp in tree_params) {
        if (tp %in% names(grid_dt)) {
          grid_dt[gblinear_rows, (tp) := NA]
        }
      }
    }
    grid_dt = unique(grid_dt)
  }
  
  return(grid_dt)
}

# ==========================================
# 6. Helper: Map learner_id to mlr3 learner name
# ==========================================
get_mlr3_learner_name = function(learner_id, type = "classif") {
  paste0(type, ".", learner_id)
}
