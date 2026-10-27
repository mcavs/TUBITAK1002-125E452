# TÜBİTAK 1002 - Core Experiment Execution Script
# Based on Cavus et al. (2025) and Elif Serce & Mustafa Cavus (2026) experimental design:
#   - Grid search over hyperparameters
#   - 10-fold cross-validation
#   - Raw Fold 1 predictions and truth values saved per configuration for post-hoc discrepancy calculation
#   - Scale-independent regression normalization range capture

library(mlr3)
library(mlr3learners)
library(paradox)
library(data.table)
library(future)
library(future.apply)
library(progressr)

# Increase the max size of exported objects for parallel workers to prevent errors on large datasets
options(future.globals.maxSize = Inf)

# Source utility functions
source("src/utils.R")

# Preprocessing function to encode, impute missing values, and remove constant columns
preprocess_task = function(task) {
  library(mlr3pipelines)
  
  # 1. Verideki eksik (NA) değerleri imputation (ortalama/mod) ile doldur
  # Nümerik değişkenler için mean, kategorik (fct) değişkenler için en sık geçen mod kullanılır
  imputer_num = po("imputemean", affect_columns = selector_type("numeric"))
  imputer_fct = po("imputemode", affect_columns = selector_type("factor"))
  
  trained_output = imputer_num$train(list(task))
  new_task = trained_output[[1]]
  
  trained_output = imputer_fct$train(list(new_task))
  new_task = trained_output[[1]]
  
  # 2. Faktör (kategorik) değişkenleri dummy değişkenlere dönüştürür (one-hot encoding)
  encoder = po("encode", method = "one-hot", affect_columns = selector_type("factor"))
  trained_output = encoder$train(list(new_task))
  new_task = trained_output[[1]]
  
  # 3. Standartlaştırma hatasına yol açan (sabit/sıfır varyanslı) kolonları temizle
  cleaner = po("removeconstants")
  cleaned_output = cleaner$train(list(new_task))
  new_task = cleaned_output[[1]]
  
  return(new_task)
}

# Helper: train a single learner on given train/test indices and return predictions + metrics
train_and_predict = function(learner_id, learner_name, task_type, preprocessed_task,
                             params, train_idx, test_idx, test_truth) {
  l = lrn(learner_name)
  if (task_type == "classif") {
    # Request both class response and probabilities for positive class '1'
    l$predict_type = "prob"
  }
  
  # Clear dependencies to allow setting arbitrary hyperparameter combinations
  l$param_set$deps = data.table::data.table()
  
  # Filter NA parameters (representing inactive dependent parameters)
  params = params[!is.na(params)]
  
  # Set parameters
  l$param_set$values = params
  
  # Fix for xgboost verbose parameter issue on R >= 4.3
  if (learner_id == "xgboost") {
    unlockBinding("assert", l$param_set)
    l$param_set$assert = function(xs, ...) xs
    l$param_set$values$verbose = FALSE
  }
  
  # Remove constant columns dynamically for this fold training set to avoid libsvm scaling division-by-zero errors
  fold_task = preprocessed_task$clone()
  train_data = fold_task$data(rows = train_idx)
  constant_cols = c()
  for (col in fold_task$feature_names) {
    vals = train_data[[col]]
    if (is.numeric(vals)) {
      v = var(vals, na.rm = TRUE)
      if (is.na(v) || v == 0) {
        constant_cols = c(constant_cols, col)
      }
    } else {
      if (length(unique(vals)) <= 1) {
        constant_cols = c(constant_cols, col)
      }
    }
  }
  if (length(constant_cols) > 0) {
    fold_task$select(setdiff(fold_task$feature_names, constant_cols))
  }
  
  l$train(fold_task, row_ids = train_idx)
  pred = l$predict(fold_task, row_ids = test_idx)
  
  prob_vec = NULL
  if (task_type == "classif") {
    # pred$response still available; pred$prob gives probabilities per class
    # Ensure positive class label '1' exists in prob columns
    if (!is.null(pred$prob)) {
      # if '1' column exists use it, else use second column
      if ("1" %in% colnames(pred$prob)) {
        prob_vec = pred$prob[, "1"]
      } else {
        prob_vec = pred$prob[, ncol(pred$prob)]
      }
    }
    metrics = list(
      perf_f1    = calculate_macro_f1(test_truth, pred$response),
      perf_acc   = calculate_accuracy(test_truth, pred$response),
      perf_bacc  = calculate_balanced_accuracy(test_truth, pred$response),
      perf_prec  = calculate_macro_precision(test_truth, pred$response),
      perf_rec   = calculate_macro_recall(test_truth, pred$response),
      perf_gmean = calculate_gmean(test_truth, pred$response),
      perf_kappa = calculate_cohen_kappa(test_truth, pred$response),
      perf_mcc   = calculate_mcc(test_truth, pred$response),
      perf_spec  = calculate_specificity(test_truth, pred$response),
      perf_auc   = calculate_auc(test_truth, prob_vec)
    )
  } else {
    p_count = length(preprocessed_task$feature_names)
    metrics = list(
      perf_mse    = calculate_mse(test_truth, pred$response),
      perf_rmse   = calculate_rmse(test_truth, pred$response),
      perf_mae    = calculate_mae(test_truth, pred$response),
      perf_mape   = calculate_mape(test_truth, pred$response),
      perf_rmsle  = calculate_rmsle(test_truth, pred$response),
      perf_smape  = calculate_smape(test_truth, pred$response),
      perf_r2     = calculate_r2(test_truth, pred$response),
      perf_adj_r2 = calculate_adjusted_r2(test_truth, pred$response, p_count),
      perf_max_ae = calculate_max_ae(test_truth, pred$response),
      perf_med_ae = calculate_med_ae(test_truth, pred$response)
    )
  }
  
  return(list(metrics = metrics, predictions = pred$response, probs = prob_vec))
}

# Run 10-fold CV for a parameter configuration, return averaged metrics + fold1 predictions
run_cv_for_config = function(learner_id, learner_name, task_type, preprocessed_task,
                             params, cv_folds) {
  n_folds = length(cv_folds)
  fold_metrics = list()
  fold1_predictions = NULL
  fold1_truth = NULL
  
  for (fold_idx in seq_len(n_folds)) {
    train_idx = cv_folds[[fold_idx]]$train
    test_idx = cv_folds[[fold_idx]]$test
    test_truth = preprocessed_task$truth(test_idx)
    
    result = train_and_predict(learner_id, learner_name, task_type, preprocessed_task,
                               params, train_idx, test_idx, test_truth)
    fold_metrics[[fold_idx]] = result$metrics
    
    # Save fold 1 predictions and truth for post-hoc analysis
    if (fold_idx == 1) {
      fold1_predictions = result$predictions
      fold1_truth = test_truth
      # capture probabilities for fold1 if available
      if (!is.null(result$probs)) {
        fold1_prob = result$probs
      }
    }
  }
  
  # Average metrics across folds
  metric_names = names(fold_metrics[[1]])
  avg_metrics = list()
  for (mn in metric_names) {
    vals = sapply(fold_metrics, function(x) x[[mn]])
    avg_metrics[[mn]] = mean(vals, na.rm = TRUE)
  }
  
  return(list(
    avg_metrics = avg_metrics,
    fold1_predictions = fold1_predictions,
    fold1_truth = fold1_truth,
    fold1_prob = if (exists('fold1_prob')) fold1_prob else NULL
  ))
}

# Run a single experiment for a model on a task
run_single_experiment = function(learner_id, task, resolution = 10, seed = 123, parallel = TRUE) {
  set.seed(seed)
  task_type = task$task_type  # "classif" or "regr"
  
  cat(sprintf("[LOG] Starting experiment for Learner: %s on Task: %s (%s)\n", learner_id, task$id, task_type))
  
  # 1. Preprocess the task
  preprocessed_task = preprocess_task(task)
  
  # If classification: map majority -> "0" and minority -> "1" BEFORE training so all models learn same binary target
  if (task_type == "classif") {
    all_truth = preprocessed_task$truth()
    tab = table(all_truth)
    if (length(tab) < 2) {
      stop("Classification task must have at least 2 classes to perform binary mapping.")
    }
    maj = names(tab)[which.max(tab)]
    minc = names(tab)[which.min(tab)]
    class_mapping = list(majority = maj, minority = minc)
    # Create mapped target vector (character "0"/"1") and rebuild task with same feature columns
    target_name = preprocessed_task$target_names
    df = preprocessed_task$data()
    mapped = ifelse(df[[target_name]] == maj, "0", "1")
    df[[target_name]] = factor(mapped, levels = c("0", "1"))
    # Recreate a TaskClassif with the same id and feature set but binary target
    # Use mlr3::TaskClassif to avoid namespace issues
    new_task = mlr3::TaskClassif$new(id = preprocessed_task$id, backend = df, target = target_name)
    preprocessed_task = new_task
    cat(sprintf("[LOG] Binary mapping applied to task '%s': majority '%s'->0, minority '%s'->1\n", task$id, maj, minc))
  }
  learner_name = get_mlr3_learner_name(learner_id, task_type)
  
  # Normalization factor calculation for regression (y_max - y_min) as formulated in Methodology
  y_range = NA_real_
  if (task_type == "regr") {
    all_y = preprocessed_task$truth()
    y_range = max(all_y, na.rm = TRUE) - min(all_y, na.rm = TRUE)
  }
  
  # 2. Create 10-fold CV splits (fixed across all configurations for fair comparison)
  set.seed(seed)
  resampling = rsmp("cv", folds = 10)
  resampling$instantiate(preprocessed_task)
  
  cv_folds = list()
  for (fold_idx in 1:10) {
    cv_folds[[fold_idx]] = list(
      train = resampling$train_set(fold_idx),
      test = resampling$test_set(fold_idx)
    )
  }
  
  # 3. Get default parameters
  default_params = get_default_values(learner_id, preprocessed_task, task_type)
  
  # Clamp mtry for ranger
  if (learner_id == "ranger") {
    p = length(preprocessed_task$feature_names)
    default_params$mtry = min(default_params$mtry, p)
    default_params$mtry = max(1, default_params$mtry)
  }
  
  # 4. Train default model with 10-fold CV
  cat("[LOG] Training default model with 10-fold CV...\n")
  default_cv_result = run_cv_for_config(learner_id, learner_name, task_type, preprocessed_task,
                                        default_params, cv_folds)
  default_perf = default_cv_result$avg_metrics
  default_fold1_predictions = default_cv_result$fold1_predictions
  default_fold1_truth = default_cv_result$fold1_truth
  
  cat(sprintf("[LOG] Default CV performance: %s\n",
              paste(names(default_perf), "=", round(unlist(default_perf), 4), collapse = ", ")))
  
  # 5. Generate grid design
  search_space = get_search_space(learner_id, preprocessed_task)
  grid_dt = generate_grid_design(search_space, resolution = resolution, default_params = default_params)
  n_configs = nrow(grid_dt)
  
  cat(sprintf("[LOG] Running %d grid configurations with 10-fold CV (resolution=%d)...\n", n_configs, resolution))
  
  # 6. Set up parallel execution (handled dynamically inside the chunk loop to prevent memory leaks)
  plan(sequential)
  
  # 7. Run grid configurations
  run_config = function(i, prog = NULL) {
    config = as.list(grid_dt[i, ])
    
    # Ensure integer params are properly typed
    param_ids = search_space$ids()
    class_vec = search_space$class
    for (param in param_ids) {
      if (class_vec[[param]] == "ParamInt") {
        config[[param]] = as.integer(round(config[[param]]))
      }
    }
    
    # Clamp mtry for ranger
    if (learner_id == "ranger") {
      p = length(preprocessed_task$feature_names)
      config$mtry = min(config$mtry, p)
      config$mtry = max(1L, as.integer(config$mtry))
    }
    
    res = tryCatch({
      cv_result = run_cv_for_config(learner_id, learner_name, task_type, preprocessed_task,
                                    config, cv_folds)
      c(cv_result$avg_metrics,
        list(fold1_predictions = list(cv_result$fold1_predictions),
             fold1_truth = list(cv_result$fold1_truth),
             fold1_prob = list(cv_result$fold1_prob),
             error = NA_character_))
    }, error = function(e) {
      if (task_type == "classif") {
        list(perf_f1 = NA_real_, perf_acc = NA_real_, perf_bacc = NA_real_,
             perf_prec = NA_real_, perf_rec = NA_real_, perf_gmean = NA_real_, perf_kappa = NA_real_, 
             perf_mcc = NA_real_, perf_spec = NA_real_, perf_auc = NA_real_,
             fold1_predictions = list(NULL), fold1_truth = list(NULL), fold1_prob = list(NULL),
             error = e$message)
      } else {
        list(perf_mse = NA_real_, perf_rmse = NA_real_, perf_mae = NA_real_,
             perf_mape = NA_real_, perf_rmsle = NA_real_, perf_smape = NA_real_,
             perf_r2 = NA_real_, perf_adj_r2 = NA_real_,
             perf_max_ae = NA_real_, perf_med_ae = NA_real_,
             fold1_predictions = list(NULL), fold1_truth = list(NULL), fold1_prob = list(NULL),
             error = e$message)
      }
    })
    
    # (Progress logging removed from workers to prevent concurrent file write deadlocks)
    
    if (!is.null(prog)) prog(message = sprintf("%d/%d", i, n_configs))
    return(res)
  }
  
  # future.packages ile worker çekirdeklere mlr3learners bağımlılığı açıkça gönderiliyor
  # Initialize progress tracker file
  cat(sprintf("[%s] Starting Grid Search for Task: %s | Total: %d configs\n", Sys.time(), task$id, n_configs), 
      file = "progress_tracker.log", append = FALSE)
      
  # 7. Run grid configurations in chunks to avoid vector memory exhaustion (R process heap limit)
  chunk_size = 2000
  n_chunks = ceiling(n_configs / chunk_size)
  results_dt_list = list()
  total_valid = 0
  
  for (chunk_idx in 1:n_chunks) {
    start_i = (chunk_idx - 1) * chunk_size + 1
    end_i = min(chunk_idx * chunk_size, n_configs)
    chunk_indices = start_i:end_i
    n_chunk_configs = length(chunk_indices)
    
    cat(sprintf("[LOG] Running chunk %d/%d (configs %d to %d)...\n", chunk_idx, n_chunks, start_i, end_i))
    
    if (parallel) {
      plan(multisession)
    }
    
    with_progress({
      prog <- progressor(steps = n_chunk_configs)
      loop_results = future_lapply(
        chunk_indices,
        function(i) run_config(i, prog = prog),
        future.seed = TRUE,
        future.packages = c("mlr3", "mlr3learners", "paradox", "data.table", "progressr")
      )
    })
    
    err_vec = sapply(loop_results, function(x) x$error)
    total_valid = total_valid + sum(is.na(err_vec))
    
    fold1_preds_list = lapply(loop_results, function(x) x$fold1_predictions[[1]])
    fold1_truth_list = lapply(loop_results, function(x) x$fold1_truth[[1]])
    fold1_prob_list = lapply(loop_results, function(x) if (!is.null(x$fold1_prob)) x$fold1_prob[[1]] else NULL)
    
    grid_chunk = grid_dt[chunk_indices, ]
    
    if (task_type == "classif") {
      perf_cols = c("perf_f1", "perf_acc", "perf_bacc", "perf_prec", "perf_rec", "perf_gmean", "perf_kappa", "perf_mcc", "perf_spec", "perf_auc")
      metrics_matrix = do.call(rbind, lapply(loop_results, function(x) {
        unlist(x[perf_cols])
      }))
      
      chunk_dt = cbind(
        grid_chunk,
        data.table(metrics_matrix),
        error = err_vec
      )
      
      chunk_dt[, tunability_f1    := perf_f1 -    default_perf$perf_f1]
      chunk_dt[, tunability_acc   := perf_acc -   default_perf$perf_acc]
      chunk_dt[, tunability_bacc  := perf_bacc -  default_perf$perf_bacc]
      chunk_dt[, tunability_gmean := perf_gmean - default_perf$perf_gmean]
      chunk_dt[, tunability_kappa := perf_kappa - default_perf$perf_kappa]
      
    } else {
      perf_cols = c("perf_mse", "perf_rmse", "perf_mae", "perf_mape", "perf_rmsle", "perf_smape", "perf_r2", "perf_adj_r2", "perf_max_ae", "perf_med_ae")
      metrics_matrix = do.call(rbind, lapply(loop_results, function(x) {
        unlist(x[perf_cols])
      }))
      
      chunk_dt = cbind(
        grid_chunk,
        data.table(metrics_matrix),
        error = err_vec
      )
      
      chunk_dt[, tunability_mse  := default_perf$perf_mse - perf_mse]
      chunk_dt[, tunability_rmse := default_perf$perf_rmse - perf_rmse]
      chunk_dt[, tunability_mae  := default_perf$perf_mae - perf_mae]
      chunk_dt[, tunability_r2   := perf_r2 - default_perf$perf_r2]
    }
    
    chunk_dt[, fold1_predictions := fold1_preds_list]
    chunk_dt[, fold1_truth := fold1_truth_list]
    if (exists('fold1_prob_list')) {
      chunk_dt[, fold1_positive_prob := fold1_prob_list]
    }
    
    results_dt_list[[chunk_idx]] = chunk_dt
    
    # Log progress from the main process (safe from race conditions/file locks)
    pct = round((end_i / n_configs) * 100, 2)
    msg = sprintf("[%s] Task: %s | Config: %d/%d (%.2f%%) completed at %s\n", 
                  Sys.time(), task$id, end_i, n_configs, pct, format(Sys.time(), "%H:%M:%S"))
    cat(msg, file = "progress_tracker.log", append = TRUE)

    # Force clean memory for this chunk and shut down the parallel workers to release RAM
    rm(loop_results, err_vec, fold1_preds_list, fold1_truth_list, fold1_prob_list, metrics_matrix, chunk_dt)
    if (parallel) {
      plan(sequential)
    }
    gc()
  }
  
  cat("[LOG] Combining all chunk results...\n")
  results_dt = rbindlist(results_dt_list)
  rm(results_dt_list)
  gc()
  
  cat(sprintf("[LOG] Completed experiment. Valid configs: %d/%d\n", total_valid, n_configs))
  
  return(list(
    results                   = results_dt,
    default_performance       = default_perf,
    default_params            = default_params,
    default_fold1_predictions = default_fold1_predictions,
    default_fold1_truth       = default_fold1_truth,
    learner_id                = learner_id,
    task_id                   = task$id,
    task_type                 = task_type,
    resolution                = resolution,
    y_target_range            = y_range,
    class_mapping             = if (exists("class_mapping")) class_mapping else NULL
  ))
}

# Wrapper: run analysis for a given dataset (from datasets/), learner and resolution,
# save raw results to CSV and return a summary table.
run_and_save = function(dataset_name, learner_id, resolution = 3, seed = 123, parallel = TRUE, output_dir = "results") {
  # Locate dataset in datasets/ (recursive)
  ds_files = list.files("datasets", recursive = TRUE, full.names = TRUE)
  candidates = ds_files[grepl(dataset_name, basename(ds_files)) | grepl(dataset_name, ds_files)]
  task_obj = NULL
  if (length(candidates) >= 1) {
    path = candidates[1]
    obj = readRDS(path)
    if (is.list(obj) && !is.null(obj$task)) task_obj = obj$task
    else if (inherits(obj, "Task")) task_obj = obj
    else if (is.list(obj) && !is.null(obj$task_id) && !is.null(obj$task_name) && !is.null(obj$task)) task_obj = obj$task
    else stop(sprintf("Unable to interpret dataset RDS object at %s", path))
  } else {
    # Try mlr3 task name
    try_task = tryCatch(tsk(dataset_name), error = function(e) NULL)
    if (!is.null(try_task)) task_obj = try_task
    else stop("Dataset not found in datasets/ and not a built-in mlr3 task. Provide filename or task id.")
  }
  
  # Run experiment
  res_list = run_single_experiment(learner_id, task_obj, resolution = resolution, seed = seed, parallel = parallel)
  
  # Ensure output dir exists
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Save raw results to CSV
  task_type = res_list$task_type
  # Prefer numeric dataset id from filename (e.g., task_31_credit-g.rds)
  dataset_numeric_id = NA_character_
  if (exists('path') && !is.null(path)) {
    bn = basename(path)
    m = regmatches(bn, regexpr("task_([0-9]+)", bn))
    if (length(m) == 1 && nchar(m) > 0) {
      dataset_numeric_id = sub("task_([0-9]+)", "\\1", m)
    }
  }
  if (is.na(dataset_numeric_id) || dataset_numeric_id == "") {
    dataset_numeric_id = as.character(res_list$task_id)
  }
  csv_name = sprintf("%s_%s_%s_res%d.csv", task_type, dataset_numeric_id, learner_id, resolution)
  csv_path = file.path(output_dir, csv_name)
  # Flatten results table for CSV: fold1 predictions are list-columns; convert to JSON-like string
  dt = copy(res_list$results)
  # Remove tunability_ columns before saving
  tun_cols = grep('^tunability_', names(dt), value = TRUE)
  if (length(tun_cols) > 0) dt[, (tun_cols) := NULL]
  # Convert list columns to semicolon-joined strings
  if ("fold1_predictions" %in% names(dt)) dt[, fold1_predictions := sapply(fold1_predictions, function(x) if (is.null(x)) NA else paste(x, collapse = ";"))]
  if ("fold1_truth" %in% names(dt)) dt[, fold1_truth := sapply(fold1_truth, function(x) if (is.null(x)) NA else paste(x, collapse = ";"))]
  if ("fold1_positive_prob" %in% names(dt)) dt[, fold1_positive_prob := sapply(fold1_positive_prob, function(x) if (is.null(x)) NA else paste(round(as.numeric(x), 6), collapse = ";"))]
  data.table::fwrite(dt, csv_path)
  
  # Compute summary
  source("src/analyze_results.R")
  # For summary, use dataset_numeric_id if available, else res_list$task_id
  dataset_id = if (!is.null(dataset_numeric_id) && !is.na(dataset_numeric_id) && dataset_numeric_id != "") dataset_numeric_id else as.character(res_list$task_id)
  
  if (task_type == "classif") {
    default_perf = res_list$default_performance$perf_f1
    valid_dt = res_list$results[is.na(error)]
    best_f1 = max(valid_dt$perf_f1, na.rm = TRUE)
    max_tun = max(valid_dt$tunability_f1, na.rm = TRUE)
    max_disc = calculate_classif_discrepancy(res_list)
    summary = data.table::data.table(
      dataset = dataset_id,
      task_type = task_type,
      learner = learner_id,
      resolution = resolution,
      n_configs = nrow(res_list$results),
      default_perf_f1 = default_perf,
      best_perf_f1 = best_f1,
      max_tunability_f1 = max_tun,
      max_discrepancy = max_disc,
      csv = csv_path
    )
  } else {
    default_perf = res_list$default_performance$perf_r2
    valid_dt = res_list$results[is.na(error)]
    best_r2 = max(valid_dt$perf_r2, na.rm = TRUE)
    max_tun = max(valid_dt$tunability_r2, na.rm = TRUE)
    pred_mult = calculate_regression_predictive_multiplicity(res_list)
    delta_disc = calculate_regression_delta_discrepancy(res_list, delta = 0.05)
    summary = data.table::data.table(
      dataset = dataset_id,
      task_type = task_type,
      learner = learner_id,
      resolution = resolution,
      n_configs = nrow(res_list$results),
      default_perf_r2 = default_perf,
      best_perf_r2 = best_r2,
      max_tunability_r2 = max_tun,
      predictive_multiplicity = pred_mult,
      delta_discrepancy = delta_disc,
      csv = csv_path
    )
  }
  
  return(list(summary = summary, csv = csv_path, raw = res_list))
}
