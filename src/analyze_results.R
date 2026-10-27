# TÜBİTAK 1002 - Results Analysis Script
# Implements:
#   - Sınıflandırma için geleneksel Predictive Multiplicity / Discrepancy
#   - Regresyon için Elif Serce & Mustafa Cavus (2026) metodolojisi:
#       * Predictive Multiplicity (Eq. 1)
#       * delta-Discrepancy (Eq. 3)

library(data.table)
library(mlr3)
library(mlr3learners)

# ======================================================================
# 1. Regresyon İçin Spesifik Matematiksel Fonksiyonlar (Serce & Cavus, 2026)
# ======================================================================

# Denklem (1): Predictive Multiplicity in Regression - OPTIMIZED single pass
# delta_{bar{theta}}(f) := max ( 1/n * sum( |f_theta_j(x_i) - f_theta_d(x_i)| / (y_max - y_min) ) )
calculate_regression_predictive_multiplicity = function(res_list) {
  dt = res_list$results
  valid_dt = dt[is.na(error)]
  if (nrow(valid_dt) == 0) return(NA_real_)
  
  y_range = res_list$y_target_range
  def_preds = res_list$default_fold1_predictions
  
  if (is.null(def_preds) || is.na(y_range) || y_range == 0) return(NA_real_)
  
  # OPTIMIZED: Vectorized computation
  multiplicities = vapply(valid_dt$fold1_predictions, function(j_preds) {
    if (is.null(j_preds) || length(j_preds) == 0) return(NA_real_)
    mean(abs(j_preds - def_preds) / y_range, na.rm = TRUE)
  }, FUN.VALUE = numeric(1))
  
  return(max(multiplicities, na.rm = TRUE))
}

# Denklem (3): delta-Discrepancy in Regression - OPTIMIZED single pass
# D_delta := max ( 1/|S| * sum( I[ (|f_theta_j(x_i) - f_theta_d(x_i)| / (y_max - y_min)) >= delta ] ) )
calculate_regression_delta_discrepancy = function(res_list, delta = 0.05) {
  dt = res_list$results
  valid_dt = dt[is.na(error)]
  if (nrow(valid_dt) == 0) return(NA_real_)
  
  y_range = res_list$y_target_range
  def_preds = res_list$default_fold1_predictions
  
  if (is.null(def_preds) || is.na(y_range) || y_range == 0) return(NA_real_)
  
  # OPTIMIZED: Vectorized computation with vapply for faster execution
  discrepancies = vapply(valid_dt$fold1_predictions, function(j_preds) {
    if (is.null(j_preds) || length(j_preds) == 0) return(NA_real_)
    
    # Normalized absolute deviations
    norm_deviations = abs(j_preds - def_preds) / y_range
    
    # Proportion meeting threshold
    mean(norm_deviations >= delta, na.rm = TRUE)
  }, FUN.VALUE = numeric(1))
  
  return(max(discrepancies, na.rm = TRUE))
}

# ======================================================================
# 2. Sınıflandırma İçin Geleneksel Fonksiyonlar
# ======================================================================
calculate_classif_discrepancy = function(res_list) {
  dt = res_list$results
  valid_dt = dt[is.na(error)]
  if (nrow(valid_dt) == 0) return(NA_real_)
  
  def_preds = res_list$default_fold1_predictions
  if (is.null(def_preds)) return(NA_real_)
  
  discs = sapply(valid_dt$fold1_predictions, function(j_preds) {
    if (is.null(j_preds)) return(NA_real_)
    mean(j_preds != def_preds)
  })
  return(max(discs, na.rm = TRUE))
}

# ======================================================================
# 3. Pilot Sonuçları Özeti ve Vekil Model Analiz Modülü
# ======================================================================
analyze_pilot_results = function(rds_path, delta_regr = 0.05) {
  cat("\n======================================================================\n")
  cat("Analyzing Results from:", rds_path, "\n")
  cat("======================================================================\n")
  
  results = readRDS(rds_path)
  
  for (m in names(results)) {
    cat(sprintf("\n--- Model: %s ---\n", m))
    res_list = results[[m]]
    dt = res_list$results
    default_perf = res_list$default_performance
    task_type = res_list$task_type
    
    valid_dt = dt[is.na(error)]
    if (nrow(valid_dt) == 0) {
      cat("[WARN] No valid runs found.\n")
      next
    }
    
    if (task_type == "classif") {
      cat("[Performance & Multiplicity Metrics]\n")
      cat(sprintf("  - Default Macro F1:  %f\n", default_perf$perf_f1))
      cat(sprintf("  - Max Tuned F1:      %f\n", max(valid_dt$perf_f1, na.rm = TRUE)))
      cat(sprintf("  - Max Tunability F1: %f\n", max(valid_dt$tunability_f1, na.rm = TRUE)))
      
      # Sınıflandırma Discrepancy çıktısı
      max_disc = calculate_classif_discrepancy(res_list)
      cat(sprintf("  - Max Discrepancy:   %f\n", max_disc))
      
      # Hiperparametre sütunlarını bul
      ignore_cols = c("perf_f1", "perf_acc", "perf_bacc", "perf_prec", "perf_rec", "perf_gmean", "perf_kappa",
                      "tunability_f1", "tunability_acc", "tunability_bacc", "tunability_gmean", "tunability_kappa",
                      "error", "fold1_predictions", "fold1_truth")
      hps = setdiff(names(valid_dt), ignore_cols)
      
      cat("\nSurrogate Tree for Performance (F1):\n")
      task_surr = TaskRegr$new(id = paste0(m, "_f1"), backend = valid_dt[, c(hps, "perf_f1"), with = FALSE], target = "perf_f1")
      surrogate = lrn("regr.rpart", cp = 0.01)$train(task_surr)
      print(surrogate$model)
      
    } else {
      # REGRESYON BLOĞU
      cat("[Performance & Multiplicity Metrics]\n")
      cat(sprintf("  - Default R-squared:   %f\n", default_perf$perf_r2))
      cat(sprintf("  - Max Tuned R-squared: %f\n", max(valid_dt$perf_r2, na.rm = TRUE)))
      cat(sprintf("  - Max Tunability R2:   %f\n", max(valid_dt$tunability_r2, na.rm = TRUE)))
      
      # Yeni Regresyon Metriklerinin Çalıştırılması
      pred_mult = calculate_regression_predictive_multiplicity(res_list)
      delta_disc = calculate_regression_delta_discrepancy(res_list, delta = delta_regr)
      
      cat(sprintf("  - Predictive Multiplicity (Eq. 1): %f\n", pred_mult))
      cat(sprintf("  - delta-Discrepancy (Eq. 3, delta=%s): %f\n", delta_regr, delta_disc))
      
      ignore_cols = c("perf_mse", "perf_rmse", "perf_mae", "perf_mape", "perf_r2", "perf_adj_r2", "perf_max_ae", "perf_med_ae",
                      "tunability_mse", "tunability_rmse", "tunability_mae", "tunability_r2",
                      "error", "fold1_predictions", "fold1_truth")
      hps = setdiff(names(valid_dt), ignore_cols)
      
      cat("\nSurrogate Tree for Performance (R-squared):\n")
      task_surr = TaskRegr$new(id = paste0(m, "_r2"), backend = valid_dt[, c(hps, "perf_r2"), with = FALSE], target = "perf_r2")
      surrogate = lrn("regr.rpart", cp = 0.01)$train(task_surr)
      print(surrogate$model)
    }
  }
}

# Betik doğrudan çalıştırıldığında pilot verilerini test eder
if (file.exists("results/pilot_classif_sonar.rds")) {
  analyze_pilot_results("results/pilot_classif_sonar.rds")
}
if (file.exists("results/pilot_regr_mtcars.rds")) {
  # Regresyon pilot testi için deltayı 0.05 (yüzde 5) alarak çalıştırıyoruz
  analyze_pilot_results("results/pilot_regr_mtcars.rds", delta_regr = 0.05)
}