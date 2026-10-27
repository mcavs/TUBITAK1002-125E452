################################################################################
# download_datasets.R
# OpenML-CC18 (binary classif) ve OpenML-CTR23 (regresyon) veri setlerini
# mlr3oml araciligiyla indirip datasets/ klasorune kaydeder.
#
# Calistirmak icin: Rscript src/download_datasets.R
################################################################################

suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3oml)
})

# Maksimum kabul edilebilir satir sayisi (metadata'dan kontrol edilir – veri indirilmeden)
MAX_ROWS <- 200000

# ---------------------------------------------------------------------------- #
#  Task ID Listeleri
# ---------------------------------------------------------------------------- #

# OpenML-CC18 – Yalnizca binary siniflandirma task'lari (suite ID: 99)
classif_task_ids <- c(
  3,      # kr-vs-kp
  15,     # breast-w
  28,     # colic
  31,     # credit-g
  37,     # diabetes
  43,     # spambase
  45,     # sylvine
  49,     # tic-tac-toe
  219,    # electricity
  3021,   # sick
  3902,   # pc4
  3903,   # pc3
  3904,   # jm1
  3913,   # kc2
  3917,   # kc1
  3918,   # pc1
  7592,   # adult
  9910,   # Bioresponse
  9946,   # wdbc
  9952,   # phoneme
  9957,   # qsar-biodeg
  9971,   # ilpd
  9976,   # madelon
  9977,   # nomao
  9978,   # ozone-level-8hr
  10093,  # banknote-authentication
  10101,  # blood-transfusion-service-center
  14952,  # PhishingWebsites
  14954,  # cylinder-bands
  14965,  # bank-marketing
  125920, # dresses-sales
  146819, # climate-model-simulation-crashes
  146820, # wilt
  167119, # jungle_chess_2pcs_raw_endgame_complete
  167120, # numerai28.6
  167125, # Internet-Advertisements
  167141  # churn
)

# OpenML-CTR23 – Regresyon task'lari (suite ID: 353)
regr_task_ids <- c(
  167210, # Moneyball
  233211, # diamonds
  233212, # Allstate_Claims_Severity
  233213, # Buzzinsocialmedia_Twitter
  233214, # Santander_transaction_value
  233215, # Mercedes_Benz_Greener_Manufacturing
  317614, # Yolanda
  359929, # Airlines_DepDelay_10M (~10M satir)
  359930, # quake
  359932, # socmob
  359933, # space_ga
  359934, # tecator
  359935, # wine_quality
  359936, # elevators
  359937, # black_friday
  359938, # Brazilian_houses
  359939, # topo_2_1
  359940, # yprop_4_1
  359941, # OnlineNewsPopularity
  359942, # colleges
  359943, # nyc-taxi-green-dec-2016
  359944, # abalone
  359945, # us_crime
  359946, # pol
  359948, # SAT11-HAND-runtime-regression
  359949, # house_sales
  359950, # boston
  359951, # house_prices_nominal
  359952, # house_16H
  360932, # QSAR-TID-11
  360933, # QSAR-TID-10980
  360945  # MIP-2016-regression
)

# ---------------------------------------------------------------------------- #
#  Yardimci Fonksiyon
# ---------------------------------------------------------------------------- #

download_and_save <- function(task_ids,
                              out_dir,
                              type     = c("classif", "regr"),
                              max_rows = MAX_ROWS) {
  type <- match.arg(type)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  summary_records <- list()

  for (tid in task_ids) {
    # Zaten indirilmis mi kontrol et
    existing <- list.files(out_dir, pattern = sprintf("^task_%d_", tid), full.names = TRUE)
    if (length(existing) > 0) {
      info <- readRDS(existing[1])
      cat(sprintf("\n[Task %d] Zaten mevcut: %s (%d x %d)\n",
                  tid, info$task_name, nrow(info$data), ncol(info$data)))
      summary_records[[length(summary_records) + 1L]] <- data.frame(
        task_id   = tid,
        task_name = info$task_name,
        n_rows    = nrow(info$data),
        n_cols    = ncol(info$data),
        stringsAsFactors = FALSE
      )
      next
    }

    cat(sprintf("\n[Task %d] Indiriliyor...\n", tid))

    tryCatch({
      # 1. OMLTask nesnesini al (sadece metadata – veri inmez)
      oml_task <- mlr3oml::otsk(tid)

      # 2. Metadata'dan boyut kontrolu (veri indirilmeden)
      meta_nrow <- oml_task$nrow
      meta_ncol <- oml_task$ncol

      if (!is.null(meta_nrow) && !is.na(meta_nrow) && meta_nrow > max_rows) {
        cat(sprintf("  [ATLA] Metadata: %d satir > %d limit. Veri indirilmedi.\n",
                    meta_nrow, max_rows))
        next
      }

      # 3. Sinif sayisi kontrolu: classif icin metadata'dan kontrol
      if (type == "classif") {
        # task_type alanina bak
        task_type <- oml_task$task_type
        if (!grepl("Classification", task_type, ignore.case = TRUE)) {
          cat(sprintf("  [ATLA] Task type: %s\n", task_type))
          next
        }
      }

      # 4. Veriyi indir -> mlr3 task'a donustur
      task_mlr3 <- as_task(oml_task)

      # 5. Binary sinif kontrolu (classif icin)
      if (type == "classif") {
        n_classes <- length(task_mlr3$class_names)
        if (n_classes != 2L) {
          cat(sprintf("  [ATLA] %d sinif var – binary degil.\n", n_classes))
          next
        }
      }

      # 6. Gercek boyutlari al
      n_row <- task_mlr3$nrow
      n_col <- task_mlr3$ncol

      # 7. Kaydet
      data_df  <- task_mlr3$data()
      out_file <- file.path(out_dir, sprintf("task_%d_%s.rds", tid, task_mlr3$id))
      saveRDS(
        list(task = task_mlr3, data = data_df, task_id = tid, task_name = task_mlr3$id),
        out_file
      )

      cat(sprintf("  [OK]  %-45s -> %6d satir x %3d sutun\n",
                  task_mlr3$id, n_row, n_col))

      summary_records[[length(summary_records) + 1L]] <- data.frame(
        task_id   = tid,
        task_name = task_mlr3$id,
        n_rows    = n_row,
        n_cols    = n_col,
        stringsAsFactors = FALSE
      )

    }, error = function(e) {
      cat(sprintf("  [HATA] %s\n", conditionMessage(e)))
    })
  }

  if (length(summary_records) > 0L)
    do.call(rbind, summary_records)
  else
    data.frame()
}

# ---------------------------------------------------------------------------- #
#  Indirme
# ---------------------------------------------------------------------------- #

cat("==========================================================\n")
cat("   OpenML Veri Seti Indirme Scripti (v2 – metadata-first)\n")
cat("==========================================================\n\n")

cat("--- SINIFLANDIRMA (CC18 - Binary) ------------------------\n")
classif_summary <- download_and_save(
  task_ids = classif_task_ids,
  out_dir  = "datasets/classification",
  type     = "classif"
)

cat("\n--- REGRESYON (CTR23) ------------------------------------\n")
regr_summary <- download_and_save(
  task_ids = regr_task_ids,
  out_dir  = "datasets/regression",
  type     = "regr"
)

# ---------------------------------------------------------------------------- #
#  Ozet Tablosu ve k-Fold Onerisi
# ---------------------------------------------------------------------------- #

print_summary <- function(df, label) {
  if (nrow(df) == 0) {
    cat(sprintf("\n[%s] Hic veri seti indirilemedi.\n", label))
    return(invisible(NULL))
  }
  df_sorted <- df[order(df$n_rows), ]
  cat(sprintf("\n%-6s %-45s %8s %6s\n", "TaskID", "Isim", "Satir", "Sutun"))
  cat(strrep("-", 72), "\n")
  for (i in seq_len(nrow(df_sorted))) {
    cat(sprintf("%-6d %-45s %8d %6d\n",
                df_sorted$task_id[i], df_sorted$task_name[i],
                df_sorted$n_rows[i],  df_sorted$n_cols[i]))
  }
  cat(strrep("-", 72), "\n")
  cat(sprintf("Toplam: %d veri seti | Satir: min=%d, median=%.0f, max=%d\n",
              nrow(df), min(df$n_rows), median(df$n_rows), max(df$n_rows)))
}

suggest_k <- function(df, label) {
  if (nrow(df) == 0) return(invisible(NULL))
  med_n <- median(df$n_rows)
  # Literaturde kabul goren esikler:
  # < 1000 satir  -> 10-fold (veya LOOCV)
  # 1000-10000    -> 10-fold
  # > 10000       ->  5-fold (hesaplama maliyeti)
  k <- if (med_n < 10000) 10L else 5L
  cat(sprintf("  [%s] Median: %.0f satir => Onerilen k = %d-fold CV\n",
              label, med_n, k))
}

cat("\n\n==========================================================\n")
cat("   OZET\n")
cat("==========================================================\n")

cat("\n-- SINIFLANDIRMA ----------------------------------------\n")
print_summary(classif_summary, "Classif")

cat("\n-- REGRESYON --------------------------------------------\n")
print_summary(regr_summary, "Regr")

cat("\nk-Fold Cross-Validation Onerisi:\n")
suggest_k(classif_summary, "Classif")
suggest_k(regr_summary, "Regr")

# Ozet CSV kaydet
if (nrow(classif_summary) > 0)
  write.csv(classif_summary, "datasets/classification_summary.csv", row.names = FALSE)
if (nrow(regr_summary) > 0)
  write.csv(regr_summary, "datasets/regression_summary.csv", row.names = FALSE)

cat("\n[OK] Ozet CSV'ler datasets/ klasorune kaydedildi.\n")
cat("[OK] Indirme tamamlandi.\n")
