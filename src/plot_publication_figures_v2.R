# Script to generate theme_bw, header-free publication figures
library(data.table)
library(ggplot2)
library(scales)
library(patchwork)

dir.create("figures", showWarnings = FALSE, recursive = TRUE)

model_colors <- c(
  "glmnet"  = "#E41A1C", # Red
  "rpart"   = "#377EB8", # Blue
  "kknn"    = "#4DAF4A", # Green
  "svm"     = "#984EA3", # Purple
  "ranger"  = "#FF7F00", # Orange
  "xgboost" = "#A65628"  # Brown
)

model_labels <- c(
  "glmnet"  = "Elastic Net (glmnet)",
  "rpart"   = "Decision Tree (rpart)",
  "kknn"    = "k-NN (kknn)",
  "svm"     = "SVM (svm)",
  "ranger"  = "Random Forest (ranger)",
  "xgboost" = "XGBoost (xgboost)"
)

# Strict theme_bw with NO TITLES
theme_academic_bw <- theme_bw(base_size = 12) +
  theme(
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    axis.title = element_text(face = "bold", size = 11),
    axis.text = element_text(color = "black", size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10),
    strip.text = element_text(face = "bold", size = 11),
    strip.background = element_rect(fill = "gray95")
  )

# ==============================================================================
# 1. SINIFLANDIRMA GRAFİKLERİ
# ==============================================================================
dt_model_classif <- fread("results/summary_metrics/model_discrepancy.csv")
dt_hp_classif    <- fread("results/summary_metrics/hp_discrepancy.csv")

dt_model_classif <- dt_model_classif[!is.na(discrepancy) & discrepancy <= 1]
dt_hp_classif    <- dt_hp_classif[!is.na(discrepancy) & discrepancy <= 1]

# Şekil 1: Sınıflandırma Model Discrepancy
cat("1. Sınıflandırma Model Discrepancy Grafiği...\n")
p_c1 <- ggplot(dt_model_classif, aes(x = factor(learner, levels = c("kknn", "svm", "rpart", "glmnet", "xgboost", "ranger")), 
                                     y = discrepancy, fill = learner)) +
  geom_boxplot(width = 0.5, alpha = 0.75, outlier.shape = NA, color = "black") +
  geom_jitter(width = 0.18, alpha = 0.6, size = 1.8, color = "gray25") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.5, fill = "white", color = "black") +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_x_discrete(labels = c(
    "kknn" = "k-NN", "svm" = "SVM", "rpart" = "Decision Tree",
    "glmnet" = "Elastic Net", "xgboost" = "XGBoost", "ranger" = "Random Forest"
  )) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), labels = percent_format(accuracy = 1)) +
  labs(x = "Model Ailesi", y = "Maksimum Discrepancy (δ)") +
  theme_academic_bw

ggsave("figures/classif_fig1_model_discrepancy.png", p_c1, width = 8.5, height = 5.2, dpi = 300)
ggsave("figures/classif_fig1_model_discrepancy.pdf", p_c1, width = 8.5, height = 5.2)

# Şekil 2: Sınıflandırma Hiperparametre Discrepancy (Faceted)
cat("2. Sınıflandırma HP Discrepancy Grafiği...\n")
dt_hp_classif[, learner_label := factor(learner, levels = c("glmnet", "rpart", "kknn", "svm", "ranger", "xgboost"),
                                        labels = c("Elastic Net", "Decision Tree", "k-NN", "SVM", "Random Forest", "XGBoost"))]

p_c2 <- ggplot(dt_hp_classif, aes(x = reorder(hyperparameter, discrepancy, FUN = median), y = discrepancy, fill = learner)) +
  geom_boxplot(alpha = 0.8, outlier.size = 1.2, color = "black") +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 2.8, color = "black") +
  facet_wrap(~ learner_label, scales = "free_x", nrow = 2) +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), labels = percent_format(accuracy = 1)) +
  labs(x = "Hiperparametre", y = "Discrepancy (δ)") +
  theme_academic_bw +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 9))

ggsave("figures/classif_fig2_hp_discrepancy.png", p_c2, width = 10.5, height = 6.5, dpi = 300)
ggsave("figures/classif_fig2_hp_discrepancy.pdf", p_c2, width = 10.5, height = 6.5)

# Şekil 3: Sınıflandırma Tunability vs Discrepancy
cat("3. Sınıflandırma Tunability vs Discrepancy Grafiği...\n")
tbl1_c <- dt_model_classif[, .(
  mean_disc = mean(discrepancy, na.rm = TRUE),
  sd_disc   = sd(discrepancy, na.rm = TRUE),
  mean_tun  = mean(tunability, na.rm = TRUE),
  sd_tun    = sd(tunability, na.rm = TRUE)
), by = learner]

p_c3 <- ggplot(dt_model_classif, aes(x = tunability, y = discrepancy, color = learner)) +
  geom_point(alpha = 0.45, size = 2.2) +
  geom_point(data = tbl1_c, aes(x = mean_tun, y = mean_disc, fill = learner), 
             color = "black", shape = 21, size = 5.5, stroke = 1.5) +
  geom_errorbar(data = tbl1_c, aes(x = mean_tun, y = mean_disc, ymin = mean_disc - sd_disc/2, ymax = mean_disc + sd_disc/2),
                width = 0.005, linewidth = 0.8, color = "gray30") +
  geom_errorbar(data = tbl1_c, aes(x = mean_tun, y = mean_disc, xmin = mean_tun - sd_tun/2, xmax = mean_tun + sd_tun/2),
                orientation = "y", width = 0.02, linewidth = 0.8, color = "gray30") +
  scale_color_manual(values = model_colors, labels = model_labels, name = "Model:") +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(x = "Ayarlanabilirlik (F1 Kazancı: Tuned - Default)", y = "Maksimum Discrepancy (δ)") +
  theme_academic_bw

ggsave("figures/classif_fig3_tunability_vs_discrepancy.png", p_c3, width = 9, height = 5.8, dpi = 300)
ggsave("figures/classif_fig3_tunability_vs_discrepancy.pdf", p_c3, width = 9, height = 5.8)

# ==============================================================================
# 2. REGRESYON GRAFİKLERİ
# ==============================================================================
dt_regr <- fread("results/summary_metrics/regr_model_discrepancy_clean.csv")

# Şekil 4: Regresyon Model Multiplicity Dağılımı (Normalized Absolute Deviation)
cat("4. Regresyon Multiplicity Grafiği...\n")
p_r1 <- ggplot(dt_regr, aes(x = factor(learner, levels = c("kknn", "svm", "glmnet", "rpart", "ranger", "xgboost")), 
                            y = multiplicity_clamped, fill = learner)) +
  geom_boxplot(width = 0.5, alpha = 0.75, outlier.shape = NA, color = "black") +
  geom_jitter(width = 0.18, alpha = 0.6, size = 1.8, color = "gray25") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.5, fill = "white", color = "black") +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_x_discrete(labels = c(
    "kknn" = "k-NN", "svm" = "SVM", "glmnet" = "Elastic Net",
    "rpart" = "Decision Tree", "ranger" = "Random Forest", "xgboost" = "XGBoost"
  )) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2), labels = percent_format(accuracy = 1)) +
  labs(x = "Model Ailesi", y = "Normalize Tahmin Tutarsızlığı (Multiplicity)") +
  theme_academic_bw

ggsave("figures/regr_fig1_multiplicity.png", p_r1, width = 8.5, height = 5.2, dpi = 300)
ggsave("figures/regr_fig1_multiplicity.pdf", p_r1, width = 8.5, height = 5.2)

# Şekil 5: Regresyon delta-Discrepancy (delta = 0.05) Dağılımı
cat("5. Regresyon delta-Discrepancy Grafiği...\n")
p_r2 <- ggplot(dt_regr, aes(x = factor(learner, levels = c("kknn", "glmnet", "svm", "ranger", "rpart", "xgboost")), 
                            y = delta_discrepancy, fill = learner)) +
  geom_boxplot(width = 0.5, alpha = 0.75, outlier.shape = NA, color = "black") +
  geom_jitter(width = 0.18, alpha = 0.6, size = 1.8, color = "gray25") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.5, fill = "white", color = "black") +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_x_discrete(labels = c(
    "kknn" = "k-NN", "glmnet" = "Elastic Net", "svm" = "SVM",
    "ranger" = "Random Forest", "rpart" = "Decision Tree", "xgboost" = "XGBoost"
  )) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2), labels = percent_format(accuracy = 1)) +
  labs(x = "Model Ailesi", y = "δ-Discrepancy (Sapma ≥ %5 Olan Gözlem Oranı)") +
  theme_academic_bw

ggsave("figures/regr_fig2_delta_discrepancy.png", p_r2, width = 8.5, height = 5.2, dpi = 300)
ggsave("figures/regr_fig2_delta_discrepancy.pdf", p_r2, width = 8.5, height = 5.2)

# Şekil 6: Regresyon Tunability (R2 Kazancı) vs Multiplicity Trade-off
cat("6. Regresyon Tunability vs Multiplicity Grafiği...\n")
tbl_r <- dt_regr[, .(
  mean_mult = mean(multiplicity_clamped, na.rm = TRUE),
  sd_mult   = sd(multiplicity_clamped, na.rm = TRUE),
  mean_tun  = mean(tunability_r2_clamped, na.rm = TRUE),
  sd_tun    = sd(tunability_r2_clamped, na.rm = TRUE)
), by = learner]

p_r3 <- ggplot(dt_regr, aes(x = tunability_r2_clamped, y = multiplicity_clamped, color = learner)) +
  geom_point(alpha = 0.45, size = 2.2) +
  geom_point(data = tbl_r, aes(x = mean_tun, y = mean_mult, fill = learner), 
             color = "black", shape = 21, size = 5.5, stroke = 1.5) +
  geom_errorbar(data = tbl_r, aes(x = mean_tun, y = mean_mult, ymin = pmax(0, mean_mult - sd_mult/2), ymax = pmin(1, mean_mult + sd_mult/2)),
                width = 0.01, linewidth = 0.8, color = "gray30") +
  geom_errorbar(data = tbl_r, aes(x = mean_tun, y = mean_mult, xmin = pmax(0, mean_tun - sd_tun/2), xmax = pmin(1, mean_tun + sd_tun/2)),
                orientation = "y", width = 0.02, linewidth = 0.8, color = "gray30") +
  scale_color_manual(values = model_colors, labels = model_labels, name = "Model:") +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(x = "Ayarlanabilirlik (R² Artışı: Tuned - Default)", y = "Normalize Tahmin Tutarsızlığı") +
  theme_academic_bw

ggsave("figures/regr_fig3_tunability_vs_multiplicity.png", p_r3, width = 9, height = 5.8, dpi = 300)
ggsave("figures/regr_fig3_tunability_vs_multiplicity.pdf", p_r3, width = 9, height = 5.8)

# ==============================================================================
# 3. KIYASLAMA GRAFİĞİ (Sınıflandırma vs Regresyon)
# ==============================================================================
cat("7. Kıyaslama Grafiği (Sınıflandırma vs Regresyon)...\n")
comp_classif <- dt_model_classif[, .(
  task_type = "Sınıflandırma",
  mean_discrepancy = mean(discrepancy, na.rm = TRUE),
  sd_discrepancy = sd(discrepancy, na.rm = TRUE)
), by = learner]

comp_regr <- dt_regr[, .(
  task_type = "Regresyon",
  mean_discrepancy = mean(multiplicity_clamped, na.rm = TRUE),
  sd_discrepancy = sd(multiplicity_clamped, na.rm = TRUE)
), by = learner]

comp_all <- rbind(comp_classif, comp_regr)

p_comp <- ggplot(comp_all, aes(x = factor(learner, levels = c("kknn", "svm", "rpart", "glmnet", "ranger", "xgboost")), 
                               y = mean_discrepancy, fill = task_type)) +
  geom_bar(stat = "identity", position = position_dodge(0.75), width = 0.65, color = "black") +
  geom_errorbar(aes(ymin = pmax(0, mean_discrepancy - sd_discrepancy/2), ymax = mean_discrepancy + sd_discrepancy/2),
                position = position_dodge(0.75), width = 0.25, linewidth = 0.7, color = "gray20") +
  scale_fill_manual(values = c("Sınıflandırma" = "#2B83BA", "Regresyon" = "#D7191C"), name = "Problem Türü:") +
  scale_x_discrete(labels = c(
    "kknn" = "k-NN", "svm" = "SVM", "rpart" = "Decision Tree",
    "glmnet" = "Elastic Net", "ranger" = "Random Forest", "xgboost" = "XGBoost"
  )) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.1)) +
  labs(x = "Model Ailesi", y = "Ortalama Tahmin Tutarsızlığı (Discrepancy / Multiplicity)") +
  theme_academic_bw

ggsave("figures/comparison_classif_vs_regr.png", p_comp, width = 9.5, height = 5.5, dpi = 300)
ggsave("figures/comparison_classif_vs_regr.pdf", p_comp, width = 9.5, height = 5.5)

cat("\nTüm grafikler theme_bw ve başlıksız (raporda altyazı verilmek üzere) başarıyla üretildi!\n")
