# Visualization of Surrogate Model Feature Importances (theme_bw, No Titles)
library(data.table)
library(ggplot2)
library(scales)

dir.create("figures", showWarnings = FALSE, recursive = TRUE)

df_imp <- fread("results/surrogates/surrogate_feature_importance.csv")

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

# Exclude kknn (single feature k = 1.0) for richer comparison
df_multi <- df_imp[learner != "kknn"]
df_multi[, param_label := paste0(learner, ": ", hyperparameter)]
df_multi[, task_label := ifelse(task_type == "classif", "Sınıflandırma", "Regresyon")]

# Order by discrepancy importance
df_multi <- df_multi[order(task_type, importance_disc)]

p_surr <- ggplot(df_multi, aes(x = reorder(param_label, importance_disc), y = importance_disc, fill = learner)) +
  geom_bar(stat = "identity", width = 0.7, color = "black", alpha = 0.85) +
  coord_flip() +
  facet_wrap(~ task_label, scales = "free_y") +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.0)) +
  labs(x = "Model ve Hiperparametre", y = "Vekil Model Önem Derecesi (Gini / MDI)") +
  scale_fill_brewer(palette = "Set1", name = "Model Ailesi:") +
  theme_academic_bw

ggsave("figures/surrogate_feature_importance.png", p_surr, width = 10.5, height = 7, dpi = 300)
ggsave("figures/surrogate_feature_importance.pdf", p_surr, width = 10.5, height = 7)

cat("Surrogate feature importance figure generated successfully!\n")
