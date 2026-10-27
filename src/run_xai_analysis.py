"""
DALEX XAI Pipeline for Trained Surrogate Models
Computes:
1. Permutation Feature Importance (PFI) using DALEX
2. Partial Dependence Plots (PDP) using DALEX / model_profile
Generates theme_bw academic figures for each model family and task type.
"""

import os, glob, re
import pandas as pd
import numpy as np
import dalex as dx
import joblib
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.abspath("."))
from src.train_tuned_global_surrogates import build_dataset, HP_MAP

os.makedirs("figures/xai_pfi", exist_ok=True)
os.makedirs("figures/xai_pdp", exist_ok=True)
os.makedirs("results/xai", exist_ok=True)

# Styling matplotlib to look strictly like theme_bw
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#E5E5E5'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def run_dalex_analysis(learner="glmnet", task_type="classif", max_per_ds=100):
    print(f"--> Running DALEX XAI for {learner.upper()} ({task_type.upper()})...")
    df = build_dataset(learner, task_type=task_type, max_per_ds=max_per_ds)
    hp_cols = HP_MAP[learner]
    feature_cols = hp_cols + ["log_n", "log_p", "def_perf"]
    
    X = df[feature_cols].copy()
    y_perf = df["perf_target"].values
    y_disc = df["disc_target"].values
    
    # Load tuned surrogates
    perf_model_path = f"models/surrogates_tuned/rf_{learner}_{task_type}_perf_tuned.joblib"
    disc_model_path = f"models/surrogates_tuned/rf_{learner}_{task_type}_disc_tuned.joblib"
    
    if not os.path.exists(perf_model_path) or not os.path.exists(disc_model_path):
        print(f"Models not found for {learner}")
        return None
        
    model_perf = joblib.load(perf_model_path)
    model_disc = joblib.load(disc_model_path)
    
    # DALEX Explainers
    exp_perf = dx.Explainer(model_perf, X, y_perf, label=f"{learner}_{task_type}_perf", verbose=False)
    exp_disc = dx.Explainer(model_disc, X, y_disc, label=f"{learner}_{task_type}_disc", verbose=False)
    
    # 1. Permutation Feature Importance (PFI)
    pfi_perf = exp_perf.model_parts(loss_function="rmse", random_state=42)
    pfi_disc = exp_disc.model_parts(loss_function="rmse", random_state=42)
    
    # Save PFI data
    res_pfi_p = pfi_perf.result[~pfi_perf.result["variable"].isin(["_baseline_", "_full_model_"])].copy()
    res_pfi_d = pfi_disc.result[~pfi_disc.result["variable"].isin(["_baseline_", "_full_model_"])].copy()
    
    # Plot PFI Comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    
    # Perf PFI
    res_pfi_p = res_pfi_p.sort_values(by="dropout_loss")
    axes[0].barh(res_pfi_p["variable"], res_pfi_p["dropout_loss"], color="#2B83BA", edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Dropout Loss (RMSE Artışı)")
    axes[0].set_ylabel("Tahminci / Hiperparametre")
    axes[0].set_title(f"a) Performans Kestirimi ({'F1' if task_type=='classif' else 'R²'})", fontsize=11, fontweight="bold")
    
    # Disc PFI
    res_pfi_d = res_pfi_d.sort_values(by="dropout_loss")
    axes[1].barh(res_pfi_d["variable"], res_pfi_d["dropout_loss"], color="#D7191C", edgecolor="black", alpha=0.85)
    axes[1].set_xlabel("Dropout Loss (RMSE Artışı)")
    axes[1].set_title("b) Tutarsızlık Kestirimi (Discrepancy)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    pfi_fig_path = f"figures/xai_pfi/pfi_{learner}_{task_type}.png"
    plt.savefig(pfi_fig_path, dpi=300)
    plt.close()
    
    # 2. Partial Dependence Profiles (PDP) for Hyperparameters
    pdp_perf = exp_perf.model_profile(variables=hp_cols, verbose=False)
    pdp_disc = exp_disc.model_profile(variables=hp_cols, verbose=False)
    
    n_hps = len(hp_cols)
    n_cols_plot = min(4, n_hps)
    n_rows_plot = int(np.ceil(n_hps / n_cols_plot))
    
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.2 * n_cols_plot, 3.6 * n_rows_plot), squeeze=False)
    
    res_pdp_p = pdp_perf.result
    res_pdp_d = pdp_disc.result
    
    for idx, hp in enumerate(hp_cols):
        r_idx = idx // n_cols_plot
        c_idx = idx % n_cols_plot
        ax = axes[r_idx, c_idx]
        
        sub_p = res_pdp_p[res_pdp_p["_vname_"] == hp]
        sub_d = res_pdp_d[res_pdp_d["_vname_"] == hp]
        
        line1 = ax.plot(sub_p["_x_"], sub_p["_yhat_"], color="#2B83BA", linewidth=2.0, label="Performans")
        ax.set_xlabel(hp, fontweight="bold")
        ax.set_ylabel("Beklenen Performans", color="#2B83BA", fontweight="bold")
        ax.tick_params(axis='y', labelcolor="#2B83BA")
        
        # Twin axis for discrepancy
        ax2 = ax.twinx()
        line2 = ax2.plot(sub_d["_x_"], sub_d["_yhat_"], color="#D7191C", linewidth=2.0, linestyle="--", label="Discrepancy")
        ax2.set_ylabel("Beklenen Discrepancy", color="#D7191C", fontweight="bold")
        ax2.tick_params(axis='y', labelcolor="#D7191C")
        ax2.grid(False) # avoid dual grid clutter
        
    # Hide unused subplots
    for idx in range(n_hps, n_rows_plot * n_cols_plot):
        r_idx = idx // n_cols_plot
        c_idx = idx % n_cols_plot
        axes[r_idx, c_idx].set_visible(False)
        
    plt.tight_layout()
    pdp_fig_path = f"figures/xai_pdp/pdp_{learner}_{task_type}.png"
    plt.savefig(pdp_fig_path, dpi=300)
    plt.close()
    
    return {
        "learner": learner,
        "task_type": task_type,
        "pfi_fig": pfi_fig_path,
        "pdp_fig": pdp_fig_path,
        "top_pfi_perf": res_pfi_p.iloc[-1]["variable"],
        "top_pfi_disc": res_pfi_d.iloc[-1]["variable"],
        "pfi_perf_data": res_pfi_p,
        "pfi_disc_data": res_pfi_d
    }

if __name__ == "__main__":
    # Test sample run for GLMNET and RANGER in Classification
    res_glmnet = run_dalex_analysis("glmnet", "classif", max_per_ds=80)
    res_ranger = run_dalex_analysis("ranger", "classif", max_per_ds=80)
    print("\nSample DALEX analysis completed successfully!")
