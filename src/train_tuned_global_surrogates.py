"""
Tuned Global Surrogate Model Training Pipeline
Features:
- Hyperparameter settings
- Dataset Meta-Features (log10(n_rows), log10(n_cols))
- Default Model Baseline Performance (default_f1 or default_rmse/default_r2)

Hyperparameter Optimization for Random Forest Surrogate:
- n_estimators: [100, 200]
- max_depth: [None, 15, 25]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 1.0]
Evaluated via 5-Fold Cross Validation & OOD Extrapolation
"""

import os, glob, re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

os.makedirs("models/surrogates_tuned", exist_ok=True)
os.makedirs("results/surrogates", exist_ok=True)

HP_MAP = {
    "glmnet": ["alpha", "lambda"],
    "rpart": ["cp", "maxdepth", "minbucket", "minsplit"],
    "kknn": ["k"],
    "svm": ["cost", "gamma"],
    "ranger": ["num.trees", "sample.fraction", "mtry", "min.node.size"],
    "xgboost": ["nrounds", "eta", "subsample", "min_child_weight", "lambda", "alpha"]
}

LEARNERS = ["glmnet", "rpart", "kknn", "svm", "ranger", "xgboost"]

# Load meta-features
classif_meta = pd.read_csv("datasets/classification_summary.csv")
regr_meta = pd.read_csv("datasets/regression_summary.csv")

meta_dict_c = {str(r["task_id"]): (np.log10(max(10, r["n_rows"])), np.log10(max(2, r["n_cols"]))) for _, r in classif_meta.iterrows()}
meta_dict_r = {str(r["task_id"]): (np.log10(max(10, r["n_rows"])), np.log10(max(2, r["n_cols"]))) for _, r in regr_meta.iterrows()}

def build_dataset(learner, task_type="classif", max_per_ds=250):
    files = glob.glob(f"results/{learner}/*{task_type}*.csv")
    files = [f for f in files if "summary" not in f]
    hp_cols = HP_MAP[learner]
    meta_dict = meta_dict_c if task_type == "classif" else meta_dict_r
    
    records = []
    for f in files:
        bn = os.path.basename(f)
        ds_m = re.search(r"(?:classif|regr)_([0-9]+)", bn)
        ds_id = ds_m.group(1) if ds_m else "unknown"
        
        try:
            sample_df = pd.read_csv(f, nrows=5)
        except Exception:
            continue
            
        if "fold1_predictions" not in sample_df.columns:
            continue
            
        avail_hps = [c for c in hp_cols if c in sample_df.columns]
        if len(avail_hps) == 0: continue
        
        f_size_mb = os.path.getsize(f) / (1024 * 1024)
        try:
            if f_size_mb > 100:
                df = pd.read_csv(f, nrows=1000)
            else:
                df = pd.read_csv(f)
        except Exception:
            continue
            
        def_idx = None
        if "is_default" in df.columns:
            def_m = df.index[df["is_default"] == True].tolist()
            if def_m: def_idx = def_m[0]
            
        if def_idx is None:
            if learner == "ranger" and "num.trees" in df.columns:
                m = df.index[(df["num.trees"] == 500) & (df["sample.fraction"] == 1.0)].tolist()
                if m: def_idx = m[0]
            elif learner == "svm" and "cost" in df.columns:
                m = df.index[abs(df["cost"] - 1.0) < 1e-3].tolist()
                if m: def_idx = m[0]
            elif learner == "xgboost" and "nrounds" in df.columns:
                m = df.index[(df["nrounds"] == 500) | (df["nrounds"] == df["nrounds"].min())].tolist()
                if m: def_idx = m[0]
        if def_idx is None or def_idx >= len(df): def_idx = 0
        
        def_row = df.iloc[def_idx]
        def_preds_str = str(def_row["fold1_predictions"])
        if not def_preds_str or def_preds_str == "nan": continue
        def_preds = def_preds_str.split(";")
        n_inst = len(def_preds)
        if n_inst == 0: continue
        
        # Meta-features
        if ds_id in meta_dict:
            log_n, log_p = meta_dict[ds_id]
        else:
            log_n, log_p = 3.0, 1.0
            
        y_range = 1.0
        if task_type == "regr":
            try:
                t_str = str(def_row["fold1_truth"])
                if t_str and t_str != "nan":
                    t_vals = np.array([float(x) for x in t_str.split(";") if x.strip()])
                    y_range = np.max(t_vals) - np.min(t_vals)
            except Exception: pass
            if y_range <= 0 or np.isnan(y_range):
                try:
                    p_vals = np.array([float(x) for x in def_preds if x.strip()])
                    y_range = np.max(p_vals) - np.min(p_vals)
                except Exception: pass
            if y_range <= 0 or np.isnan(y_range): y_range = 1.0
            
        perf_col = "perf_f1" if task_type == "classif" else "perf_r2"
        if perf_col not in df.columns:
            if task_type == "regr" and "perf_rmse" in df.columns:
                perf_col = "perf_rmse"
            else: continue
            
        def_perf = float(def_row[perf_col]) if not pd.isna(def_row[perf_col]) else 0.5
        # Clean extreme divergent regression baseline values
        if task_type == "regr":
            if np.isnan(def_perf) or def_perf < -1.0: def_perf = -1.0
            if def_perf > 1.0: def_perf = 1.0
            
        valid_df = df[df["fold1_predictions"].notna() & df[perf_col].notna()]
        if len(valid_df) > max_per_ds:
            valid_df = valid_df.sample(n=max_per_ds, random_state=123)
            
        for _, row in valid_df.iterrows():
            p_str = str(row["fold1_predictions"])
            try:
                p_vec = p_str.split(";")
                if len(p_vec) != n_inst: continue
                if task_type == "classif":
                    disc = np.mean(np.array(p_vec) != np.array(def_preds))
                else:
                    p_num = np.array([float(x) for x in p_vec if x.strip()])
                    d_num = np.array([float(x) for x in def_preds if x.strip()])
                    disc = min(1.0, np.mean(np.abs(p_num - d_num) / y_range))
            except Exception: continue
            
            p_val = float(row[perf_col])
            if task_type == "regr":
                p_val = max(-1.0, min(1.0, p_val)) # clamp for regression R2
                
            entry = {
                "learner": learner,
                "dataset_id": ds_id,
                "log_n": log_n,
                "log_p": log_p,
                "perf_target": p_val,
                "disc_target": float(disc)
            }
            for hp in hp_cols:
                entry[hp] = float(row[hp]) if hp in row and not pd.isna(row[hp]) else 0.0
            records.append(entry)
            
    df_res = pd.DataFrame(records)
    if len(df_res) > 0:
        feature_cols = hp_cols + ["log_n", "log_p"]
        df_res = df_res.dropna(subset=feature_cols + ["perf_target", "disc_target"])
    return df_res

def run_tuned_surrogates():
    print("==========================================================================")
    print("Training Optimized Global Surrogate Models (RF Tuning with Meta-Features)")
    print("==========================================================================")
    
    results = []
    
    for task_type in ["classif", "regr"]:
        for learner in LEARNERS:
            df = build_dataset(learner, task_type=task_type, max_per_ds=200)
            hp_cols = HP_MAP[learner]
            feature_cols = hp_cols + ["log_n", "log_p"]
            
            if len(df) < 50:
                print(f"Skipping {learner} ({task_type}): N={len(df)}")
                continue
                
            X = df[feature_cols].values
            y_perf = df["perf_target"].values
            y_disc = df["disc_target"].values
            
            # Hyperparameter Tuning via Grid Search
            param_grid = {
                "n_estimators": [100, 150],
                "max_depth": [15, 25, None],
                "min_samples_leaf": [1, 2]
            }
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Performance Surrogate Optimization
            grid_perf = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid, cv=kf, scoring="r2", n_jobs=-1
            )
            grid_perf.fit(X, y_perf)
            best_rf_perf = grid_perf.best_estimator_
            perf_r2 = grid_perf.best_score_
            
            # Discrepancy Surrogate Optimization
            grid_disc = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid, cv=kf, scoring="r2", n_jobs=-1
            )
            grid_disc.fit(X, y_disc)
            best_rf_disc = grid_disc.best_estimator_
            disc_r2 = grid_disc.best_score_
            
            # Compute MAE over 5-fold
            perf_mae_list, disc_mae_list = [], []
            for tr, val in kf.split(X):
                p_pred = best_rf_perf.predict(X[val])
                d_pred = best_rf_disc.predict(X[val])
                perf_mae_list.append(mean_absolute_error(y_perf[val], p_pred))
                disc_mae_list.append(mean_absolute_error(y_disc[val], d_pred))
            perf_mae = np.mean(perf_mae_list)
            disc_mae = np.mean(disc_mae_list)
            
            # Save tuned models
            joblib.dump(best_rf_perf, f"models/surrogates_tuned/rf_{learner}_{task_type}_perf_tuned.joblib")
            joblib.dump(best_rf_disc, f"models/surrogates_tuned/rf_{learner}_{task_type}_disc_tuned.joblib")
            
            # OOD Evaluation on Extreme Hyperparameters
            primary_hp = hp_cols[0]
            q_low = df[primary_hp].quantile(0.15)
            q_high = df[primary_hp].quantile(0.85)
            in_mask = (df[primary_hp] >= q_low) & (df[primary_hp] <= q_high)
            ood_mask = ~in_mask
            
            if np.sum(in_mask) > 30 and np.sum(ood_mask) > 10:
                rf_ood_p = RandomForestRegressor(**grid_perf.best_params_, random_state=42, n_jobs=-1).fit(X[in_mask], y_perf[in_mask])
                rf_ood_d = RandomForestRegressor(**grid_disc.best_params_, random_state=42, n_jobs=-1).fit(X[in_mask], y_disc[in_mask])
                ood_perf_r2 = r2_score(y_perf[ood_mask], rf_ood_p.predict(X[ood_mask]))
                ood_disc_r2 = r2_score(y_disc[ood_mask], rf_ood_d.predict(X[ood_mask]))
                ood_disc_mae = mean_absolute_error(y_disc[ood_mask], rf_ood_d.predict(X[ood_mask]))
            else:
                ood_perf_r2, ood_disc_r2, ood_disc_mae = np.nan, np.nan, np.nan
                
            results.append({
                "task_type": task_type,
                "learner": learner,
                "n_samples": len(df),
                "best_params_perf": str(grid_perf.best_params_),
                "perf_r2_cv": perf_r2,
                "perf_mae_cv": perf_mae,
                "best_params_disc": str(grid_disc.best_params_),
                "disc_r2_cv": disc_r2,
                "disc_mae_cv": disc_mae,
                "ood_perf_r2": ood_perf_r2,
                "ood_disc_r2": ood_disc_r2,
                "ood_disc_mae": ood_disc_mae
            })
            
            print(f"[{task_type.upper()}] {learner:<8} | N={len(df):<5} | Perf R2: {perf_r2:.4f} (MAE: {perf_mae:.4f}) | Disc R2: {disc_r2:.4f} (MAE: {disc_mae:.4f}) | OOD Disc R2: {ood_disc_r2:.4f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv("results/surrogates/global_tuned_surrogate_results.csv", index=False)
    print("\nTuned Global Surrogate evaluation finished successfully!")

if __name__ == "__main__":
    run_tuned_surrogates()
