"""
Surrogate Model Training and Evaluation Pipeline (Probst et al. 2019 & TÜBİTAK 1002 İP 4)
This script trains Random Forest surrogate models for:
1. Performance prediction (F1-score / R2)
2. Multiplicity / Discrepancy prediction
across 6 model families (glmnet, rpart, kknn, svm, ranger, xgboost).

Validation Schemes:
- In-Domain Cross-Validation (5-Fold CV) to assess high-fidelity fit / surrogate accuracy
- Out-of-Distribution (OOD) / Extrapolation Test (trained on core range, tested on extreme / boundary hyperparameter values)
- Feature Importance (Gini / MDI) to quantify individual hyperparameter contributions
"""

import os, glob, re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

os.makedirs("models/surrogates", exist_ok=True)
os.makedirs("results/surrogates", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Hyperparameter feature mapping per learner
HP_MAP = {
    "glmnet": ["alpha", "lambda"],
    "rpart": ["cp", "maxdepth", "minbucket", "minsplit"],
    "kknn": ["k"],
    "svm": ["cost", "gamma"],
    "ranger": ["num.trees", "sample.fraction", "mtry", "min.node.size"],
    "xgboost": ["nrounds", "eta", "subsample", "min_child_weight", "lambda", "alpha"]
}

LEARNERS = ["glmnet", "rpart", "kknn", "svm", "ranger", "xgboost"]

def load_and_prep_data(learner, task_type="classif", max_rows_per_file=300):
    files = glob.glob(f"results/{learner}/*{task_type}*.csv")
    files = [f for f in files if "summary" not in f]
    hp_cols = HP_MAP[learner]
    
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
            
        # Check hp cols
        avail_hps = [c for c in hp_cols if c in sample_df.columns]
        if len(avail_hps) == 0:
            continue
            
        # Read full or sample
        f_size_mb = os.path.getsize(f) / (1024 * 1024)
        if f_size_mb > 100:
            df = pd.read_csv(f, nrows=1000)
        else:
            df = pd.read_csv(f)
            
        # Determine default row
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
        if def_idx is None or def_idx >= len(df):
            def_idx = 0
            
        def_row = df.iloc[def_idx]
        def_preds_str = str(def_row["fold1_predictions"])
        if not def_preds_str or def_preds_str == "nan":
            continue
            
        def_preds = def_preds_str.split(";")
        n_inst = len(def_preds)
        if n_inst == 0: continue
        
        # Calculate y_range for regression
        y_range = 1.0
        if task_type == "regr":
            try:
                t_str = str(def_row["fold1_truth"])
                if t_str and t_str != "nan":
                    t_vals = np.array([float(x) for x in t_str.split(";") if x.strip()])
                    y_range = np.max(t_vals) - np.min(t_vals)
            except Exception:
                pass
            if y_range <= 0 or np.isnan(y_range):
                try:
                    p_vals = np.array([float(x) for x in def_preds if x.strip()])
                    y_range = np.max(p_vals) - np.min(p_vals)
                except Exception:
                    pass
            if y_range <= 0 or np.isnan(y_range): y_range = 1.0
            
        # Target performance column
        perf_col = "perf_f1" if task_type == "classif" else "perf_r2"
        if perf_col not in df.columns:
            if task_type == "regr" and "perf_rmse" in df.columns:
                perf_col = "perf_rmse"
            else:
                continue
                
        valid_df = df[df["fold1_predictions"].notna() & df[perf_col].notna()]
        if len(valid_df) > max_rows_per_file:
            valid_df = valid_df.sample(n=max_rows_per_file, random_state=123)
            
        for _, row in valid_df.iterrows():
            p_str = str(row["fold1_predictions"])
            try:
                p_vec = p_str.split(";")
                if len(p_vec) != n_inst: continue
                
                if task_type == "classif":
                    disc = np.mean(np.array(p_vec) != np.array(def_preds))
                else:
                    # regression normalized discrepancy
                    p_num = np.array([float(x) for x in p_vec if x.strip()])
                    d_num = np.array([float(x) for x in def_preds if x.strip()])
                    disc = np.mean(np.abs(p_num - d_num) / y_range)
                    disc = min(1.0, disc) # clamp
            except Exception:
                continue
                
            entry = {
                "learner": learner,
                "dataset_id": ds_id,
                "perf_target": float(row[perf_col]),
                "disc_target": float(disc)
            }
            for hp in hp_cols:
                entry[hp] = float(row[hp]) if hp in row and not pd.isna(row[hp]) else np.nan
            records.append(entry)
            
    df_all = pd.DataFrame(records)
    if len(df_all) > 0:
        df_all = df_all.dropna(subset=hp_cols + ["perf_target", "disc_target"])
    return df_all

def train_surrogate_models():
    print("=================================================================")
    print("Training Surrogate Models (Random Forest) for 6 Model Families")
    print("=================================================================")
    
    cv_results = []
    ood_results = []
    importance_records = []
    
    for task_type in ["classif", "regr"]:
        print(f"\n--- TASK TYPE: {task_type.upper()} ---")
        
        for learner in LEARNERS:
            df = load_and_prep_data(learner, task_type=task_type, max_rows_per_file=250)
            hp_cols = HP_MAP[learner]
            
            if len(df) < 50:
                print(f"Skipping {learner} ({task_type}): Not enough samples ({len(df)})")
                continue
                
            X = df[hp_cols].values
            y_perf = df["perf_target"].values
            y_disc = df["disc_target"].values
            
            # 1. In-Domain 5-Fold Cross-Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=123)
            perf_r2_scores, perf_mae_scores = [], []
            disc_r2_scores, disc_mae_scores = [], []
            
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                
                # Fit RF surrogate for performance (tuned to capture high nonlinear interaction / high fidelity)
                rf_p = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
                rf_p.fit(X_tr, y_perf[train_idx])
                pred_p = rf_p.predict(X_val)
                perf_r2_scores.append(r2_score(y_perf[val_idx], pred_p))
                perf_mae_scores.append(mean_absolute_error(y_perf[val_idx], pred_p))
                
                # Fit RF surrogate for discrepancy
                rf_d = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
                rf_d.fit(X_tr, y_disc[train_idx])
                pred_d = rf_d.predict(X_val)
                disc_r2_scores.append(r2_score(y_disc[val_idx], pred_d))
                disc_mae_scores.append(mean_absolute_error(y_disc[val_idx], pred_d))
                
            # Full model training & save
            full_rf_p = RandomForestRegressor(n_estimators=150, min_samples_leaf=1, random_state=42, n_jobs=-1)
            full_rf_p.fit(X, y_perf)
            joblib.dump(full_rf_p, f"models/surrogates/rf_{learner}_{task_type}_perf.joblib")
            
            full_rf_d = RandomForestRegressor(n_estimators=150, min_samples_leaf=1, random_state=42, n_jobs=-1)
            full_rf_d.fit(X, y_disc)
            joblib.dump(full_rf_d, f"models/surrogates/rf_{learner}_{task_type}_disc.joblib")
            
            # Save feature importance
            for hp, imp_p, imp_d in zip(hp_cols, full_rf_p.feature_importances_, full_rf_d.feature_importances_):
                importance_records.append({
                    "task_type": task_type,
                    "learner": learner,
                    "hyperparameter": hp,
                    "importance_perf": imp_p,
                    "importance_disc": imp_d
                })
                
            # 2. Out-of-Distribution (OOD) Extrapolation Evaluation
            # Train on inner 70% range of first principal hyperparameter, test on extreme 30% outer boundaries
            primary_hp = hp_cols[0]
            val_threshold_low = df[primary_hp].quantile(0.15)
            val_threshold_high = df[primary_hp].quantile(0.85)
            
            in_domain_mask = (df[primary_hp] >= val_threshold_low) & (df[primary_hp] <= val_threshold_high)
            ood_mask = ~in_domain_mask
            
            if np.sum(in_domain_mask) > 30 and np.sum(ood_mask) > 10:
                X_in, X_ood = X[in_domain_mask], X[ood_mask]
                y_p_in, y_p_ood = y_perf[in_domain_mask], y_perf[ood_mask]
                y_d_in, y_d_ood = y_disc[in_domain_mask], y_disc[ood_mask]
                
                rf_ood_p = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
                rf_ood_p.fit(X_in, y_p_in)
                pred_ood_p = rf_ood_p.predict(X_ood)
                ood_r2_p = r2_score(y_p_ood, pred_ood_p)
                ood_mae_p = mean_absolute_error(y_p_ood, pred_ood_p)
                
                rf_ood_d = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
                rf_ood_d.fit(X_in, y_d_in)
                pred_ood_d = rf_ood_d.predict(X_ood)
                ood_r2_d = r2_score(y_d_ood, pred_ood_d)
                ood_mae_d = mean_absolute_error(y_d_ood, pred_ood_d)
            else:
                ood_r2_p, ood_mae_p = np.nan, np.nan
                ood_r2_d, ood_mae_d = np.nan, np.nan
                
            cv_results.append({
                "task_type": task_type,
                "learner": learner,
                "n_samples": len(df),
                "perf_r2_cv": np.mean(perf_r2_scores),
                "perf_mae_cv": np.mean(perf_mae_scores),
                "disc_r2_cv": np.mean(disc_r2_scores),
                "disc_mae_cv": np.mean(disc_mae_scores),
                "ood_perf_r2": ood_r2_p,
                "ood_perf_mae": ood_mae_p,
                "ood_disc_r2": ood_r2_d,
                "ood_disc_mae": ood_mae_d
            })
            
            print(f"[{task_type.upper()}] {learner:<8} | N={len(df):<5} | Perf R2: {np.mean(perf_r2_scores):.3f} | Disc R2: {np.mean(disc_r2_scores):.3f} | OOD Disc R2: {ood_r2_d:.3f}")

    df_cv = pd.DataFrame(cv_results)
    df_imp = pd.DataFrame(importance_records)
    
    df_cv.to_csv("results/surrogates/surrogate_validation_results.csv", index=False)
    df_imp.to_csv("results/surrogates/surrogate_feature_importance.csv", index=False)
    print("\nSurrogate model training & validation completed successfully!")

if __name__ == "__main__":
    train_surrogate_models()
