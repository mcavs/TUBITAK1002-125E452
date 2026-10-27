# Fast vectorized computation of Regression Discrepancy & Multiplicity
import glob, os, re
import pandas as pd
import numpy as np

regr_files_pattern = "results/*/*regr*.csv"
learners = ["glmnet", "rpart", "kknn", "svm", "ranger", "xgboost"]

records = []
hp_records = []

for l in learners:
    files = [f for f in glob.glob(f"results/{l}/*regr*.csv") if "summary" not in f]
    print(f"Learner: {l:<8} | Total Regr Files: {len(files)}")
    
    for f in files:
        bn = os.path.basename(f)
        ds_m = re.search(r"regr_([0-9]+)", bn)
        ds_id = ds_m.group(1) if ds_m else "unknown"
        
        # Read header only to check columns
        try:
            sample_df = pd.read_csv(f, nrows=5)
        except Exception:
            continue
            
        if "fold1_predictions" not in sample_df.columns:
            continue
            
        # If file is too large (> 150MB), sample rows to avoid RAM exhaustion
        file_mb = os.path.getsize(f) / (1024 * 1024)
        try:
            if file_mb > 150:
                # read 1500 rows
                df = pd.read_csv(f, nrows=1500)
            else:
                df = pd.read_csv(f)
        except Exception:
            continue
            
        # Determine default row
        def_idx = None
        if "is_default" in df.columns:
            def_matches = df.index[df["is_default"] == True].tolist()
            if def_matches:
                def_idx = def_matches[0]
                
        if def_idx is None:
            # Fallback based on learner defaults
            if l == "ranger" and "num.trees" in df.columns:
                matches = df.index[(df["num.trees"] == 500) & (df["sample.fraction"] == 1.0)].tolist()
                if matches: def_idx = matches[0]
            elif l == "svm" and "cost" in df.columns:
                matches = df.index[abs(df["cost"] - 1.0) < 1e-3].tolist()
                if matches: def_idx = matches[0]
            elif l == "xgboost" and "nrounds" in df.columns:
                matches = df.index[(df["nrounds"] == 500) | (df["nrounds"] == df["nrounds"].min())].tolist()
                if matches: def_idx = matches[0]
                
        if def_idx is None or def_idx >= len(df):
            def_idx = 0
            
        def_row = df.iloc[def_idx]
        def_preds_str = str(def_row["fold1_predictions"])
        if not def_preds_str or def_preds_str == "nan":
            continue
            
        try:
            def_preds = np.array([float(x) for x in def_preds_str.split(";") if x.strip()])
        except Exception:
            continue
            
        n_inst = len(def_preds)
        if n_inst == 0:
            continue
            
        # Target range
        y_range = 1.0
        if "fold1_truth" in df.columns and str(def_row["fold1_truth"]) != "nan":
            try:
                truth_vals = np.array([float(x) for x in str(def_row["fold1_truth"]).split(";") if x.strip()])
                y_range = np.max(truth_vals) - np.min(truth_vals)
            except Exception:
                pass
        if y_range <= 0 or np.isnan(y_range):
            y_range = np.max(def_preds) - np.min(def_preds)
        if y_range <= 0 or np.isnan(y_range):
            y_range = 1.0
            
        def_rmse = float(def_row["perf_rmse"]) if "perf_rmse" in df.columns and not pd.isna(def_row["perf_rmse"]) else np.nan
        def_r2 = float(def_row["perf_r2"]) if "perf_r2" in df.columns and not pd.isna(def_row["perf_r2"]) else np.nan
        
        # Calculate multiplicity across rows
        multiplicities = []
        delta_discrepancies = []
        
        # Filter valid rows
        valid_rows = df[df["fold1_predictions"].notna()]
        # sample at most 500 configurations for speed if still large
        if len(valid_rows) > 500:
            valid_rows = valid_rows.sample(n=500, random_state=123)
            
        for _, row in valid_rows.iterrows():
            p_str = str(row["fold1_predictions"])
            try:
                p_vec = np.array([float(x) for x in p_str.split(";") if x.strip()])
                if len(p_vec) == n_inst:
                    norm_diff = np.abs(p_vec - def_preds) / y_range
                    multiplicities.append(np.mean(norm_diff))
                    delta_discrepancies.append(np.mean(norm_diff >= 0.05))
            except Exception:
                continue
                
        if not multiplicities:
            continue
            
        max_mult = np.max(multiplicities)
        mean_mult = np.mean(multiplicities)
        max_delta = np.max(delta_discrepancies)
        
        best_rmse = df["perf_rmse"].min() if "perf_rmse" in df.columns else np.nan
        best_r2 = df["perf_r2"].max() if "perf_r2" in df.columns else np.nan
        
        tunability_r2 = max(0.0, best_r2 - def_r2) if not np.isnan(best_r2) and not np.isnan(def_r2) else np.nan
        
        records.append({
            "learner": l,
            "dataset_id": ds_id,
            "multiplicity": max_mult,
            "delta_discrepancy": max_delta,
            "mean_multiplicity": mean_mult,
            "default_rmse": def_rmse,
            "best_rmse": best_rmse,
            "default_r2": def_r2,
            "best_r2": best_r2,
            "tunability_r2": tunability_r2,
            "n_inst": n_inst
        })

df_regr = pd.DataFrame(records)
os.makedirs("results/summary_metrics", exist_ok=True)
df_regr.to_csv("results/summary_metrics/regr_model_discrepancy.csv", index=False)

summary = df_regr.groupby("learner").agg(
    mean_multiplicity=("multiplicity", "mean"),
    sd_multiplicity=("multiplicity", "std"),
    mean_delta_disc=("delta_discrepancy", "mean"),
    sd_delta_disc=("delta_discrepancy", "std"),
    mean_tunability_r2=("tunability_r2", "mean"),
    sd_tunability_r2=("tunability_r2", "std"),
    n_datasets=("dataset_id", "count")
).reset_index()

summary.to_csv("results/summary_metrics/table_regr_model_summary.csv", index=False)
print("\n=== Fast Regression Metrics Summary Table ===")
print(summary.to_string(index=False))
