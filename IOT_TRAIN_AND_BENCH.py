import pandas as pd
import numpy as np
import time
import os
import joblib
import io
import warnings
import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def run_v7_1_academic_benchmark():
    print("Initializing V7.1 Master Academic Edge-Hardware Benchmark...\n")
    np.random.seed(42)
    
    # ==========================================
    # 1. DATA LOADING & DEFENSIVE CLEANING
    # ==========================================
    file_names = [
        'UNSW_2018_IoT_Botnet_Full5pc_1.csv',
        'UNSW_2018_IoT_Botnet_Full5pc_2.csv',
        'UNSW_2018_IoT_Botnet_Full5pc_3.csv',
        'UNSW_2018_IoT_Botnet_Full5pc_4.csv'
    ]
    
    print("Loading all dataset chunks. This may take a moment...")
    dataframes = [pd.read_csv(f, low_memory=False) for f in file_names if os.path.exists(f)]
    
    if not dataframes:
        raise ValueError("CRITICAL: No dataset chunks found. Check your filenames in this directory.")
        
    dataset = pd.concat(dataframes, ignore_index=True)
    print(f"Successfully stitched {len(dataframes)} files. Total raw packets: {len(dataset):,}\n")
    
    features = ['mean', 'stddev', 'bytes', 'seq']
    required = features + ['attack']
    
    missing = [c for c in required if c not in dataset.columns]
    if missing:
        raise ValueError(f"CRITICAL: Missing columns: {missing}")
        
    # Drop rows with any missing data 
    dataset = dataset.dropna(subset=features)
    for col in features:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
        
    dataset['attack'] = pd.to_numeric(dataset['attack'], errors='coerce')
    dataset = dataset.dropna(subset=required)
    dataset['attack'] = dataset['attack'].astype(int)
    
    X = dataset[features]
    y = dataset['attack']

    # ==========================================
    # 2. NESTED EVALUATION (Outer Holdout Split)
    # ==========================================
    X_train_outer, X_test, y_train_outer, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("--- Traffic Distribution (Outer Split) ---")
    print(f"Training Set : Safe ({len(y_train_outer[y_train_outer==0]):,}) | Attack ({len(y_train_outer[y_train_outer==1]):,})")
    print(f"Holdout Set  : Safe ({len(y_test[y_test==0]):,}) | Attack ({len(y_test[y_test==1]):,})\n")

    # ==========================================
    # 3. ROBUST BALANCER FUNCTION
    # ==========================================
    def balance_data(X_data, y_data):
        train_data = pd.concat([X_data, y_data], axis=1)
        safe = train_data[train_data['attack'] == 0]
        attack = train_data[train_data['attack'] == 1]
        
        if len(attack) > len(safe) and len(safe) > 0:
            attack_down = resample(attack, replace=False, n_samples=len(safe), random_state=42)
            balanced = pd.concat([safe, attack_down])
        elif len(safe) > len(attack) and len(attack) > 0:
            safe_down = resample(safe, replace=False, n_samples=len(attack), random_state=42)
            balanced = pd.concat([safe_down, attack])
        else:
            balanced = train_data.copy()
            
        balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        return balanced[features], balanced['attack']

    # Balance the full outer training set for the final model build
    print("Balancing training data to ensure fair evaluation...")
    X_train_bal, y_train_bal = balance_data(X_train_outer, y_train_outer)
    X_test_np = X_test.values
    test_batch = X_test_np[:100]

    # ==========================================
    # 4. ARCHITECTURE DEFINITIONS
    # ==========================================
    models = {
        "DNN (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(criterion='entropy', max_depth=5, n_estimators=100, random_state=42),
        "V7.1 Edge Shield": DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    }
    
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, 
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric='logloss', random_state=42
        )

    print(f"\n{'Model Architecture':<18} | {'Acc':<5} | {'CV 95% CI':<13} | {'AUC':<5} | {'Size(KB)':<8} | {'Train(s)':<8} | {'Lat(μs)':<8}")
    print("-" * 90)

    benchmark_results = []
    metadata = {}

    # ==========================================
    # 5. EXECUTION GAUNTLET
    # ==========================================
    for name, model in models.items():
        # Inner CV (5-Fold) using Robust Balancer
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_train_outer, y_train_outer):
            X_fold_train, y_fold_train = X_train_outer.iloc[train_idx], y_train_outer.iloc[train_idx]
            X_fold_val, y_fold_val = X_train_outer.iloc[val_idx], y_train_outer.iloc[val_idx]
            
            X_fold_bal, y_fold_bal = balance_data(X_fold_train, y_fold_train)
            
            model.fit(X_fold_bal, y_fold_bal)
            preds = model.predict(X_fold_val.values)
            cv_scores.append(f1_score(y_fold_val, preds, zero_division=0))
            
        # Calculate 95% Confidence Interval for the F1 Score
        stderr = np.std(cv_scores, ddof=1) / np.sqrt(len(cv_scores))
        ci95 = 1.96 * stderr
        f1_mean = np.mean(cv_scores)
        ci_string = f"[{f1_mean-ci95:.3f},{f1_mean+ci95:.3f}]"

        # Final Training on fully balanced Outer Set
        start_train = time.perf_counter()
        model.fit(X_train_bal, y_train_bal)
        train_time = time.perf_counter() - start_train
        
        preds = model.predict(X_test_np)
        probs = model.predict_proba(X_test_np)[:, 1] if hasattr(model, "predict_proba") else [0]*len(preds)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, probs) if hasattr(model, "predict_proba") else 0.0
        cm = confusion_matrix(y_test, preds)
        
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        size_kb = len(buffer.getvalue()) / 1024.0
        
        for _ in range(10): model.predict(test_batch) 
        start_lat = time.perf_counter_ns()
        for _ in range(100): model.predict(test_batch)
        end_lat = time.perf_counter_ns()
        
        num_packets = len(test_batch) * 100
        avg_latency_us = ((end_lat - start_lat) / num_packets) / 1000.0 
        
        print(f"{name:<18} | {acc:<5.3f} | {ci_string:<13} | {roc:<5.3f} | {size_kb:<8.2f} | {train_time:<8.3f} | {avg_latency_us:<8.2f}")
        
        benchmark_results.append({
            "Model": name, "Accuracy": acc, "F1_Mean": f1_mean, "Precision": prec, 
            "Recall": rec, "ROC-AUC": roc, "TrainTime_s": train_time, 
            "Latency_us": avg_latency_us, "Size_KB": size_kb
        })
        
        if name == "V7.1 Edge Shield":
            metadata = {
                "model": model, "features": features, "version": "V7.1",
                "training_date": str(datetime.datetime.now()),
                "dataset": "UNSW_2018_IoT_Botnet_Full",
                "class_distribution": {"Safe": int(len(y_train_outer[y_train_outer==0])), "Attack": int(len(y_train_outer[y_train_outer==1]))}
            }

    # ==========================================
    # 6. EXPORT ARTIFACTS
    # ==========================================
    pd.DataFrame(benchmark_results).to_csv("v7_1_benchmark_results.csv", index=False)
    joblib.dump(metadata, 'v7_1_master_shield.pkl')
    print("\nBenchmark Complete. Artifacts saved: 'v7_1_benchmark_results.csv' & 'v7_1_master_shield.pkl'.")

if __name__ == "__main__":
    run_v7_1_academic_benchmark()
