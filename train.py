"""
Edge Shield: Cross-Domain Training & Evaluation Pipeline
Author: Arna Nandi
Description: Reproducibility script for the cross-domain IDS model. 
Aggregates UNSW-2018 and CIC-IoT-2023 datasets, applies memory downcasting, 
and executes a 5-Fold Stratified CV with in-fold SMOTE balancing.

Note: The pre-trained production model (edge_shield_model.pkl) was generated 
using this pipeline on a 15GB Kaggle environment. 
"""

import time
import pickle
import numpy as np
import pandas as pd
from sys import getsizeof
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Configuration & Hyperparameters
UNSW_CSV = "data/unsw_2018_extract.csv"
CIC_CSV = "data/cic_iot_2023_extract.csv"
FEATURES = ['mean_iat', 'jitter_variance', 'payload_entropy', 'tcp_state']
TARGET = 'label'
MAX_DEPTH = 5
RANDOM_SEED = 42

def calculate_model_size(model):
    """Calculates the serialized memory footprint of the model in KB."""
    serialized_model = pickle.dumps(model)
    return len(serialized_model) / 1024

def downcast_matrix(df):
    """
    Enforces the Kaggle memory constraints by downcasting float64 to float32.
    Essential for processing 3.6+ million rows on constrained RAM.
    """
    print("[*] Downcasting memory footprint...")
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"    Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    return df

def main():
    print("[*] Loading Cross-Domain Datasets...")
    
    try:
        # Load and concatenate datasets
        df_unsw = pd.read_csv(UNSW_CSV)
        df_cic = pd.read_csv(CIC_CSV)
        df = pd.concat([df_unsw, df_cic], ignore_index=True)
        
        # Apply strict memory downcasting
        df = downcast_matrix(df)
        
        X = df[FEATURES].values
        y = df[TARGET].values
        
    except FileNotFoundError:
        print("[!] Datasets not found locally. Proceeding with matrix simulation for pipeline validation...")
        # Fallback simulation to validate pipeline logic if CSVs are not present
        np.random.seed(RANDOM_SEED)
        X = np.random.rand(3668045, 4)  # 3.6M rows simulating combined datasets
        y = np.random.randint(0, 2, 3668045)

    # 2. Initialize cross-validation and SMOTE
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    smote = SMOTE(random_state=RANDOM_SEED)
    
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'latency': []}
    best_model = None
    best_f1 = 0

    print(f"\n[*] Starting 5-Fold Cross-Validation (Model: Decision Tree, Depth: {MAX_DEPTH})")
    
    # 3. The Validation Gauntlet
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply SMOTE strictly inside the training fold (Zero Data Leakage)
        print(f"    Fold {fold}: Applying in-fold SMOTE balancing...")
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Initialize and train the constrained model
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=MAX_DEPTH, random_state=RANDOM_SEED)
        clf.fit(X_train_res, y_train_res)

        # 4. Latency Benchmarking (Micro-batch stress test)
        start_time = time.perf_counter()
        y_pred = clf.predict(X_test)
        end_time = time.perf_counter()
        
        # Calculate per-packet latency in microseconds
        latency_us = ((end_time - start_time) / len(X_test)) * 1e6

        # Calculate Classification Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)

        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['roc_auc'].append(roc)
        metrics['latency'].append(latency_us)
        
        print(f"    Fold {fold} - F1: {f1:.4f} | Latency: {latency_us:.3f} µs")

        # Save the champion model
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf

    # 5. Final Reporting
    print("\n" + "="*50)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
    print(f"Precision: {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
    print(f"Recall:    {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
    print(f"F1-Score:  {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
    print(f"ROC-AUC:   {np.mean(metrics['roc_auc']):.4f} ± {np.std(metrics['roc_auc']):.4f}")
    print("-" * 50)
    print(f"Mean Inference Latency: {np.mean(metrics['latency']):.3f} µs per packet")
    print(f"Serialized Memory Size: {calculate_model_size(best_model):.2f} KB")
    print("="*50)

    # Note: For GitHub documentation purposes, we save the localized run. 
    # The actual production model is pulled from the Kaggle environment.
    with open('models/local_edge_shield_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("[*] Local validation model saved to 'models/local_edge_shield_model.pkl'")

if __name__ == "__main__":
    main()
