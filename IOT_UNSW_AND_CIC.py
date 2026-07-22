import pandas as pd
import numpy as np
import time
import joblib
import logging
import gc
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE 

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==========================================
# 1. KAGGLE AUTO-DISCOVERY & LOGGING
# ==========================================
working_dir = Path('/kaggle/working')
input_base = Path('/kaggle/input') 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(working_dir / "forge_kaggle_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
np.random.seed(42)

logger.info("==================================================")
logger.info("THE FORGE : 15GB REDLINE IEEE BENCHMARK PIPELINE")
logger.info("==================================================")

# ==========================================
# 2. BULLETPROOF INGESTION & DOWNCASTING
# ==========================================
UNIFIED_FEATURES = ['unified_mean_iat', 'unified_jitter_std', 'unified_byte_vol', 'unified_state_flag']
TARGET = 'unified_target'

def optimize_dtypes(df):
    """Crushes base memory footprint to allow for massive 20M+ packet scaling."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            if col == TARGET:
                df[col] = df[col].astype('int8')
            else:
                df[col] = df[col].astype('int32')
    return df

def process_file(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
        if df.empty: return pd.DataFrame()

        col_map = {str(c).lower().strip(): c for c in df.columns}
        unified_df = pd.DataFrame()
        
        if 'label' in col_map and 'iat' in col_map:
            unified_df['unified_mean_iat'] = pd.to_numeric(df[col_map['iat']], errors='coerce')
            unified_df['unified_jitter_std'] = pd.to_numeric(df[col_map['std']], errors='coerce')
            unified_df['unified_byte_vol'] = pd.to_numeric(df[col_map['tot size']], errors='coerce')
            unified_df['unified_state_flag'] = pd.to_numeric(df[col_map['syn_count']], errors='coerce')
            labels = df[col_map['label']].astype(str).str.lower()
            unified_df[TARGET] = np.where(labels.str.contains('benign'), 0, 1)

        elif 'attack' in col_map and 'mean' in col_map:
            unified_df['unified_mean_iat'] = pd.to_numeric(df[col_map['mean']], errors='coerce')
            unified_df['unified_jitter_std'] = pd.to_numeric(df[col_map['stddev']], errors='coerce')
            unified_df['unified_byte_vol'] = pd.to_numeric(df[col_map['bytes']], errors='coerce')
            unified_df['unified_state_flag'] = pd.to_numeric(df[col_map['seq']], errors='coerce')
            unified_df[TARGET] = pd.to_numeric(df[col_map['attack']], errors='coerce').astype(int)
        else:
            return pd.DataFrame()

        return optimize_dtypes(unified_df.dropna())
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 3. 15GB REDLINE EXTRACTION (12 MILLION PACKETS)
# ==========================================
logger.info(f"Scanning {input_base} for CSVs...")
all_csv_files = list(input_base.rglob('*.csv'))

if not all_csv_files:
    raise ValueError("Missing Data: Please add datasets to your Kaggle session.")

master_frames = []
for f in all_csv_files:
    df_chunk = process_file(f)
    if not df_chunk.empty: 
        master_frames.append(df_chunk)

raw_df = pd.concat(master_frames, ignore_index=True)
logger.info(f"Raw Matrix Combined. Total Packets: {len(raw_df):,}")

# Pushing the RAM: Extracting ~12 Million packets 
logger.info("Redlining the 15GB RAM Limit: Extracting 12,000,000 Packets...")

safe_df = raw_df[raw_df[TARGET] == 0]
attack_df = raw_df[raw_df[TARGET] == 1]

# Take literally every Safe packet available (~1.1 Million), and 10.9M attack packets
n_safe = len(safe_df)
n_attack = min(len(attack_df), 10900000)

master_df = pd.concat([
    safe_df, 
    attack_df.sample(n=n_attack, random_state=42)
], ignore_index=True)

# Shuffle thoroughly
master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)

logger.info(f"Massive Cross-Domain Matrix Built. Total Packets: {len(master_df):,}")
logger.info(f"Safe Traffic: {len(master_df[master_df[TARGET]==0]):,} | Botnet: {len(master_df[master_df[TARGET]==1]):,}")

del raw_df, safe_df, attack_df, master_frames
gc.collect() 

X = master_df[UNIFIED_FEATURES]
y = master_df[TARGET]

# ==========================================
# 4. NESTED EVALUATION & GRID SEARCH
# ==========================================
logger.info("Allocating 80/20 train-test partitions...")
X_train_outer, X_test, y_train_outer, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

del master_df
gc.collect()

X_test_np = X_test.values
test_batch = X_test_np[:1000]

logger.info("Constructing ML Architectures for Benchmarking...")

models = {
    "DNN (MLP)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(criterion='entropy', max_depth=5, n_estimators=100, n_jobs=2, random_state=42))
    ])
}

if XGB_AVAILABLE:
    models["XGBoost"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, tree_method="hist", n_jobs=2, random_state=42))
    ])

logger.info("Running GridSearchCV for Edge Shield...")
edge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': [3, 4, 5],
    'clf__min_samples_leaf': [1, 5]
}

grid_search = GridSearchCV(edge_pipeline, param_grid, cv=3, scoring='f1', n_jobs=2)

logger.info("Running Global SMOTE for GridSearch (This will push RAM to ~12GB)...")
smote_global = SMOTE(random_state=42)
X_gs_bal, y_gs_bal = smote_global.fit_resample(X_train_outer, y_train_outer)
grid_search.fit(X_gs_bal, y_gs_bal)

models["Production Edge Shield"] = grid_search.best_estimator_
logger.info(f"Optimal Edge Parameters: {grid_search.best_params_}")

del X_gs_bal, y_gs_bal
gc.collect()

# ==========================================
# 5. FULL 5-FOLD SMOTE GAUNTLET EXECUTION
# ==========================================
logger.info("Commencing 5-Fold Gauntlet on 20M+ Packet Space...")
benchmark_results = []
metadata = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in models.items():
    logger.info(f"--> Evaluating architecture: {name}")
    cv_scores = []

    for train_idx, val_idx in skf.split(X_train_outer, y_train_outer):
        X_fold_train = X_train_outer.iloc[train_idx]
        y_fold_train = y_train_outer.iloc[train_idx]
        X_fold_val = X_train_outer.iloc[val_idx]
        y_fold_val = y_train_outer.iloc[val_idx]

        smote_fold = SMOTE(random_state=42)
        X_f_bal, y_f_bal = smote_fold.fit_resample(X_fold_train, y_fold_train)
        
        pipeline.fit(X_f_bal.values, y_f_bal.values)
        cv_scores.append(f1_score(y_fold_val, pipeline.predict(X_fold_val.values), zero_division=0))
        
        del X_f_bal, y_f_bal, X_fold_train, y_fold_train, X_fold_val, y_fold_val
        gc.collect()

    f1_mean = np.mean(cv_scores)
    ci95 = 1.96 * (np.std(cv_scores, ddof=1) / np.sqrt(len(cv_scores)))

    start_train = time.perf_counter()
    X_train_final_bal, y_train_final_bal = smote_global.fit_resample(X_train_outer, y_train_outer)
    pipeline.fit(X_train_final_bal.values, y_train_final_bal.values)
    train_time = time.perf_counter() - start_train
    
    del X_train_final_bal, y_train_final_bal
    gc.collect()

    preds = pipeline.predict(X_test_np)
    probs = pipeline.predict_proba(X_test_np)[:, 1] if hasattr(pipeline[-1], "predict_proba") else [0]*len(preds)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs) if hasattr(pipeline[-1], "predict_proba") else 0.0

    temp_path = working_dir / "temp.pkl"
    size_kb = len(joblib.dump(pipeline, temp_path)) / 1024.0 if temp_path.exists() else 0

    for _ in range(50): pipeline.predict(test_batch) 
    start_lat = time.perf_counter_ns()
    for _ in range(1000): pipeline.predict(test_batch)
    end_lat = time.perf_counter_ns()

    avg_latency_us = ((end_lat - start_lat) / (len(test_batch) * 1000)) / 1000.0

    logger.info(f"[{name}] F1: {f1_mean:.4f} ± {ci95:.4f} | Acc: {acc:.4f} | Latency: {avg_latency_us:.2f}μs | Size: {size_kb:.2f}KB")

    benchmark_results.append({
        "Model": name, "Accuracy": acc, "F1_Mean": f1_mean, "Precision": prec,
        "Recall": rec, "ROC-AUC": roc, "TrainTime_s": train_time,
        "Latency_us": avg_latency_us, "Size_KB": size_kb
    })

    if name == "Production Edge Shield":
        logger.info("\n--- Classification Report (Production Edge Shield) ---")
        logger.info("\n" + classification_report(y_test, preds))

        importances = pipeline.named_steps['clf'].feature_importances_
        pd.DataFrame({'Feature': UNIFIED_FEATURES, 'Importance': importances}).to_csv(working_dir / "feature_importances.csv", index=False)

        metadata = {
            "model": pipeline,
            "features": UNIFIED_FEATURES,
            "version": "15GB Redline IEEE Release",
            "hyperparameters": grid_search.best_params_,
            "training_date": str(datetime.now())
        }

# ==========================================
# 6. EXPORT ARTIFACTS
# ==========================================
pd.DataFrame(benchmark_results).to_csv(working_dir / "production_benchmark_results.csv", index=False)
joblib.dump(metadata, working_dir / 'production_master_shield.pkl')
temp_path.unlink(missing_ok=True)
logger.info(f"Pipeline Complete. Heavy-duty artifacts securely exported to {working_dir}.")
