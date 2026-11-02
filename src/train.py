"""
PredictIQ - Model Training & Evaluation Pipeline
Trains Logistic Regression, Random Forest, and XGBoost with baseline vs. engineered features.
Evaluates precision, recall, F1, ROC-AUC and exports the champion model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from features import (
    clean_raw_data,
    engineer_physics_features,
    FEATURE_COLUMNS_RAW,
    FEATURE_COLUMNS_ENGINEERED
)


def evaluate_model(name: str, y_true, y_pred, y_prob) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Accuracy  : {acc:.4f} (Note: Naive baseline = 0.9661)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}  <-- Critical Metric (Catches Failures)")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"  TN: {cm[0][0]:<5} | FP (False Alarms): {cm[0][1]}")
    print(f"  FN: {cm[1][0]:<5} | TP (Caught Breaks): {cm[1][1]}")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm
    }


def run_pipeline(data_path="data/ai4i2020.csv", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"[+] Loading dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    print(f"[+] Raw data shape: {df_raw.shape}")
    
    # Clean data (target leakage removal + column renaming)
    df_clean = clean_raw_data(df_raw)
    y = df_clean["failure"]
    
    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Raw Baseline Features
    # -------------------------------------------------------------------------
    print("\n" + "#"*60)
    print(" EXPERIMENT 1: BASELINE (RAW SENSOR FEATURES ONLY)")
    print("#"*60)
    X_raw = df_clean[FEATURE_COLUMNS_RAW]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scaler for Logistic Regression
    scaler_raw = StandardScaler()
    X_train_raw_sc = scaler_raw.fit_transform(X_train_raw)
    X_test_raw_sc = scaler_raw.transform(X_test_raw)
    
    # 1. Baseline Logistic Regression
    lr_base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_base.fit(X_train_raw_sc, y_train)
    res_lr_base = evaluate_model(
        "Logistic Regression (Baseline)",
        y_test,
        lr_base.predict(X_test_raw_sc),
        lr_base.predict_proba(X_test_raw_sc)[:, 1]
    )
    
    # 2. Baseline Random Forest
    rf_base = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
    rf_base.fit(X_train_raw, y_train)
    res_rf_base = evaluate_model(
        "Random Forest (Baseline)",
        y_test,
        rf_base.predict(X_test_raw),
        rf_base.predict_proba(X_test_raw)[:, 1]
    )
    
    # 3. Baseline XGBoost
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_base = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42
    )
    xgb_base.fit(X_train_raw, y_train)
    res_xgb_base = evaluate_model(
        "XGBoost (Baseline)",
        y_test,
        xgb_base.predict(X_test_raw),
        xgb_base.predict_proba(X_test_raw)[:, 1]
    )
    
    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Engineered Physics Features (PredictIQ Innovation)
    # -------------------------------------------------------------------------
    print("\n" + "#"*60)
    print(" EXPERIMENT 2: PHYSICS-ENGINEERED FEATURES (temp_diff, power, wear_torque)")
    print("#"*60)
    df_engineered = engineer_physics_features(df_clean)
    X_eng = df_engineered[FEATURE_COLUMNS_ENGINEERED]
    
    X_train_eng, X_test_eng, _, _ = train_test_split(
        X_eng, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scaler for Logistic Regression
    scaler_eng = StandardScaler()
    X_train_eng_sc = scaler_eng.fit_transform(X_train_eng)
    X_test_eng_sc = scaler_eng.transform(X_test_eng)
    
    # 1. Engineered Logistic Regression
    lr_eng = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_eng.fit(X_train_eng_sc, y_train)
    res_lr_eng = evaluate_model(
        "Logistic Regression (Engineered)",
        y_test,
        lr_eng.predict(X_test_eng_sc),
        lr_eng.predict_proba(X_test_eng_sc)[:, 1]
    )
    
    # 2. Engineered Random Forest
    rf_eng = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
    rf_eng.fit(X_train_eng, y_train)
    res_rf_eng = evaluate_model(
        "Random Forest (Engineered)",
        y_test,
        rf_eng.predict(X_test_eng),
        rf_eng.predict_proba(X_test_eng)[:, 1]
    )
    
    # 3. Engineered XGBoost
    xgb_eng = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42
    )
    xgb_eng.fit(X_train_eng, y_train)
    res_xgb_eng = evaluate_model(
        "XGBoost (Engineered)",
        y_test,
        xgb_eng.predict(X_test_eng),
        xgb_eng.predict_proba(X_test_eng)[:, 1]
    )
    
    # -------------------------------------------------------------------------
    # COMPARISON TABLE & EXPORT
    # -------------------------------------------------------------------------
    all_results = [
        res_lr_base, res_rf_base, res_xgb_base,
        res_lr_eng, res_rf_eng, res_xgb_eng
    ]
    
    print("\n" + "="*80)
    print(" COMPREHENSIVE PERFORMANCE COMPARISON TABLE")
    print("="*80)
    summary_df = pd.DataFrame(all_results)[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    print(summary_df.to_string(index=False))
    
    # Determine Champion Model by F1-Score and Recall
    # Random Forest / XGBoost with engineered features are the top contenders
    champion_model = xgb_eng if res_xgb_eng["recall"] >= res_rf_eng["recall"] else rf_eng
    champion_name = "XGBoost (Engineered)" if champion_model == xgb_eng else "Random Forest (Engineered)"
    print(f"\n[+] Champion Model Selected: {champion_name}")
    
    # Save Model Bundle (Model + metadata + columns + sample data for SHAP)
    model_bundle = {
        "model": champion_model,
        "model_name": champion_name,
        "feature_columns": FEATURE_COLUMNS_ENGINEERED,
        "background_data": X_train_eng.sample(200, random_state=42),  # Background sample for SHAP
        "all_metrics": all_results
    }
    
    save_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(model_bundle, save_path)
    print(f"[+] Successfully saved champion model bundle to: {save_path}")
    
    # Save JSON summary for easy dashboard loading
    metrics_json_path = os.path.join(models_dir, "metrics_summary.json")
    with open(metrics_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[+] Saved metrics summary JSON to: {metrics_json_path}")
    
    return all_results


if __name__ == "__main__":
    run_pipeline()
