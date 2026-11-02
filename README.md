# PredictIQ: Physics-Informed Predictive Maintenance with Explainable AI (XAI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

PredictIQ is a physics-informed Machine Learning system designed to predict industrial machine failures from sensor telemetry before breakdowns occur. Built on the **AI4I 2020 Predictive Maintenance Dataset**, it addresses severe class imbalance (3.4% failure rate), prevents target leakage, engineers thermodynamic/mechanical features, and provides model explainability via **SHAP**.

---

## 🌟 Key Features

- **Physics-Informed Feature Engineering**: Computes real-time thermal gradient ($\Delta T$), mechanical rotational power ($P = \tau \omega$), and tool overstrain ($ToolWear \times Torque$).
- **Multi-Model Benchmark**: Rigorous empirical comparison of **Logistic Regression**, **Random Forest**, and **XGBoost** (Baseline vs. Engineered Features).
- **Target Leakage Remediation**: Drops post-mortem failure code columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) that artificially inflate test accuracy.
- **Cost-Sensitive Imbalance Optimization**: Optimizes for **Recall** and **PR-AUC** over naive accuracy to catch high-cost industrial breakdowns.
- **Explainable AI (SHAP)**: Individual prediction waterfall plots pinpointing the physical drivers pushing equipment risk up or down.
- **Interactive Streamlit Dashboard**: Live telemetry simulation with presets for heat failure, overstrain, power surge, and healthy operation.
- **JupyterLab Walkthrough**: Complete notebook in `notebooks/PredictIQ_Walkthrough.ipynb` for academic presentations and code inspection.

---

## 📁 Project Architecture

```
minor_prj/
├── data/
│   └── ai4i2020.csv                    # AI4I 2020 dataset (10,000 observations)
├── models/
│   ├── best_model.pkl                  # Trained champion model bundle (XGBoost)
│   └── metrics_summary.json            # Benchmark evaluation results
├── notebooks/
│   └── PredictIQ_Walkthrough.ipynb     # Step-by-step JupyterLab notebook
├── src/
│   ├── __init__.py
│   ├── features.py                     # Cleaning & physics engineering functions
│   ├── train.py                        # Standalone ML training & evaluation script
│   └── explain.py                      # SHAP explainability module
├── app.py                              # Streamlit web dashboard
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation & Viva guide
```

---

## 📊 Experimental Results

| Model | Feature Set | Accuracy | Precision | Recall (Critical) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Baseline | 82.45% | 14.18% | 82.35% | 0.2419 | 0.9070 |
| **Random Forest** | Baseline | 98.00% | 91.18% | 45.59% | 0.6078 | 0.9624 |
| **XGBoost** | Baseline | 97.45% | 59.34% | 79.41% | 0.6792 | 0.9686 |
| **Logistic Regression** | **Engineered** | 85.90% | 17.96% | 88.24% | 0.2985 | 0.9384 |
| **Random Forest** | **Engineered** | 99.05% | 96.23% | **75.00%** (+29.4%) | **0.8430** (+23.5%) | 0.9805 |
| **XGBoost (Champion)** | **Engineered** | 98.65% | 78.08% | **83.82%** | **0.8085** | **0.9832** |

> **Key Finding:** Adding physics-informed features boosted Random Forest recall from **45.6% to 75.0%** (+29.4%) and XGBoost to **83.8% recall** with a **0.9832 ROC-AUC**.

---

## 🚀 Getting Started

### 1. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 2. Run in JupyterLab
```bash
jupyter lab notebooks/PredictIQ_Walkthrough.ipynb
```

### 3. Re-train the Models from Scratch
```bash
python src/train.py
```

---

## 🎓 Viva Voce & Presentation Cheat-Sheet

1. **Why prioritize Recall over Accuracy?**
   * *Answer:* With 96.6% healthy machines, a naive classifier predicting all 0s gets 96.6% accuracy but detects 0 failures. In manufacturing, an undetected breakdown costs thousands of dollars; a false alarm costs only a brief inspection.
2. **What is Target Leakage in this project?**
   * *Answer:* The dataset includes post-failure columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). Using them would leak future diagnostic knowledge that is unavailable during live machine operation.
3. **What is the significance of the engineered features?**
   * *Answer:* They translate physical manufacturing mechanics into mathematical features ($\Delta T$ for heat dissipation failure, $P = \tau \omega$ for power failure, and $ToolWear \times Torque$ for structural overstrain).
4. **Why use SHAP?**
   * *Answer:* Machine learning models are often seen as "black boxes". SHAP computes cooperative game-theory Shapley values to tell the maintenance engineer exactly which physical sensor triggered the warning.
