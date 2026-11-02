"""
PredictIQ - Streamlit Web Dashboard
Interactive Predictive Maintenance & Sensor Diagnostics System with Explainable AI (SHAP)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.features import (
    prepare_single_sample,
    FEATURE_COLUMNS_ENGINEERED
)
from src.explain import PredictIQExplainer

# Page configuration
st.set_page_config(
    page_title="PredictIQ - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-safe {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #198754;
        font-weight: 600;
    }
    .risk-warning {
        background-color: #fff3cd;
        color: #664d03;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #ffc107;
        font-weight: 600;
    }
    .risk-danger {
        background-color: #f8d7da;
        color: #842029;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #dc3545;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle():
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        return None
    bundle = joblib.load(model_path)
    explainer = PredictIQExplainer(bundle["model"], bundle.get("background_data"))
    return bundle, explainer


bundle_data = load_model_bundle()

# App Header
st.title("⚙️ PredictIQ: Predictive Maintenance Intelligence")
st.markdown(
    "**Physics-Informed Machine Learning for Early Equipment Failure Detection with Explainable AI (SHAP)**"
)

if bundle_data is None:
    st.error("Model file not found! Please run `python src/train.py` first to train and export the model.")
    st.stop()

bundle, explainer = bundle_data
model = bundle["model"]
model_name = bundle.get("model_name", "XGBoost (Engineered)")

# Sidebar: Preset Profiles & Inputs
st.sidebar.header("🛠️ Diagnostic Control Deck")
preset = st.sidebar.selectbox(
    "Load Preset Operating Scenario:",
    [
        "🟢 Normal Operation (Healthy)",
        "🔴 Overstrain Hazard (Tool Wear & High Torque)",
        "🔴 Heat Dissipation Hazard (Low Speed & Delta T)",
        "🔴 Power Overload Hazard (Excessive Power)",
        "Custom Configuration"
    ]
)

# Preset default values
defaults = {
    "type": "L",
    "air_temp": 300.0,
    "process_temp": 310.0,
    "rot_speed": 1500.0,
    "torque": 40.0,
    "tool_wear": 50.0
}

if preset == "🔴 Overstrain Hazard (Tool Wear & High Torque)":
    defaults = {"type": "L", "air_temp": 301.0, "process_temp": 310.5, "rot_speed": 1350.0, "torque": 65.0, "tool_wear": 225.0}
elif preset == "🔴 Heat Dissipation Hazard (Low Speed & Delta T)":
    defaults = {"type": "L", "air_temp": 303.5, "process_temp": 310.0, "rot_speed": 1200.0, "torque": 55.0, "tool_wear": 110.0}
elif preset == "🔴 Power Overload Hazard (Excessive Power)":
    defaults = {"type": "M", "air_temp": 298.5, "process_temp": 308.5, "rot_speed": 2600.0, "torque": 68.0, "tool_wear": 130.0}

st.sidebar.markdown("---")
st.sidebar.subheader("Telemetry Inputs")

m_type = st.sidebar.selectbox("Machine Type (Variant)", ["L", "M", "H"], index=["L", "M", "H"].index(defaults["type"]))
air_t = st.sidebar.slider("Air Temperature [K]", min_value=295.0, max_value=305.0, value=float(defaults["air_temp"]), step=0.1)
proc_t = st.sidebar.slider("Process Temperature [K]", min_value=305.0, max_value=315.0, value=float(defaults["process_temp"]), step=0.1)
speed = st.sidebar.slider("Rotational Speed [rpm]", min_value=1100.0, max_value=2900.0, value=float(defaults["rot_speed"]), step=10.0)
torque = st.sidebar.slider("Torque [Nm]", min_value=3.0, max_value=80.0, value=float(defaults["torque"]), step=0.5)
wear = st.sidebar.slider("Tool Wear [min]", min_value=0.0, max_value=260.0, value=float(defaults["tool_wear"]), step=1.0)

# Build feature dataframe
sample_df = prepare_single_sample(
    air_temp=air_t,
    process_temp=proc_t,
    rot_speed=speed,
    torque=torque,
    tool_wear=wear,
    machine_type=m_type
)

# Derived physics calculations
delta_t = proc_t - air_t
power_watts = torque * speed * (2 * np.pi / 60.0)
wear_torque = wear * torque

# Layout: Tabs for Dashboard & Benchmarks
tab_dashboard, tab_benchmarks, tab_architecture = st.tabs([
    "📊 Real-Time Diagnostic Dashboard",
    "📈 Model Benchmarks & Comparison",
    "🧠 System Architecture & Physics"
])

with tab_dashboard:
    # Row 1: Real-Time Physics Telemetry Cards
    st.subheader("1. Live Derived Physics Metrics")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric(
            label="Thermal Gradient (ΔT = T_proc - T_air)",
            value=f"{delta_t:.2f} K",
            delta="Normal Cooling" if delta_t >= 8.6 else "Critical Heat Warning (< 8.6 K)",
            delta_color="normal" if delta_t >= 8.6 else "inverse"
        )
    with c2:
        power_status = "Normal Power" if 3500 <= power_watts <= 9000 else "Power Hazard"
        st.metric(
            label="Mechanical Power (P = τ · ω)",
            value=f"{power_watts:.1f} W",
            delta=power_status,
            delta_color="normal" if power_status == "Normal Power" else "inverse"
        )
    with c3:
        strain_status = "Normal Strain" if wear_torque <= 11000 else "Overstrain Hazard (> 11,000)"
        st.metric(
            label="Tool Overstrain (Wear × Torque)",
            value=f"{wear_torque:.1f} min·Nm",
            delta=strain_status,
            delta_color="normal" if strain_status == "Normal Strain" else "inverse"
        )

    st.markdown("---")

    # Row 2: Prediction & Risk Assessment
    st.subheader("2. Impending Failure Risk Assessment")
    
    prob_failure = float(model.predict_proba(sample_df)[0][1])
    risk_pct = prob_failure * 100.0

    r_col1, r_col2 = st.columns([1, 2])

    with r_col1:
        st.markdown(f"### Machine Status")
        st.progress(min(max(prob_failure, 0.0), 1.0))
        st.markdown(f"**Failure Probability:** `{risk_pct:.2f}%`")
        st.markdown(f"**Champion Engine:** `{model_name}`")

    with r_col2:
        if risk_pct < 30.0:
            st.markdown(f"""
            <div class="risk-safe">
                <h4>🟢 STATUS: HEALTHY & NORMAL OPERATION</h4>
                <p>The machine vitals are operating within safe thermodynamic and mechanical bounds.</p>
                <b>Recommended Action:</b> Continue scheduled operations without interruption. Next routine inspection as per standard calendar.
            </div>
            """, unsafe_allow_html=True)
        elif 30.0 <= risk_pct < 65.0:
            st.markdown(f"""
            <div class="risk-warning">
                <h4>🟡 STATUS: MAINTENANCE ADVISORY (MODERATE HAZARD)</h4>
                <p>Sensor indicators suggest elevated stress or early-stage component fatigue.</p>
                <b>Recommended Action:</b> Schedule an inspection at the end of the current shift. Check tool wear and monitor spindle vibration.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-danger">
                <h4>🔴 STATUS: CRITICAL RISK — IMMINENT BREAKDOWN LIKELY</h4>
                <p>High probability of mechanical seizure, tool breakage, or thermal trip.</p>
                <b>Recommended Action:</b> Immediate machine stop required. Replace cutting tool insert and verify spindle cooling lines before restarting.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Row 3: Explainable AI with SHAP
    st.subheader("3. Explainable AI: Why Did the Model Flag This?")
    st.markdown(
        "Using **SHAP (Shapley Additive exPlanations)**, PredictIQ breaks open the model's decision "
        "and quantifies the exact contribution of each sensor and engineered physical property."
    )

    exp_obj, summary_table = explainer.explain_sample(sample_df)

    exp_col1, exp_col2 = st.columns([1.2, 1])

    with exp_col1:
        st.markdown("##### SHAP Waterfall Contribution Plot")
        fig = explainer.plot_waterfall(exp_obj, max_display=7)
        st.pyplot(fig)
        plt.close(fig)

    with exp_col2:
        st.markdown("##### Key Driver Breakdown")
        display_summary = summary_table[["Feature", "Actual Value", "SHAP Contribution", "Impact Direction"]].head(6)
        st.dataframe(
            display_summary.style.format({
                "Actual Value": "{:.2f}",
                "SHAP Contribution": "{:+.3f}"
            }),
            use_container_width=True
        )


with tab_benchmarks:
    st.subheader("Model Performance Benchmark: Baseline vs. Physics-Engineered")
    st.markdown(
        "This table demonstrates the direct scientific impact of incorporating physical domain equations "
        "($\\Delta T$, Mechanical Power, and Overstrain) compared to raw sensor inputs alone."
    )
    
    metrics_list = bundle.get("all_metrics", [])
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, subset=["recall", "f1", "roc_auc"], color="#c3e6cb"),
            use_container_width=True
        )
        
        # Interactive chart
        chart_df = metrics_df.set_index("model")[["recall", "f1"]]
        st.bar_chart(chart_df)
    else:
        st.info("Metrics summary not found in bundle.")


with tab_architecture:
    st.subheader("PredictIQ System Architecture & Physical Foundations")
    st.markdown("""
    ### 1. Data Ingestion & Target Leakage Prevention
    * **AI4I 2020 Dataset (UCI ML Repository)**: 10,000 synthetic CNC operational observations.
    * **Target Leakage Remediation**: Dropped `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` failure diagnostic flags. In real-world telemetry, post-failure codes do not exist prior to breakdown.

    ### 2. Physical Failure Rules & Mathematical Formulations
    * **Heat Dissipation Failure (HDF)**:
      $$\\Delta T = T_{\\text{process}} - T_{\\text{air}} < 8.6\\text{ K} \\quad \\text{and} \\quad \\omega < 1380\\text{ rpm}$$
    * **Power Failure (PWF)**:
      $$P = \\tau \\cdot \\omega = \\text{Torque} \\times \\left( \\text{Rotational Speed} \\times \\frac{2\\pi}{60} \\right) \\notin [3500\\text{ W}, 9000\\text{ W}]$$
    * **Overstrain Failure (OSF)**:
      $$\\text{Overstrain} = \\text{Tool Wear [min]} \\times \\text{Torque [Nm]} > 11000$$
    * **Tool Wear Failure (TWF)**:
      $$\\text{Tool Wear} > 200\\text{ minutes}$$

    ### 3. Class Imbalance Strategy (3.39% Minority Class)
    * **Stratified 80/20 train/test partition** guarantees equal class prevalence.
    * **Cost-sensitive learning**: Weighted loss functions (`scale_pos_weight` in XGBoost and `class_weight='balanced'` in Random Forest) prioritize high Recall, ensuring critical machine breakdowns are caught.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("PredictIQ v1.0 | Minor Project Submission")
