"""
GazeGuard — Interactive Demo
=============================
Streamlit app: Testing page + Eval page.
Run with: streamlit run app.py
"""

import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── SHAP (optional) ─────────────────────────────────────────────────────────
try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── Paths ────────────────────────────────────────────────────────────────────
_cwd = os.getcwd()
if os.path.isdir(os.path.join(_cwd, "data")) and os.path.isdir(os.path.join(_cwd, "output")):
    PROJECT_DIR = _cwd
else:
    PROJECT_DIR = _cwd
    for _ in range(5):
        parent = os.path.dirname(PROJECT_DIR)
        if os.path.isdir(os.path.join(parent, "data")) and os.path.isdir(os.path.join(parent, "output")):
            PROJECT_DIR = parent
            break
        if parent == PROJECT_DIR:
            break
        PROJECT_DIR = parent

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")


# ── Feature engineering (must match training exactly) ───────────────────────
def engineer_features_from_fixation(fd_trial, feature_cols, trial_difficulty_map):
    pivot = fd_trial.pivot_table(
        columns=["AOI", "ImageType"],
        values="Duration_ms",
        aggfunc="sum",
    )

    row = {}
    for key, val in pivot.items():
        aoi, img = key
        col_name = f"{aoi}_{img}"
        if hasattr(val, "iloc"):
            row[col_name] = float(val.iloc[0]) if len(val) > 0 and pd.notna(val.iloc[0]) else 0.0
        else:
            row[col_name] = float(val) if pd.notna(val) else 0.0

    df = pd.DataFrame([row])

    base_aois = sorted(set(
        c.rsplit("_", 1)[0]
        for c in feature_cols
        if "_" in c
        and not any(s in c for s in ["AIratio", "Realratio", "AbsDiff", "Total",
                                       "Duration", "Proportion", "Num", "Max", "Std",
                                       "Face", "Nature", "Trial", "AI_"])
    ))

    for aoi in base_aois:
        ai_col = f"{aoi}_AI"
        real_col = f"{aoi}_Real"
        if ai_col not in df.columns:
            df[ai_col] = 0.0
        if real_col not in df.columns:
            df[real_col] = 0.0
        df[f"{aoi}_AIratio"] = (df[ai_col] + 1) / (df[real_col] + 1)
        df[f"{aoi}_Realratio"] = (df[real_col] + 1) / (df[ai_col] + 1)
        df[f"{aoi}_AbsDiff"] = df[ai_col] - df[real_col]
        df[f"{aoi}_Total"] = df[ai_col] + df[real_col]

    duration_cols = [c for c in df.columns if c.endswith("_AI") or c.endswith("_Real")]
    df["Total_Duration_All"] = df[duration_cols].sum(axis=1)
    df["Num_AOIs_Fixated"] = (df[duration_cols] > 0).sum(axis=1)
    df["Max_AOI_Duration"] = df[duration_cols].max(axis=1)
    df["Std_AOI_Duration"] = df[duration_cols].std(axis=1)

    ai_cols_all = [c for c in df.columns if c.endswith("_AI") and not c.endswith("_AIratio")]
    df["AI_Proportion"] = df[ai_cols_all].sum(axis=1) / (df["Total_Duration_All"] + 1)

    pupil_total = df["Pupil_Total"].values[0] if "Pupil_Total" in df else 0.0
    skin_total = df["Skin_Total"].values[0] if "Skin_Total" in df else 0.0
    df["Face_Total"] = pupil_total + skin_total
    pupil_ai = df["Pupil_AI"].values[0] if "Pupil_AI" in df else 0.0
    skin_ai = df["Skin_AI"].values[0] if "Skin_AI" in df else 0.0
    pupil_real = df["Pupil_Real"].values[0] if "Pupil_Real" in df else 0.0
    skin_real = df["Skin_Real"].values[0] if "Skin_Real" in df else 0.0
    df["Face_AIratio"] = (pupil_ai + skin_ai + 1) / (pupil_real + skin_real + 1)

    cloud_total = df["Cloud_Total"].values[0] if "Cloud_Total" in df else 0.0
    water_total = df["Water_Total"].values[0] if "Water_Total" in df else 0.0
    df["Nature_Total"] = cloud_total + water_total
    cloud_ai = df["Cloud_AI"].values[0] if "Cloud_AI" in df else 0.0
    water_ai = df["Water_AI"].values[0] if "Water_AI" in df else 0.0
    cloud_real = df["Cloud_Real"].values[0] if "Cloud_Real" in df else 0.0
    water_real = df["Water_Real"].values[0] if "Water_Real" in df else 0.0
    df["Nature_AIratio"] = (cloud_ai + water_ai + 1) / (cloud_real + water_real + 1)

    trial_num = int(fd_trial["Trial_num"].iloc[0]) if "Trial_num" in fd_trial.columns else 1
    df["Trial_Accuracy_Rate"] = trial_difficulty_map.get(trial_num, 0.65)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols]


# ── Cached loaders ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))
    lr_model = joblib.load(os.path.join(MODEL_DIR, "logreg_model.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    return xgb_model, lr_model, feature_cols


@st.cache_data
def load_trial_data():
    fd = pd.read_csv(os.path.join(DATA_DIR, "per_trial_fixation_data.csv"))
    rl = pd.read_csv(os.path.join(DATA_DIR, "per_trial_response_labels.csv"))

    def extract_num(s):
        return int("".join(ch for ch in str(s) if ch.isdigit()))

    fd["PID_num"] = fd["ParticipantID"].map(extract_num)
    fd["Trial_num"] = fd["Trial"].map(extract_num)
    rl["PID_num"] = rl["ParticipantID"].map(extract_num)
    rl["Trial_num"] = rl["Trial"].map(extract_num)

    fd = fd[fd["AOI"] != "Total"].copy()

    data = fd.merge(
        rl[["PID_num", "Trial_num", "Response"]],
        on=["PID_num", "Trial_num"],
        how="inner",
    )
    return data


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GazeGuard",
    page_icon="👁️",
    layout="wide",
)

# ── Sidebar nav ─────────────────────────────────────────────────────────────
st.sidebar.title("👁️ GazeGuard")
st.sidebar.caption("AI Image Detection Predictor")
page = st.sidebar.radio("Navigate", ["🧪 Testing", "📊 Evaluation"], label_visibility="collapsed")
st.sidebar.divider()


# ── Load ────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading models and data... (first load ~30s)"):
    xgb_model, lr_model, feature_cols = load_models()
    data = load_trial_data()

trial_difficulty = data.groupby("Trial_num")["Response"].apply(
    lambda x: (x == "Correct").mean()
).to_dict()

# ── Common sidebar for trial selection ───────────────────────────────────────
trials_df = data.groupby(["PID_num", "Trial_num"]).agg(
    Response=("Response", "first"),
    Participant=("ParticipantID", "first"),
    Trial=("Trial", "first"),
).reset_index()

st.sidebar.subheader("Select Trial")
selected_participant = st.sidebar.selectbox(
    "Participant",
    sorted(trials_df["PID_num"].unique()),
    format_func=lambda x: f"P{x:02d}",
)
participant_trials = trials_df[trials_df["PID_num"] == selected_participant]
selected_trial = st.sidebar.selectbox(
    "Trial",
    sorted(participant_trials["Trial_num"].unique()),
    format_func=lambda x: f"Trial {x}",
)

# ── Get trial data ──────────────────────────────────────────────────────────
trial_data = data[
    (data["PID_num"] == selected_participant) &
    (data["Trial_num"] == selected_trial)
].copy()

if trial_data.empty:
    st.error("No data found.")
    st.stop()

true_label = trial_data["Response"].iloc[0]

# ── Predict ─────────────────────────────────────────────────────────────────
X_trial = engineer_features_from_fixation(trial_data, feature_cols, trial_difficulty)

xgb_prob = xgb_model.predict_proba(X_trial)[0]
xgb_pred = int(xgb_model.predict(X_trial)[0])
lr_prob = lr_model.predict_proba(X_trial)[0]
lr_pred = int(lr_model.predict(X_trial)[0])

lr_fitted = lr_model.named_steps["model"]
lr_coef = pd.Series(lr_fitted.coef_[0], index=feature_cols)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: TESTING
# ════════════════════════════════════════════════════════════════════════════
if page == "🧪 Testing":

    st.header("🧪 Trial Explorer")
    st.markdown("*Select a participant and trial on the left to see gaze data, predictions, and explanations.*")

    # ── Row 1: Gaze data table ───────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("📊 Gaze Fixation Data")

        display = trial_data.pivot_table(
            index="AOI", columns="ImageType", values="Duration_ms",
            aggfunc="sum", fill_value=0,
        ).reset_index()

        if "AI" in display.columns and "Real" in display.columns:
            display["AI/Real"] = ((display["AI"] + 1) / (display["Real"] + 1)).round(2)

        st.dataframe(
            display.style.format({"AI": "{:.0f}", "Real": "{:.0f}"}),
            use_container_width=True, hide_index=True,
        )

        total_ai = trial_data[trial_data["ImageType"] == "AI"]["Duration_ms"].sum()
        total_real = trial_data[trial_data["ImageType"] == "Real"]["Duration_ms"].sum()
        n_aois = trial_data[trial_data["Duration_ms"] > 0]["AOI"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI Duration", f"{total_ai:,} ms")
        c2.metric("Real Duration", f"{total_real:,} ms")
        c3.metric("AI Proportion", f"{total_ai/(total_ai+total_real)*100:.1f}%")
        c4.metric("AOIs Fixated", f"{n_aois}/12")

    st.markdown("")

    # ── Row 2: XGBoost + LogReg predictions side by side ───────────────────
    col_xgb, col_lr = st.columns(2)

    with col_xgb:
        with st.container(border=True):
            st.subheader("🌲 XGBoost")

            xgb_label_str = "Correct" if xgb_pred == 1 else "Fooled"
            xgb_match = (xgb_pred == 1 and true_label == "Correct") or (xgb_pred == 0 and true_label == "Fooled")
            correct_card_color = "#1B5E20" if xgb_match else "#B71C1C"
            correct_card_text = "✅ Correct Prediction" if xgb_match else "❌ Wrong Prediction"

            st.markdown(f"""
<div style="background:#1565C0;padding:1px 16px;border-radius:8px;margin-bottom:12px;">
    <h4 style="color:white;margin:8px 0;">Prediction: {xgb_label_str}</h4>
    <p style="color:#BBDEFB;margin:0;">Actual: {true_label} &nbsp;|&nbsp; {correct_card_text}</p>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span>P(Fooled)</span><span><b>{xgb_prob[0]:.1%}</b></span>
</div>
<div style="background:#E0E0E0;border-radius:4px;height:12px;margin-bottom:12px;">
    <div style="background:#F44336;width:{xgb_prob[0]*100:.0f}%;border-radius:4px;height:12px;"></div>
</div>
<div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span>P(Correct)</span><span><b>{xgb_prob[1]:.1%}</b></span>
</div>
<div style="background:#E0E0E0;border-radius:4px;height:12px;">
    <div style="background:#4CAF50;width:{xgb_prob[1]*100:.0f}%;border-radius:4px;height:12px;"></div>
</div>
""", unsafe_allow_html=True)

    with col_lr:
        with st.container(border=True):
            st.subheader("📈 Logistic Regression (L1)")

            lr_label_str = "Correct" if lr_pred == 1 else "Fooled"
            lr_match = (lr_pred == 1 and true_label == "Correct") or (lr_pred == 0 and true_label == "Fooled")
            lr_card_text = "✅ Correct Prediction" if lr_match else "❌ Wrong Prediction"

            st.markdown(f"""
<div style="background:#E65100;padding:1px 16px;border-radius:8px;margin-bottom:12px;">
    <h4 style="color:white;margin:8px 0;">Prediction: {lr_label_str}</h4>
    <p style="color:#FFE0B2;margin:0;">Actual: {true_label} &nbsp;|&nbsp; {lr_card_text}</p>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span>P(Fooled)</span><span><b>{lr_prob[0]:.1%}</b></span>
</div>
<div style="background:#E0E0E0;border-radius:4px;height:12px;margin-bottom:12px;">
    <div style="background:#F44336;width:{lr_prob[0]*100:.0f}%;border-radius:4px;height:12px;"></div>
</div>
<div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span>P(Correct)</span><span><b>{lr_prob[1]:.1%}</b></span>
</div>
<div style="background:#E0E0E0;border-radius:4px;height:12px;">
    <div style="background:#4CAF50;width:{lr_prob[1]*100:.0f}%;border-radius:4px;height:12px;"></div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Why This Prediction? ────────────────────────────────────────────────
    st.subheader("🧠 Why This Prediction?")
    st.markdown("*Per-trial feature contributions to the Logistic Regression (L1) decision — and SHAP values from XGBoost.*")

    # Build LR contribution table
    lr_contrib = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": lr_coef.values,
        "Feature Value": [X_trial[c].values[0] for c in feature_cols],
        "Contribution (coef × value)": lr_coef.values * np.array([X_trial[c].values[0] for c in feature_cols]),
    })
    lr_contrib = lr_contrib.reindex(lr_contrib["Contribution (coef × value)"].abs().sort_values(ascending=False).index)
    lr_top = lr_contrib.head(10).reset_index(drop=True)

    lr_top_driving_correct = lr_top[lr_top["Contribution (coef × value)"] > 0].head(5)
    lr_top_driving_fooled = lr_top[lr_top["Contribution (coef × value)"] < 0].head(5)

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        with st.container(border=True):
            st.markdown("**Pushing → ✅ Correct**")
            rows = []
            for _, r in lr_top_driving_correct.iterrows():
                rows.append({
                    "Feature": f"`{r['Feature']}`",
                    "Coef": f"+{r['Coefficient']:.3f}",
                    "Value": f"{r['Feature Value']:.1f}",
                    "Contribution": f"+{r['Contribution (coef × value)']:.2f}",
                })
            if rows:
                st.table(pd.DataFrame(rows))
            else:
                st.caption("No positive contributions for this trial.")

    with exp_col2:
        with st.container(border=True):
            st.markdown("**Pushing → ❌ Fooled**")
            rows = []
            for _, r in lr_top_driving_fooled.iterrows():
                rows.append({
                    "Feature": f"`{r['Feature']}`",
                    "Coef": f"{r['Coefficient']:.3f}",
                    "Value": f"{r['Feature Value']:.1f}",
                    "Contribution": f"{r['Contribution (coef × value)']:.2f}",
                })
            if rows:
                st.table(pd.DataFrame(rows))
            else:
                st.caption("No negative contributions for this trial.")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if HAS_SHAP:
        st.divider()
        st.subheader("🔍 SHAP Explanation (XGBoost)")
        st.markdown("*How each feature pushes the XGBoost prediction away from the base value for this specific trial.*")

        with st.spinner("Computing SHAP values..."):
            xgb_raw = xgb_model.named_steps["model"] if hasattr(xgb_model, "named_steps") else xgb_model
            scaler = xgb_model.named_steps["scaler"] if hasattr(xgb_model, "named_steps") else None
            X_scaled = scaler.transform(X_trial) if scaler else X_trial.values
            explainer = _shap.TreeExplainer(xgb_raw)
            shap_vals = explainer.shap_values(X_scaled)

        # Handle binary classification output (some xgb versions return list)
        if isinstance(shap_vals, list):
            # class 0 = Fooled, class 1 = Correct; show the predicted class
            idx = xgb_pred
            sv = shap_vals[idx][0]
        else:
            sv = shap_vals[0]

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP Value": sv,
            "Feature Value": X_trial.values[0],
        }).sort_values("SHAP Value", key=abs, ascending=False)

        shap_plot_df = shap_df.head(15).copy()
        shap_plot_df["Color"] = shap_plot_df["SHAP Value"].apply(lambda x: "#F44336" if x < 0 else "#4CAF50")

        # Horizontal bar chart using st.bar_chart on a reshaped df
        chart_df = shap_plot_df.set_index("Feature")["SHAP Value"]

        col_shap_chart, col_shap_table = st.columns([3, 2])

        with col_shap_chart:
            st.bar_chart(chart_df, color="#1565C0", horizontal=True)

        with col_shap_table:
            table_rows = []
            for _, r in shap_plot_df.iterrows():
                direction = "→ Fooled" if r["SHAP Value"] < 0 else "→ Correct"
                table_rows.append({
                    "Feature": r["Feature"],
                    "SHAP": f"{r['SHAP Value']:+.3f}",
                    "Value": f"{r['Feature Value']:.1f}",
                    "Direction": direction,
                })
            st.table(pd.DataFrame(table_rows))

        st.caption(f"Base value (expected log-odds): {explainer.expected_value[idx] if isinstance(shap_vals, list) else explainer.expected_value:.3f} &nbsp;|&nbsp; Prediction: {xgb_label_str} (P={xgb_prob[xgb_pred]:.1%})")
    else:
        st.divider()
        st.info("💡 Install `shap` for per-trial SHAP waterfall charts: `pip install shap`")

    st.divider()
    st.caption("GazeGuard v2 | WID2003 Cognitive Science | Tan Shan Chien | [GitHub](https://github.com/TSCHermes/GazeGuard)")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: EVALUATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation":

    st.header("📊 Model Evaluation")
    st.markdown("*5-fold Stratified Grouped CV — all 6 trials from each participant held out together.*")

    # ── Metrics table ────────────────────────────────────────────────────────
    metrics_data = {
        "Metric": [
            "Balanced Accuracy",
            "Raw Accuracy",
            "Recall (Correct)",
            "Recall (Fooled)",
            "F1 (Fooled)",
            "ROC-AUC",
            "Avg Precision (PR)",
        ],
        "XGBoost": ["0.827", "0.774", "0.938", "0.716", "0.782", "0.923", "0.919"],
        "LogReg (L1)": ["0.855", "0.791", "0.873", "0.836", "0.807", "0.913", "0.909"],
    }
    metrics_df = pd.DataFrame(metrics_data)

    mcol1, mcol2 = st.columns([2, 1])
    with mcol1:
        st.subheader("📋 Performance Summary")
        st.table(metrics_df)
    with mcol2:
        st.subheader("🏆 Recommendation")
        st.success("""
**LogReg (L1) recommended**

- Catches more fooled cases: **83.6%** vs 71.6%
- Better balanced accuracy: **0.855** vs 0.827
- Interpretable coefficients
- Faster inference
""")

    st.divider()

    # ── Confusion matrices ───────────────────────────────────────────────────
    st.subheader("🔢 Confusion Matrices")
    cm1, cm2 = st.columns(2)

    with cm1:
        with st.container(border=True):
            st.markdown("**🌲 XGBoost**")
            cm_xgb = np.array([[282, 20], [51, 170]])
            cm_xgb_df = pd.DataFrame(
                cm_xgb,
                index=["True Correct", "True Fooled"],
                columns=["Pred Correct", "Pred Fooled"],
            )
            st.table(cm_xgb_df)

    with cm2:
        with st.container(border=True):
            st.markdown("**📈 LogReg (L1)**")
            cm_lr = np.array([[273, 29], [32, 189]])
            cm_lr_df = pd.DataFrame(
                cm_lr,
                index=["True Correct", "True Fooled"],
                columns=["Pred Correct", "Pred Fooled"],
            )
            st.table(cm_lr_df)

    st.divider()

    # ── Figures ──────────────────────────────────────────────────────────────
    st.subheader("📈 Figures")

    fig_tabs = st.tabs(["Model Comparison", "Feature Importance", "ROC Curves", "Per-Trial Accuracy"])

    with fig_tabs[0]:
        fp = os.path.join(FIG_DIR, "model_comparison.png")
        if os.path.exists(fp):
            st.image(fp, caption="Model comparison — accuracy, balanced accuracy, recall (Fooled), ROC-AUC, and confusion matrices.", use_container_width=True)
        else:
            st.warning("Figure not found: model_comparison.png")

    with fig_tabs[1]:
        fp = os.path.join(FIG_DIR, "feature_importance.png")
        if os.path.exists(fp):
            st.image(fp, caption="Top 15 features for each model. XGBoost uses gain-based importance; LogReg uses |coefficient|.", use_container_width=True)
        else:
            st.warning("Figure not found: feature_importance.png")

    with fig_tabs[2]:
        fp = os.path.join(FIG_DIR, "roc_curves.png")
        if os.path.exists(fp):
            st.image(fp, caption="ROC curves from 5-fold grouped CV.", use_container_width=True)
        else:
            st.warning("Figure not found: roc_curves.png")

    with fig_tabs[3]:
        fp = os.path.join(FIG_DIR, "per_trial_accuracy.png")
        if os.path.exists(fp):
            st.image(fp, caption="Detection accuracy per trial — varies due to image difficulty.", use_container_width=True)
        else:
            st.warning("Figure not found: per_trial_accuracy.png")

    st.divider()

    # ── Interpretation ───────────────────────────────────────────────────────
    st.subheader("🧠 Interpretation")
    st.markdown("""
**Cognitive Science Narrative**

The model learns that participants who **correctly identify** AI-generated images exhibit distinctive gaze patterns:

- **Higher scrutiny on glitch-prone AOIs** — River, Cloud, Leaves, Pupil show larger AI vs. Real fixation differences (`AbsDiff`, `AIratio`)
- **Longer total fixation on AI images** — suggesting deliberate comparison (higher `AI_Proportion`, `Total_Duration_All`)
- **Face_AIratio & Nature_AIratio** — composite features capturing holistic face/nature scrutiny

Participants who are **fooled** tend to distribute gaze more evenly between AI and Real images — their scanpaths don't "flag" the same artifacts.

**Key features (LogReg L1):**
`River_AbsDiff`, `Cloud_AI`, `Leaves_AIratio`, `Pupil_AbsDiff`, `AI_Proportion`

This aligns with **Cognitive Load Theory** — successful detection requires allocating extra attentional resources to regions where AI generation artifacts manifest (facial features, water reflections, cloud boundaries).
""")

    st.divider()
    st.caption("GazeGuard v2 | WID2003 Cognitive Science | Tan Shan Chien | [GitHub](https://github.com/TSCHermes/GazeGuard)")
