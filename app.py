"""
GazeGuard — Interactive Demo
=============================
Streamlit app that lets you select a participant-trial from the dataset
and see the model's prediction of whether they were Correct or Fooled.

Run with: streamlit run app.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "output")


# ── Feature engineering (must match training exactly) ───────────────────────
def engineer_features_from_fixation(fd_trial, feature_cols, trial_difficulty_map):
    """
    Engineer the 82 features from raw fixation data for one trial.
    fd_trial: DataFrame with columns [AOI, ImageType, Duration_ms, Trial_num]
    """
    pivot = fd_trial.pivot_table(
        columns=["AOI", "ImageType"],
        values="Duration_ms",
        aggfunc="sum",
    )

    row = {}
    for (aoi, img), val in pivot.items():
        row[f"{aoi}_{img}"] = float(val) if not pd.isna(val) else 0.0

    df = pd.DataFrame([row])

    # Discover base AOI names from feature columns
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

    ai_cols = [c for c in df.columns if c.endswith("_AI") and not c.endswith("_AIratio")]
    df["AI_Proportion"] = df[ai_cols].sum(axis=1) / (df["Total_Duration_All"] + 1)

    df["Face_Total"] = df.get("Pupil_Total", 0).iloc[0] + df.get("Skin_Total", 0).iloc[0]
    df["Face_AIratio"] = (df.get("Pupil_AI", 0).iloc[0] + df.get("Skin_AI", 0).iloc[0] + 1) / \
                          (df.get("Pupil_Real", 0).iloc[0] + df.get("Skin_Real", 0).iloc[0] + 1)
    df["Nature_Total"] = df.get("Cloud_Total", 0).iloc[0] + df.get("Water_Total", 0).iloc[0]
    df["Nature_AIratio"] = (df.get("Cloud_AI", 0).iloc[0] + df.get("Water_AI", 0).iloc[0] + 1) / \
                            (df.get("Cloud_Real", 0).iloc[0] + df.get("Water_Real", 0).iloc[0] + 1)

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

st.title("👁️ GazeGuard — AI Image Detection Predictor")
st.markdown("""
**WID2003 Cognitive Science | Eye Wonder Project**

Predicts whether a participant will **correctly identify** an AI-generated image
or be **fooled** by it — using only their eye-tracking data.
""")

# ── Load ────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading models and data... (first load ~30s)"):
    xgb_model, lr_model, feature_cols = load_models()
    data = load_trial_data()

trial_difficulty = data.groupby("Trial_num")["Response"].apply(
    lambda x: (x == "Correct").mean()
).to_dict()

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Select Trial")

trials_df = data.groupby(["PID_num", "Trial_num"]).agg(
    Response=("Response", "first"),
    Participant=("ParticipantID", "first"),
    Trial=("Trial", "first"),
).reset_index()

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

# ── 3-column layout ────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📊 Gaze Data")

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
    c1.metric("AI", f"{total_ai:,} ms")
    c2.metric("Real", f"{total_real:,} ms")
    c3.metric("AI %", f"{total_ai/(total_ai+total_real)*100:.1f}%")
    c4.metric("AOIs", f"{n_aois}/12")

with col2:
    st.subheader("🌲 XGBoost")

    st.markdown(f"**P(Fooled): {xgb_prob[0]:.1%}**")
    st.progress(xgb_prob[0])
    st.markdown(f"**P(Correct): {xgb_prob[1]:.1%}**")
    st.progress(xgb_prob[1])

    st.markdown(f"**Prediction:** {'✅ Correct' if xgb_pred == 1 else '❌ Fooled'}")
    st.markdown(f"**Actual:** {'✅ Correct' if true_label == 'Correct' else '❌ Fooled'}")
    match = (xgb_pred == 1 and true_label == "Correct") or (xgb_pred == 0 and true_label == "Fooled")
    st.markdown(f"**{'✅ Right' if match else '❌ Wrong'}**")

with col3:
    st.subheader("📈 LogReg (L1)")

    st.markdown(f"**P(Fooled): {lr_prob[0]:.1%}**")
    st.progress(lr_prob[0])
    st.markdown(f"**P(Correct): {lr_prob[1]:.1%}**")
    st.progress(lr_prob[1])

    st.markdown(f"**Prediction:** {'✅ Correct' if lr_pred == 1 else '❌ Fooled'}")
    st.markdown(f"**Actual:** {'✅ Correct' if true_label == 'Correct' else '❌ Fooled'}")
    match = (lr_pred == 1 and true_label == "Correct") or (lr_pred == 0 and true_label == "Fooled")
    st.markdown(f"**{'✅ Right' if match else '❌ Wrong'}**")

# ── Explanation ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🧠 Why This Prediction?")

lr_fitted = lr_model.named_steps["model"]
lr_coef = pd.Series(lr_fitted.coef_[0], index=feature_cols)

ec1, ec2 = st.columns(2)
with ec1:
    st.markdown("**Pushing → ✅ Correct:**")
    for feat, coef in lr_coef.nlargest(5).items():
        val = X_trial[feat].iloc[0]
        st.markdown(f"- `{feat}` = **{val:.1f}** (coef +{coef:.3f})")
with ec2:
    st.markdown("**Pushing → ❌ Fooled:**")
    for feat, coef in lr_coef.nsmallest(5).items():
        val = X_trial[feat].iloc[0]
        st.markdown(f"- `{feat}` = **{val:.1f}** (coef {coef:.3f})")

# ── Performance table ───────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Model Performance (5-fold Grouped CV)")

st.table(pd.DataFrame({
    "Metric": ["Balanced Accuracy", "Recall (Fooled)", "Recall (Correct)", "F1 (Fooled)", "ROC-AUC"],
    "XGBoost": ["0.827", "0.716", "0.938", "0.782", "0.923"],
    "LogReg (L1)": ["0.855", "0.836", "0.873", "0.807", "0.913"],
}))

st.info("""
**LogReg recommended** — catches more fooled cases (83.6% vs 71.6%), better balanced accuracy, interpretable.
Key features: `River_AbsDiff`, `Cloud_AI`, `Leaves_AIratio` — scrutinizing glitch-prone AOIs on AI images
predicts correct detection (Cognitive Load Theory).
""")

st.divider()
st.caption("GazeGuard v2 | WID2003 Cognitive Science | Tan Shan Chien | [GitHub](https://github.com/TSCHermes/GazeGuard)")
