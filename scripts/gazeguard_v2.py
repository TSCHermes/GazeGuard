"""
GazeGuard v2 — Per-Trial Prediction of AI-Image Detection
==========================================================
Predicts per-trial Correct/Fooled from gaze features.
Uses grouped CV (all 6 trials from a participant stay in same fold).

Models:
  1. XGBoost (gradient boosted trees)
  2. Logistic Regression (L1-regularized, interpretable)

Author: Tan Shan Chien
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, f1_score, recall_score, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5  # grouped CV folds

# ── 1. Load & merge ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

fd = pd.read_csv(os.path.join(DATA_DIR, "per_trial_fixation_data.csv"))
rl = pd.read_csv(os.path.join(DATA_DIR, "per_trial_response_labels.csv"))

print(f"Fixation data: {fd.shape[0]} rows, {fd['ParticipantID'].nunique()} participants")
print(f"Response data: {rl.shape[0]} rows, {rl['ParticipantID'].nunique()} participants")

# Normalize IDs: extract numeric part
def extract_num(s):
    return int(''.join(ch for ch in str(s) if ch.isdigit()))

fd["PID_num"] = fd["ParticipantID"].map(extract_num)
fd["Trial_num"] = fd["Trial"].map(extract_num)
rl["PID_num"] = rl["ParticipantID"].map(extract_num)
rl["Trial_num"] = rl["Trial"].map(extract_num)

# Drop Total AOI rows (we'll compute our own aggregates)
fd = fd[fd["AOI"] != "Total"].copy()

# Merge labels into fixation data
data = fd.merge(
    rl[["PID_num", "Trial_num", "Response"]],
    on=["PID_num", "Trial_num"],
    how="inner",
)
print(f"Merged: {data.shape[0]} rows")
print(f"Class distribution:\n{data.groupby(['PID_num','Trial_num'])['Response'].first().value_counts()}")

# ── 2. Feature engineering ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature engineering")
print("=" * 60)

# One row per participant-trial
grp = data.groupby(["PID_num", "Trial_num"])

# Pivot: per-AOI, per-ImageType durations
pivot = data.pivot_table(
    index=["PID_num", "Trial_num"],
    columns=["AOI", "ImageType"],
    values="Duration_ms",
    aggfunc="sum",
)
pivot = pivot.fillna(0)
pivot.columns = [f"{aoi}_{img}" for aoi, img in pivot.columns]
pivot = pivot.reset_index()

# Label
labels = grp["Response"].first().reset_index()
labels["y"] = (labels["Response"] == "Correct").astype(int)

# Merge
df = pivot.merge(labels[["PID_num", "Trial_num", "y"]], on=["PID_num", "Trial_num"])

# Engineered features
aoi_names = sorted(set(c.rsplit("_", 1)[0] for c in pivot.columns if c not in ["PID_num", "Trial_num"]))

for aoi in aoi_names:
    ai_col = f"{aoi}_AI"
    real_col = f"{aoi}_Real"
    if ai_col in df.columns and real_col in df.columns:
        # AI/Real ratio (higher = more scrutiny on AI)
        df[f"{aoi}_AIratio"] = (df[ai_col] + 1) / (df[real_col] + 1)
        # Real/AI ratio (higher = more scrutiny on Real)
        df[f"{aoi}_Realratio"] = (df[real_col] + 1) / (df[ai_col] + 1)
        # Absolute difference
        df[f"{aoi}_AbsDiff"] = df[ai_col] - df[real_col]
        # Total on this AOI
        df[f"{aoi}_Total"] = df[ai_col] + df[real_col]

# Trial-level aggregate features
duration_cols = [c for c in df.columns if c.endswith("_AI") or c.endswith("_Real")]
df["Total_Duration_All"] = df[duration_cols].sum(axis=1)
df["Num_AOIs_Fixated"] = (df[duration_cols] > 0).sum(axis=1)
df["Max_AOI_Duration"] = df[duration_cols].max(axis=1)
df["Std_AOI_Duration"] = df[duration_cols].std(axis=1)

# AI bias: proportion of total duration spent on AI
ai_cols = [c for c in df.columns if c.endswith("_AI") and not c.endswith("_AIratio")]
df["AI_Proportion"] = df[ai_cols].sum(axis=1) / (df["Total_Duration_All"] + 1)

# Cross-AOI interaction features (domain-driven)
# Pupil + Skin = face scrutiny
if "Pupil_Total" in df.columns and "Skin_Total" in df.columns:
    df["Face_Total"] = df["Pupil_Total"] + df["Skin_Total"]
    df["Face_AIratio"] = (df.get("Pupil_AI", 0) + df.get("Skin_AI", 0) + 1) / \
                          (df.get("Pupil_Real", 0) + df.get("Skin_Real", 0) + 1)

# Cloud + Water = nature scrutiny
if "Cloud_Total" in df.columns and "Water_Total" in df.columns:
    df["Nature_Total"] = df["Cloud_Total"] + df["Water_Total"]
    df["Nature_AIratio"] = (df.get("Cloud_AI", 0) + df.get("Water_AI", 0) + 1) / \
                            (df.get("Cloud_Real", 0) + df.get("Water_Real", 0) + 1)

# Trial difficulty indicator (from data)
trial_difficulty = df.groupby("Trial_num")["y"].mean()
df["Trial_Accuracy_Rate"] = df["Trial_num"].map(trial_difficulty)

print(f"Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Class balance: {dict(df['y'].value_counts().rename({1:'Correct', 0:'Fooled'}))}")

# Prepare X, y, groups
feature_cols = [c for c in df.columns if c not in ["PID_num", "Trial_num", "y"]]
X = df[feature_cols].copy()
y = df["y"].values
groups = df["PID_num"].values  # for grouped CV

# Fill any NaN/inf
X = X.replace([np.inf, -np.inf], 0).fillna(0)

print(f"Features ({len(feature_cols)}): {feature_cols[:10]}...")

# ── 3. Evaluation framework ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Model training & evaluation")
print("=" * 60)

def evaluate_model(name, model_pipeline, X, y, groups, n_splits):
    """Grouped stratified CV evaluation."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_pred = np.zeros(len(y))
    y_prob = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model_pipeline.fit(X_tr, y_tr)
        y_pred[test_idx] = model_pipeline.predict(X_te)

        # Get probabilities if available
        if hasattr(model_pipeline, 'predict_proba'):
            prob = model_pipeline.predict_proba(X_te)[:, 1]
            y_prob[test_idx] = prob
        elif hasattr(model_pipeline, 'decision_function'):
            y_prob[test_idx] = model_pipeline.decision_function(X_te)

        fold_acc = accuracy_score(y_te, y_pred[test_idx])
        fold_bal = balanced_accuracy_score(y_te, y_pred[test_idx])
        fold_metrics.append({"fold": fold+1, "acc": fold_acc, "bal_acc": fold_bal,
                              "n_test": len(y_te), "n_fooled": (y_te==0).sum()})

    # Overall metrics
    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    f1_fooled = f1_score(y, y_pred, pos_label=0, zero_division=0)
    rec_fooled = recall_score(y, y_pred, pos_label=0, zero_division=0)
    rec_correct = recall_score(y, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[1, 0])

    # AUC
    try:
        auc = roc_auc_score(y, y_prob)
        avg_prec = average_precision_score(y, y_prob)
    except Exception:
        auc = 0.0
        avg_prec = 0.0

    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")
    print(f"Raw accuracy        : {acc:.3f}")
    print(f"Balanced accuracy   : {bal:.3f}")
    print(f"Recall (Correct)    : {rec_correct:.3f}")
    print(f"Recall (Fooled)     : {rec_fooled:.3f}")
    print(f"F1 (Fooled)         : {f1_fooled:.3f}")
    print(f"ROC-AUC             : {auc:.3f}")
    print(f"Avg Precision (PR)  : {avg_prec:.3f}")
    print(f"\nConfusion matrix (rows=true, cols=pred) [Correct, Fooled]:")
    print(f"            pred_Correct  pred_Fooled")
    print(f"true_Correct {cm[0,0]:>11}  {cm[0,1]:>11}")
    print(f"true_Fooled  {cm[1,0]:>11}  {cm[1,1]:>11}")
    print(f"\n{classification_report(y, y_pred, target_names=['Fooled', 'Correct'], zero_division=0)}")

    print("Fold details:")
    for fm in fold_metrics:
        print(f"  Fold {fm['fold']}: acc={fm['acc']:.3f}, bal={fm['bal_acc']:.3f}, "
              f"n_test={fm['n_test']}, n_fooled={fm['n_fooled']}")

    return {
        "name": name, "y_pred": y_pred, "y_prob": y_prob,
        "acc": acc, "bal_acc": bal, "f1_fooled": f1_fooled,
        "rec_fooled": rec_fooled, "rec_correct": rec_correct,
        "auc": auc, "avg_prec": avg_prec, "cm": cm,
        "fold_metrics": fold_metrics,
    }


# ── Model 1: XGBoost ────────────────────────────────────────────────────────
print("\n--- Training XGBoost ---")

# Compute scale_pos_weight for class imbalance
n_correct = (y == 1).sum()
n_fooled = (y == 0).sum()
scale_pos_weight = n_correct / n_fooled
print(f"Class ratio (Correct/Fooled): {n_correct}/{n_fooled} = {scale_pos_weight:.2f}")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=2,
    verbosity=0,
)

xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb_model),
])

xgb_results = evaluate_model("XGBoost", xgb_pipeline, X, y, groups, N_SPLITS)


# ── Model 2: Logistic Regression (L1) ───────────────────────────────────────
print("\n--- Training Logistic Regression ---")

lr_model = LogisticRegressionCV(
    Cs=10,
    cv=3,
    penalty="l1",
    solver="saga",
    class_weight="balanced",
    max_iter=5000,
    random_state=RANDOM_STATE,
    scoring="balanced_accuracy",
    n_jobs=2,
)

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", lr_model),
])

lr_results = evaluate_model("Logistic Regression (L1)", lr_pipeline, X, y, groups, N_SPLITS)


# ── 4. Feature importance comparison ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Feature importance")
print("=" * 60)

# XGBoost feature importance
xgb_fitted = xgb_pipeline.named_steps["model"]
xgb_imp = pd.Series(xgb_fitted.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 15 XGBoost features:")
for feat, val in xgb_imp.head(15).items():
    print(f"  {feat:30s} {val:.4f}")

# LR coefficients (absolute)
lr_fitted = lr_pipeline.named_steps["model"]
lr_coef = pd.Series(np.abs(lr_fitted.coef_[0]), index=feature_cols).sort_values(ascending=False)
print("\nTop 15 Logistic Regression features (|coef|):")
for feat, val in lr_coef.head(15).items():
    print(f"  {feat:30s} {val:.4f}")

# Also show signed LR coefficients for direction
lr_signed = pd.Series(lr_fitted.coef_[0], index=feature_cols).sort_values()
print("\nLR coefficients (negative = predicts Fooled, positive = predicts Correct):")
for feat, val in lr_signed.head(10).items():
    print(f"  {feat:30s} {val:+.4f}")
print("  ...")
for feat, val in lr_signed.tail(10).items():
    print(f"  {feat:30s} {val:+.4f}")


# ── 5. Plots ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating plots")
print("=" * 60)

# Plot 1: Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["acc", "bal_acc", "rec_fooled", "auc"]
labels_plot = ["Accuracy", "Balanced Acc", "Recall (Fooled)", "ROC-AUC"]
xgb_vals = [xgb_results[m] for m in metrics]
lr_vals = [lr_results[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35
axes[0].bar(x - width/2, xgb_vals, width, label="XGBoost", color="#2E75B6", edgecolor="#1F3864")
axes[0].bar(x + width/2, lr_vals, width, label="LogReg (L1)", color="#ED7D31", edgecolor="#7F2E00")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_plot, rotation=20, ha="right")
axes[0].set_ylabel("Score")
axes[0].set_title("Model Comparison")
axes[0].legend()
axes[0].set_ylim(0, 1.05)
for i, (xv, lv) in enumerate(zip(xgb_vals, lr_vals)):
    axes[0].text(i - width/2, xv + 0.02, f"{xv:.3f}", ha="center", fontsize=8)
    axes[0].text(i + width/2, lv + 0.02, f"{lv:.3f}", ha="center", fontsize=8)

# Plot 2: Confusion matrices
for idx, (res, title) in enumerate([(xgb_results, "XGBoost"), (lr_results, "LogReg (L1)")]):
    ax = axes[idx + 1]
    cm = res["cm"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Correct", "Fooled"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Correct", "Fooled"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {title}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontweight="bold", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: model_comparison.png")

# Plot 3: Feature importance (top 15, side by side)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

top_n = 15
xgb_top = xgb_imp.head(top_n)[::-1]
axes[0].barh(range(top_n), xgb_top.values, color="#2E75B6", edgecolor="#1F3864")
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(xgb_top.index, fontsize=9)
axes[0].set_xlabel("Feature Importance (gain)")
axes[0].set_title(f"XGBoost — Top {top_n} Features")

lr_top = lr_coef.head(top_n)[::-1]
axes[1].barh(range(top_n), lr_top.values, color="#ED7D31", edgecolor="#7F2E00")
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(lr_top.index, fontsize=9)
axes[1].set_xlabel("|Coefficient|")
axes[1].set_title(f"Logistic Regression — Top {top_n} Features")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")

# Plot 4: Per-trial performance
fig, ax = plt.subplots(figsize=(8, 5))
trial_perf = df.groupby("Trial_num").agg(
    accuracy=("y", "mean"),
    n=("y", "count"),
    n_fooled=("y", lambda x: (x == 0).sum()),
).reset_index()
trial_perf["trial_label"] = "Trial " + trial_perf["Trial_num"].astype(str)

bars = ax.bar(trial_perf["trial_label"], trial_perf["accuracy"], color="#2E75B6", edgecolor="#1F3864")
ax.set_ylabel("Correct Rate")
ax.set_title("Per-Trial Detection Accuracy")
ax.set_ylim(0, 1.05)
for bar, row in zip(bars, trial_perf.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{row.accuracy:.1%}\n({row.n_fooled}F/{row.n-row.n_fooled}C)",
            ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "per_trial_accuracy.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: per_trial_accuracy.png")

# Plot 5: ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for res, name, color in [(xgb_results, "XGBoost", "#2E75B6"), (lr_results, "LogReg (L1)", "#ED7D31")]:
    if res["y_prob"] is not None:
        fpr, tpr, _ = roc_curve(y, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color, linewidth=2)

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (Grouped CV)")
ax.legend()
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: roc_curves.png")


# ── 6. Save results ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving results")
print("=" * 60)

# Save feature matrix with labels
df.to_csv(os.path.join(OUT_DIR, "feature_matrix.csv"), index=False)
print("Saved: feature_matrix.csv")

# Save predictions
pred_df = df[["PID_num", "Trial_num", "y"]].copy()
pred_df["XGB_pred"] = xgb_results["y_pred"].astype(int)
pred_df["XGB_prob"] = xgb_results["y_prob"]
pred_df["LR_pred"] = lr_results["y_pred"].astype(int)
pred_df["LR_prob"] = lr_results["y_prob"]
pred_df.to_csv(os.path.join(OUT_DIR, "cv_predictions.csv"), index=False)
print("Saved: cv_predictions.csv")

# Save feature importances
imp_df = pd.DataFrame({
    "feature": feature_cols,
    "xgb_importance": xgb_imp.reindex(feature_cols).values,
    "lr_abs_coef": lr_coef.reindex(feature_cols).values,
    "lr_signed_coef": lr_signed.reindex(feature_cols).values,
}).sort_values("xgb_importance", ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
print("Saved: feature_importance.csv")

# Save metrics summary
with open(os.path.join(OUT_DIR, "metrics_report.txt"), "w") as f:
    f.write("GazeGuard v2 — Per-Trial Prediction Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: {len(df)} trials × {len(feature_cols)} features\n")
    f.write(f"Participants: {df['PID_num'].nunique()}\n")
    f.write(f"Class balance: {dict(df['y'].value_counts().rename({1:'Correct', 0:'Fooled'}))}\n")
    f.write(f"CV: {N_SPLITS}-fold StratifiedGroupKFold (grouped by participant)\n\n")

    for res in [xgb_results, lr_results]:
        f.write(f"\n{'='*50}\n")
        f.write(f"MODEL: {res['name']}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Raw accuracy        : {res['acc']:.3f}\n")
        f.write(f"Balanced accuracy   : {res['bal_acc']:.3f}\n")
        f.write(f"Recall (Correct)    : {res['rec_correct']:.3f}\n")
        f.write(f"Recall (Fooled)     : {res['rec_fooled']:.3f}\n")
        f.write(f"F1 (Fooled)         : {res['f1_fooled']:.3f}\n")
        f.write(f"ROC-AUC             : {res['auc']:.3f}\n")
        f.write(f"Avg Precision       : {res['avg_prec']:.3f}\n")
        f.write(f"Confusion matrix:\n{res['cm']}\n")

    f.write(f"\n\nTop 15 XGBoost features:\n")
    for feat, val in xgb_imp.head(15).items():
        f.write(f"  {feat:30s} {val:.4f}\n")

    f.write(f"\n\nTop 15 LR features (|coef|):\n")
    for feat, val in lr_coef.head(15).items():
        f.write(f"  {feat:30s} {val:.4f}\n")

    f.write(f"\n\nLR signed coefficients (direction):\n")
    for feat, val in lr_signed.items():
        direction = "→ Correct" if val > 0 else "→ Fooled"
        f.write(f"  {feat:30s} {val:+.4f}  {direction}\n")

print("Saved: metrics_report.txt")

# ── 7. Save trained models with joblib ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Saving trained models")
print("=" * 60)

# Fit final models on ALL data
print("Fitting final XGBoost on all data...")
xgb_pipeline.fit(X, y)
joblib.dump(xgb_pipeline, os.path.join(OUT_DIR, "xgboost_model.joblib"))
print("Saved: xgb_model.joblib")

print("Fitting final Logistic Regression on all data...")
lr_pipeline.fit(X, y)
joblib.dump(lr_pipeline, os.path.join(OUT_DIR, "logreg_model.joblib"))
print("Saved: logreg_model.joblib")

# Save feature column names for inference
import json
with open(os.path.join(OUT_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f)
print("Saved: feature_columns.json")

print(f"\n{'='*60}")
print(f"ALL DONE. Outputs in: {OUT_DIR}/")
print(f"{'='*60}")
