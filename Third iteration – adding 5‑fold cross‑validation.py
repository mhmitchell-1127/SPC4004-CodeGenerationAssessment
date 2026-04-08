"""
Heart Disease Severity Classifier
UCI Heart Disease Dataset — Random Forest (5-class prediction)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ── 0. Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = "heart_disease_uci.csv"
df = pd.read_csv(DATA_PATH)

# Drop non-feature columns (id = row index, dataset = source hospital)
df = df.drop(columns=["id", "dataset"])

TARGET = "num"
CLASS_NAMES = ["0 - None", "1 - Mild", "2 - Moderate", "3 - Serious", "4 - Severe"]

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Dataset shape  : {df.shape}")
print(f"Class distribution:\n{y.value_counts().sort_index()}\n")

# ── 2. Identify column types ──────────────────────────────────────────────────
#  Numeric  : continuous / ordinal integer columns
#  Categorical : string / boolean / object columns that need OHE
NUMERIC_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

print("Numeric features    :", NUMERIC_COLS)
print("Categorical features:", CATEGORICAL_COLS, "\n")

# ── 3. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}\n")

# ── 4. Preprocessing via ColumnTransformer ────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_COLS),
    ("cat", categorical_transformer, CATEGORICAL_COLS),
])

# ── 5. Full Pipeline (preprocessor + classifier) ──────────────────────────────
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",   # compensate for class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

# ── 6. Train ──────────────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)
print("Model trained successfully.\n")

# ── 7. Classification report ──────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

print("=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ── 8. 5-fold cross-validation on training set ────────────────────────────────
print("=" * 65)
print("5-FOLD CROSS-VALIDATION (training set, accuracy)")
print("=" * 65)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
for fold_idx, score in enumerate(cv_scores, start=1):
    print(f"  Fold {fold_idx}: {score:.4f}")
print(f"  Mean : {cv_scores.mean():.4f}")
print(f"  Std  : {cv_scores.std():.4f}\n")

# ── 9. Confusion matrix heatmap ───────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])

fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
    linecolor="grey",
    ax=ax_cm,
    annot_kws={"size": 12, "weight": "bold"},
)
ax_cm.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax_cm.set_ylabel("True Label", fontsize=12, labelpad=10)
ax_cm.set_title(
    "Figure 1: Confusion Matrix --- Random Forest on UCI Heart Disease Dataset",
    fontsize=12, pad=14, weight="bold",
)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
fig_cm.savefig("confusion_matrix.png", dpi=150)
plt.close(fig_cm)
print("Saved → confusion_matrix.png")

# ── 10. Feature importances ───────────────────────────────────────────────────
rf_model = pipeline.named_steps["classifier"]
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
ohe_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
all_feature_names = NUMERIC_COLS + ohe_feature_names

importances = rf_model.feature_importances_
fi_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
top10 = fi_series.head(10).sort_values(ascending=True)   # ascending → horizontal bar goes left→right

fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
bars = ax_fi.barh(
    top10.index,
    top10.values,
    color=sns.color_palette("mako_r", len(top10)),
    edgecolor="white",
    height=0.65,
)
ax_fi.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
ax_fi.set_xlabel("Mean Decrease in Impurity (Gini importance)", fontsize=11, labelpad=8)
ax_fi.set_title(
    "Figure 2: Top 10 Feature Importances --- Random Forest Classifier",
    fontsize=12, pad=14, weight="bold",
)
ax_fi.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax_fi.set_xlim(0, top10.values.max() * 1.18)
ax_fi.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig_fi.savefig("feature_importances.png", dpi=150)
plt.close(fig_fi)
print("Saved → feature_importances.png")
