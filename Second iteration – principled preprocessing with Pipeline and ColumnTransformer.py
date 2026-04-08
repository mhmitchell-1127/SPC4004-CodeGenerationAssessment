"""
Heart Disease Severity Classifier
UCI Heart Disease Dataset — Random Forest with sklearn Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("heart_disease_uci.csv")

# Drop non-informative identifier and source columns
df = df.drop(columns=["id", "dataset"], errors="ignore")

TARGET = "num"
CLASS_NAMES = ["0 - None", "1 - Mild", "2 - Moderate", "3 - Serious", "4 - Severe"]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ── 2. Identify feature types ─────────────────────────────────────────────────
# Boolean-like columns stored as strings need to be treated as categorical
bool_like = [c for c in X.columns if X[c].dropna().isin([True, False, "TRUE", "FALSE"]).all()]
numeric_cols = [c for c in X.select_dtypes(include=["number"]).columns if c not in bool_like]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print("Numeric features  :", numeric_cols)
print("Categorical features:", categorical_cols)

# ── 3. Preprocessing + model pipeline ────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,  numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",   # handles class imbalance
        random_state=42,
    )),
])

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ── 5. Classification report ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_test, y_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=CLASS_NAMES,
    zero_division=0,
))

# ── 6. Confusion matrix heatmap ───────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(
    "Figure 1: Confusion Matrix — Random Forest on UCI Heart Disease Dataset",
    fontsize=11, pad=14,
)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0,  fontsize=9)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved → confusion_matrix.png")

# ── 7. Feature importances ────────────────────────────────────────────────────
rf = pipeline.named_steps["classifier"]
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
all_feature_names = numeric_cols + cat_feature_names

importances = pd.Series(rf.feature_importances_, index=all_feature_names)
top10 = importances.nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.85, len(top10)))
top10.plot(kind="barh", ax=ax, color=colors, edgecolor="grey", linewidth=0.5)
ax.set_xlabel("Mean Decrease in Impurity (Importance)", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.set_title(
    "Figure 2: Top 10 Feature Importances — Random Forest Classifier",
    fontsize=11, pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(ax.patches, top10.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.close()
print("Saved → feature_importances.png")
