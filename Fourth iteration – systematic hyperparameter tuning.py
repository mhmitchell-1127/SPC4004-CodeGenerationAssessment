"""
Heart Disease Severity Classifier
==================================
UCI Heart Disease Dataset  |  5-class target: num (0–4)
Model: Random Forest inside a sklearn Pipeline with ColumnTransformer
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. Load & clean
# ─────────────────────────────────────────────
print("=" * 65)
print("  Heart Disease Severity Classifier  –  UCI Dataset")
print("=" * 65)

df = pd.read_csv("heart_disease_uci.csv")

# Drop administrative columns that are not clinical features
df.drop(columns=["id", "dataset"], inplace=True)

# Treat bool-typed columns as strings so they go through
# the categorical branch consistently
for col in df.select_dtypes(include="bool").columns:
    df[col] = df[col].astype(str)

TARGET = "num"
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Class distribution:\n{y.value_counts().sort_index().to_string()}\n")

# ─────────────────────────────────────────────
# 2. Identify numeric vs. categorical columns
# ─────────────────────────────────────────────
numeric_cols     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

print(f"Numeric features    ({len(numeric_cols)})  : {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}) : {categorical_cols}\n")

# ─────────────────────────────────────────────
# 3. Preprocessing — ColumnTransformer
# ─────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# ─────────────────────────────────────────────
# 4. Full Pipeline (preprocessor + classifier)
# ─────────────────────────────────────────────
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(random_state=42, n_jobs=-1)),
])

# ─────────────────────────────────────────────
# 5. Train / test split (stratified)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}\n")

# ─────────────────────────────────────────────
# 6. Hyper-parameter search (GridSearchCV)
# ─────────────────────────────────────────────
print("Running GridSearchCV … (this may take ~1 minute)")

param_grid = {
    "classifier__n_estimators"    : [100, 200, 300],
    "classifier__max_depth"       : [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator  = pipeline,
    param_grid = param_grid,
    cv         = cv_strategy,
    scoring    = "accuracy",
    n_jobs     = -1,
    verbose    = 0,
    refit      = True,          # refit best estimator on full training set
)
grid_search.fit(X_train, y_train)

print("\n--- GridSearchCV Results ---")
print(f"  Best parameters   : {grid_search.best_params_}")
print(f"  Best CV accuracy  : {grid_search.best_score_:.4f}  "
      f"({grid_search.best_score_ * 100:.2f} %)\n")

# Best pipeline (already refit on full X_train by GridSearchCV refit=True)
best_pipeline = grid_search.best_estimator_

# ─────────────────────────────────────────────
# 7. 5-fold cross-validation on training set
# ─────────────────────────────────────────────
print("--- 5-Fold Cross-Validation on Training Set (best pipeline) ---")
cv_scores = cross_val_score(
    best_pipeline, X_train, y_train,
    cv=cv_strategy, scoring="accuracy", n_jobs=-1
)
for fold_i, score in enumerate(cv_scores, 1):
    print(f"  Fold {fold_i}: {score:.4f}")
print(f"  Mean   : {cv_scores.mean():.4f}")
print(f"  Std    : {cv_scores.std():.4f}\n")

# ─────────────────────────────────────────────
# 8. Evaluation on held-out test set
# ─────────────────────────────────────────────
y_pred = best_pipeline.predict(X_test)

class_names = ["0 – None", "1 – Mild", "2 – Moderate",
               "3 – Serious", "4 – Severe"]

print("--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# ─────────────────────────────────────────────
# 9. Confusion Matrix heatmap
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(
    "Figure 1: Confusion Matrix --- Random Forest on UCI Heart Disease Dataset",
    fontsize=11, fontweight="bold", pad=12,
)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved → confusion_matrix.png")

# ─────────────────────────────────────────────
# 10. Feature Importances bar chart
# ─────────────────────────────────────────────
rf_model     = best_pipeline.named_steps["classifier"]
ohe          = (best_pipeline
                .named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["onehot"])

ohe_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
all_feature_names = numeric_cols + ohe_feature_names

importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)
top10       = importances.nlargest(10).sort_values()          # ascending for barh

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(
    top10.index,
    top10.values,
    color=plt.cm.Blues(np.linspace(0.45, 0.85, len(top10))),
    edgecolor="white",
    height=0.65,
)

# Annotate each bar with its importance value
for bar, val in zip(bars, top10.values):
    ax.text(
        val + 0.001, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left", fontsize=8.5,
    )

ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.set_title(
    "Figure 2: Top 10 Feature Importances --- Random Forest Classifier",
    fontsize=11, fontweight="bold", pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.close()
print("Feature importances saved → feature_importances.png")
print("\nDone.")
