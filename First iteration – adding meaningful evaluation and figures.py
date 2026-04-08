import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── 1. Load & inspect ────────────────────────────────────────────────────────
df = pd.read_csv("heart_disease_uci.csv")

# Drop non-informative columns
df = df.drop(columns=["id", "dataset"], errors="ignore")

# ── 2. Pre-process ───────────────────────────────────────────────────────────
# Encode boolean-like text columns
bool_cols = ["fbs", "exang"]
for col in bool_cols:
    df[col] = df[col].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})

# Encode remaining categorical columns with LabelEncoder
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "num"]

le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])

# Drop rows with missing values
df = df.dropna()

# ── 3. Split features / target ───────────────────────────────────────────────
X = df.drop(columns=["num"])
y = df["num"].astype(int)

feature_names = X.columns.tolist()
class_names   = ["0 - None", "1 - Mild", "2 - Moderate", "3 - Serious", "4 - Severe"]

# ── 4. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Train RandomForest ────────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# ── 6. Classification report ─────────────────────────────────────────────────
print("=" * 65)
print("  Classification Report — Random Forest on UCI Heart Disease")
print("=" * 65)
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

# ── 7. Confusion matrix heatmap ──────────────────────────────────────────────
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
    linecolor="grey",
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax.set_ylabel("True Label", fontsize=12, labelpad=10)
ax.set_title(
    "Figure 1: Confusion Matrix\u2014Random Forest on UCI Heart Disease Dataset",
    fontsize=13, pad=14
)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved → confusion_matrix.png")

# ── 8. Feature importances bar chart ─────────────────────────────────────────
importances = clf.feature_importances_
indices     = np.argsort(importances)[::-1][:10]          # top-10
top_features = [feature_names[i] for i in indices]
top_scores   = importances[indices]

fig, ax = plt.subplots(figsize=(9, 5))
colours = sns.color_palette("viridis", len(top_features))
bars = ax.barh(
    range(len(top_features)),
    top_scores[::-1],          # reverse so highest is at top
    color=colours[::-1],
    edgecolor="white",
    height=0.65,
)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features[::-1], fontsize=10)
ax.set_xlabel("Mean Decrease in Impurity (Importance)", fontsize=11)
ax.set_title(
    "Figure 2: Top 10 Feature Importances\u2014Random Forest Classifier",
    fontsize=13, pad=14
)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.close()
print("Saved → feature_importances.png")
