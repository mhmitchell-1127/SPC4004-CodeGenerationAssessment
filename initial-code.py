import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("heart_disease_uci.csv")

# ── 2. Drop non-feature columns ───────────────────────────────────────────────
# 'id' is just a row identifier; 'dataset' indicates the source clinic —
# neither should be used as a predictive feature.
df = df.drop(columns=["id", "dataset"])

# ── 3. Encode categorical columns ─────────────────────────────────────────────
# The dataset mixes numeric and string columns. We label-encode every
# object/bool column so RandomForest can handle them.
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

le = LabelEncoder()
for col in categorical_cols:
    df[col] = df[col].astype(str)          # normalise booleans → strings first
    df[col] = le.fit_transform(df[col])

# ── 4. Handle missing values ──────────────────────────────────────────────────
# Fill any NaNs with the column median (a safe default for mixed data).
df = df.fillna(df.median(numeric_only=True))

# ── 5. Split features and target ──────────────────────────────────────────────
X = df.drop(columns=["num"])   # 13 input features
y = df["num"]                  # 5-class target (0 = no disease … 4 = severe)

# ── 6. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Classes          : {sorted(y.unique())}\n")

# ── 7. Train RandomForestClassifier ───────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)\n")

print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["0-None", "1-Mild", "2-Moderate", "3-Serious", "4-Severe"],
    zero_division=0
))

# ── 9. Feature importance (bonus) ─────────────────────────────────────────────
importance = pd.Series(clf.feature_importances_, index=X.columns)
print("Top 5 most important features:")
print(importance.sort_values(ascending=False).head(5).to_string())
