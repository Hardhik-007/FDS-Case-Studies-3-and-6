"""
Case Study 6: Comparative Analysis of Classification Algorithms for Bank Marketing Prediction
Run from repository root: python bank_marketing_analysis.py
Dataset: data/bank-full.csv (semicolon-delimited)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-full.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
df = pd.read_csv(DATA_PATH, sep=";")
print("Shape:", df.shape)
print(df.head(10))
print(df.dtypes)
print(df.describe(include="all").T)
print("Duplicates:", df.duplicated().sum())
print("Target counts:\n", df["y"].value_counts())
print("Target percentages:\n", df["y"].value_counts(normalize=True).mul(100).round(2))
print("Unknown values by column:\n", (df == "unknown").sum().sort_values(ascending=False))

# EDA: duration is excluded from the predictive model; it may be summarized only as a historical feature.
counts = df["y"].value_counts().reindex(["no", "yes"])
plt.figure(figsize=(6, 4))
plt.bar(["No subscription", "Subscribed"], counts.values)
plt.title("Bank Term Deposit Subscription Class Distribution")
plt.xlabel("Subscription outcome"); plt.ylabel("Number of customers")
plt.tight_layout(); plt.savefig(FIG_DIR / "target_distribution.png", dpi=150); plt.close()

rates = df.groupby("contact")["y"].apply(lambda s: (s == "yes").mean() * 100).sort_values(ascending=False)
plt.figure(figsize=(7, 4))
plt.bar(rates.index, rates.values)
plt.title("Subscription Rate by Contact Channel")
plt.xlabel("Contact channel"); plt.ylabel("Customers subscribed (%)")
plt.tight_layout(); plt.savefig(FIG_DIR / "subscription_by_contact.png", dpi=150); plt.close()

for label in ["no", "yes"]:
    plt.hist(df.loc[df.y == label, "age"].clip(upper=85), bins=30, alpha=.55, density=True, label=f"y={label}")
plt.title("Age Distribution by Term Deposit Outcome")
plt.xlabel("Age (years; clipped at 85 for display)"); plt.ylabel("Density"); plt.legend()
plt.tight_layout(); plt.savefig(FIG_DIR / "age_by_outcome.png", dpi=150); plt.close()

# IMPORTANT: duration is intentionally excluded for realistic pre-contact prediction.
X = df.drop(columns=["y", "duration"])
y = (df["y"] == "yes").astype(int)
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y
)
numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, min_samples_leaf=10),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1),
}
rows, fitted, predictions, probabilities = [], {}, {}, {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    fitted[name], predictions[name], probabilities[name] = pipe, pred, prob
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred, zero_division=0),
                 "Recall": recall_score(y_test, pred, zero_division=0),
                 "F1-score": f1_score(y_test, pred, zero_division=0),
                 "ROC-AUC": roc_auc_score(y_test, prob)})
results = pd.DataFrame(rows).sort_values(["F1-score", "ROC-AUC"], ascending=False)
print("\nModel comparison:\n", results.to_string(index=False))
results.to_csv(ROOT / "model_comparison.csv", index=False)

# Common metrics comparison chart
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
ax = results.set_index("Model")[metrics].plot(kind="bar", figsize=(9, 5))
ax.set_title("Classification Model Comparison on the Held-Out Test Set")
ax.set_xlabel("Classifier"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
plt.xticks(rotation=15, ha="right"); plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison.png", dpi=160); plt.close()

# Confusion matrices and ROC curves
plt.figure(figsize=(7.5, 5.5))
for name in models:
    cm = confusion_matrix(y_test, predictions[name])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probabilities[name]):.3f})")
plt.figure(1)
plt.plot([0, 1], [0, 1], "--", label="Chance")
plt.title("ROC Curves for Four Classifiers")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(FIG_DIR / "roc_curves.png", dpi=160); plt.close()
print("\nPrimary model predictors exclude duration:", "duration" not in X.columns)
print("Interpretation tip: compare positive-class recall, precision and F1 with ROC-AUC; accuracy alone is insufficient for this imbalanced target.")
