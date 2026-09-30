"""
DecodeLabs AI Internship - Project 2: Data Classification Using AI
Goal: Build a basic classification model on a small dataset.
Steps: 1) Load & understand data  2) Train/test split  3) Train classifier  4) Evaluate
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ---------- 1. LOAD & UNDERSTAND THE DATASET ----------
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum().sum(), "(none)")
print("\nClass balance:\n", df["species"].value_counts())
print("\nSummary statistics:\n", df.describe().round(2))

X, y = iris.data, iris.target

# ---------- 2. SPLIT INTO TRAINING & TESTING SETS ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ---------- 3. TRAIN CLASSIFIERS ----------
models = {
    "K-Nearest Neighbors (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree":             DecisionTreeClassifier(max_depth=3, random_state=42),
    "Logistic Regression":       make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    cv = cross_val_score(model, X, y, cv=5).mean()
    results[name] = (acc, cv)
    print(f"{name:28s} Test acc: {acc:.3f} | 5-fold CV acc: {cv:.3f}")

# ---------- 4. EVALUATE BEST MODEL ----------
best_name = max(results, key=lambda k: results[k][1])
best = models[best_name]
pred = best.predict(X_test)
print(f"\nBest model (by CV): {best_name}")
print("\nClassification report:\n",
      classification_report(y_test, pred, target_names=iris.target_names))

ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=iris.target_names).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# ---------- 5. PREDICT ON COMPLETELY NEW DATA ----------
new = pd.DataFrame([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]], columns=iris.feature_names)
for row, p in zip(new.values, best.predict(new)):
    print("New sample", row, "->", iris.target_names[p])
