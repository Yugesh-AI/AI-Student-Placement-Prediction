import joblib
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocessing import preprocess_data

# ==========================================
# Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "placementdata.csv"

# Create models folder if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

# ==========================================
# Load Preprocessed Data
# ==========================================

X_train, X_test, y_train, y_test = preprocess_data()

# ==========================================
# Load Models
# ==========================================

models = {

    "Logistic Regression":
        joblib.load(MODELS_DIR / "logistic_regression.pkl"),

    "Decision Tree":
        joblib.load(MODELS_DIR / "decision_tree.pkl"),

    "Random Forest":
        joblib.load(MODELS_DIR / "random_forest.pkl")

}

results = []

best_model = None
best_model_name = ""
best_accuracy = 0

# ==========================================
# Evaluate Models
# ==========================================

for name, model in models.items():

    print("=" * 60)
    print(f"Evaluating {name}")
    print("=" * 60)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    results.append({

        "Model": name,

        "Accuracy": round(accuracy, 4),

        "Precision": round(precision, 4),

        "Recall": round(recall, 4),

        "F1 Score": round(f1, 4)

    })

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Placed", "Placed"]
    )

    disp.plot(cmap="Blues")

    plt.title(name)

    plt.show()

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

# ==========================================
# Comparison Table
# ==========================================

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(comparison_df)

comparison_df.to_csv(
    MODELS_DIR / "model_comparison.csv",
    index=False
)

# ==========================================
# Save Best Model
# ==========================================

joblib.dump(
    best_model,
    MODELS_DIR / "best_model.pkl"
)

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model    : {best_model_name}")
print(f"Accuracy : {best_accuracy:.4f}")

print("\nBest model saved as:")
print("models/best_model.pkl")