import os
from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess_data

# ===============================
# Project Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "placementdata.csv"
MODELS_DIR.mkdir(exist_ok=True)

# ===============================
# Feature Names
# ===============================

FEATURE_NAMES = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
    "CodingScore",
    "CommunicationScore"
]


def train_models():

    print("=" * 60)
    print("Loading Preprocessed Data...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = preprocess_data()

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )

    }

    trained_models = {}

    print("\nTraining Models...\n")

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    # ============================
    # Save Models
    # ============================

    joblib.dump(
        trained_models["Logistic Regression"],
        MODELS_DIR / "logistic_regression.pkl"
    )

    joblib.dump(
        trained_models["Decision Tree"],
        MODELS_DIR / "decision_tree.pkl"
    )

    joblib.dump(
        trained_models["Random Forest"],
        MODELS_DIR / "random_forest.pkl"
    )

    # ============================
    # Save Feature Importance
    # ============================

    rf = trained_models["Random Forest"]

    feature_importance = pd.DataFrame({

        "Feature": FEATURE_NAMES,

        "Importance": rf.feature_importances_

    })

    feature_importance.sort_values(
        by="Importance",
        ascending=False,
        inplace=True
    )

    feature_importance.to_csv(
        MODELS_DIR / "feature_importance.csv",
        index=False
    )

    print("\nFeature Importance\n")
    print(feature_importance)

    print("\nModels Saved Successfully!")

    return trained_models


if __name__ == "__main__":

    train_models()