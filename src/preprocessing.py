import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Models Folder
MODELS_DIR = BASE_DIR / "models"

# Dataset Path
DATASET_PATH = BASE_DIR / "placementdata.csv"

# Create models folder if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)


def preprocess_data():

    df = pd.read_csv("../placementdata.csv")

    df.drop_duplicates(inplace=True)

    df.drop(columns=["StudentID"], inplace=True)

    numerical_columns = [
        "CGPA",
        "Internships",
        "Projects",
        "Workshops/Certifications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "SSC_Marks",
        "HSC_Marks",
        "CodingScore",
        "CommunicationScore"
    ]

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    categorical_columns = [
        "ExtracurricularActivities",
        "PlacementTraining",
        "PlacementStatus"
    ]

    # Fill missing categorical values with mode
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    encoders = {}

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    # Save all encoders
    joblib.dump(encoders, "../models/encoders.pkl")

  
    X = df.drop(columns=["PlacementStatus"])

    y = df["PlacementStatus"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, "../models/scaler.pkl")

 
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = preprocess_data()

    print("=" * 50)
    print("Preprocessing Completed Successfully")
    print("=" * 50)
    print("Training Data Shape :", X_train.shape)
    print("Testing Data Shape  :", X_test.shape)
    print("=" * 50)