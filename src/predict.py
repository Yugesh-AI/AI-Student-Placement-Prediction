import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "placementdata.csv"

# Create models folder if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)
# ==========================================
# Load Saved Files
# ==========================================

model = joblib.load(MODELS_DIR / "best_model.pkl")

scaler = joblib.load(MODELS_DIR / "scaler.pkl")

encoders = joblib.load(MODELS_DIR / "encoders.pkl")


# ==========================================
# Prediction Function
# ==========================================

def predict_student(student_data):

    """
    student_data must be a dictionary.

    Example:

    {
        "CGPA":8.2,
        "Internships":2,
        "Projects":3,
        "Workshops/Certifications":4,
        "AptitudeTestScore":78,
        "SoftSkillsRating":8,
        "ExtracurricularActivities":"Yes",
        "PlacementTraining":"Yes",
        "SSC_Marks":87,
        "HSC_Marks":82,
        "CodingScore":74,
        "CommunicationScore":79
    }

    """

    data = student_data.copy()

    # ======================================
    # Encode Categorical Columns
    # ======================================

    categorical_columns = [

        "ExtracurricularActivities",

        "PlacementTraining"

    ]

    for column in categorical_columns:

        data[column] = encoders[column].transform(

            [data[column]]

        )[0]

    # ======================================
    # Convert Dictionary to DataFrame
    # ======================================

    input_df = pd.DataFrame([data])

    # ======================================
    # Scale Input
    # ======================================

    input_scaled = scaler.transform(input_df)

    # ======================================
    # Prediction
    # ======================================

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    placement_probability = probability[1] * 100

    confidence = max(probability) * 100

    # ======================================
    # Decode Prediction
    # ======================================

    placement_encoder = encoders["PlacementStatus"]

    prediction_label = placement_encoder.inverse_transform(

        [prediction]

    )[0]

    # ======================================
    # Return Result
    # ======================================

    return {

        "Prediction": prediction_label,

        "PlacementProbability": round(

            placement_probability,

            2

        ),

        "ConfidenceScore": round(

            confidence,

            2

        )

    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    student = {

        "CGPA":8.1,

        "Internships":2,

        "Projects":3,

        "Workshops/Certifications":2,

        "AptitudeTestScore":75,

        "SoftSkillsRating":8,

        "ExtracurricularActivities":"Yes",

        "PlacementTraining":"Yes",

        "SSC_Marks":86,

        "HSC_Marks":84,

        "CodingScore":72,

        "CommunicationScore":78

    }

    result = predict_student(student)

    print("\nPrediction Result\n")

    print(result)