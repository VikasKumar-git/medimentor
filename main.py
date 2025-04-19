import pickle
import json
import numpy as np

# Model paths
MODEL_PATHS = {
    "hypertension": "models/hypertension_predict_model",
    "stroke": "models/stroke_predict_model",
    "diabetes": "models/diabetes_predict_model",
    "obesity": "models/obesity_predict_model",
    "heart": "models/heart_predict_model"
}

# Mapping input keys to model-specific input arrays
MODEL_INPUT_KEYS = {
    "hypertension": ["gender", "age", "currentSmoker", "cigsPerDay", "diabetes", "sysBP", "diaBP", "BMI", "heartRate", "chol_category", "glucose_category"],
    "stroke": ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "BMI", "smoking_status"],
    "diabetes": ["Pregnancies", "BloodPressure", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "age", "glucose_category"],
    "obesity": ["sex", "age", "Height", "Weight"],
    "heart": ["age", "sex", "trestbps", "fbs", "thalach", "exang", "chol_category"]
}

# Categorical mappings
BINARY_MAP = {"yes": 1, "no": 0, "male": 1, "female": 0}
CATEGORY_MAP = {
    "chol_category": {"low_intake": 0, "moderate_intake": 1, "high_intake": 2},
    "glucose_category": {"low_intake": 0, "moderate_intake": 1, "high_intake": 2, "very_high_intake": 3},
    "smoking_status": {"never_smoked": 0, "formerly_smoked": 1, "smokes": 2},
    "work_type": {"child": 0, "never_worked": 1, "self_employed": 2, "govt_job": 3, "private": 4},
    "Residence_type": {"urban": 1, "rural": 0},
    "DiabetesPedigreeFunction": {"no_history": 0.0, "low_history": 0.25, "moderate_history": 0.5, "high_history": 0.75},
    "exang": BINARY_MAP
}

glucose_aliases = {
    "low": "low_intake",
    "moderate": "moderate_intake",
    "high": "high_intake",
    "very_high": "very_high_intake"
}

# Collect input from user
print("Please enter the following details:")
user_input = {
    "gender": input("Gender (male/female): ").lower(),
    "age": int(input("Age: ")),
    "Height": float(input("Height (m): ")),
    "Weight": float(input("Weight (kg): ")),
    "sysBP": float(input("Systolic BP: ")),
    "diaBP": float(input("Diastolic BP: ")),
    "heartRate": int(input("Heart Rate (bpm): ")),
    "cigsPerDay": int(input("Cigarettes Per Day: ")),
    "smoking_status": input("Smoking Status (never_smoked/formerly_smoked/smokes): ").lower(),
    "chol_category": input("Cholesterol Intake (low_intake/moderate_intake/high_intake): ").lower(),
    "glucose_category": input("Glucose Intake (low/moderate/high/very_high): ").lower(),
    "ever_married": input("Ever Married (yes/no): ").lower(),
    "Pregnancies": int(input("Number of Pregnancies: ")),
    "work_type": input("Work Type (child/never_worked/self_employed/govt_job/private): ").lower(),
    "Residence_type": input("Residence Type (urban/rural): ").lower(),
    "DiabetesPedigreeFunction": input("Family History of Diabetes (no_history/low_history/moderate_history/high_history): ").lower(),
    "exang": input("Chest Pain During Exercise (yes/no): ").lower(),
    # Derived
    "BMI": 0.0,
    "currentSmoker": 0,
    "diabetes": 0,
    "hypertension": 0,
    "heart_disease": 0,
    "sex": 0,
    "trestbps": 0,
    "fbs": 0,
    "thalach": 0,
    "BloodPressure": 0,
    "SkinThickness": 0
}

# Normalize glucose alias if needed
if user_input["glucose_category"] in glucose_aliases:
    user_input["glucose_category"] = glucose_aliases[user_input["glucose_category"]]

user_input["BMI"] = round(user_input["Weight"] / (user_input["Height"] ** 2), 2)
user_input["currentSmoker"] = 1 if user_input["smoking_status"] == "smokes" else 0
user_input["sex"] = BINARY_MAP.get(user_input["gender"], 0)

# Safety check to avoid KeyError
glucose_level_encoded = CATEGORY_MAP["glucose_category"].get(user_input["glucose_category"], 0)
user_input["fbs"] = 1 if glucose_level_encoded > 2 else 0

user_input["trestbps"] = user_input["sysBP"]
user_input["thalach"] = user_input["heartRate"]

# Load models and predict
predictions = {}
for model_name, model_path in MODEL_PATHS.items():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    input_keys = MODEL_INPUT_KEYS[model_name]
    input_vector = []

    for key in input_keys:
        val = user_input.get(key)
        if val is None:
            val = 0
        elif key in CATEGORY_MAP:
            val = CATEGORY_MAP[key].get(val, 0)
        elif key in BINARY_MAP:
            val = BINARY_MAP.get(val, 0)
        elif isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                val = 0
        input_vector.append(float(val))

    # Get the probability of the positive class (1)
    proba = model.predict_proba([input_vector])[:, 1][0] if hasattr(model, 'predict_proba') else model.predict([input_vector])[0]
    
    # Apply smoothing: if proba is close to 0 or 1, we adjust it
    if proba == 1:
        proba = 0.95  # Avoid 100%
    elif proba == 0:
        proba = 0.05  # Avoid 0%
    
    # Round the probability to show it as a realistic percentage
    prediction_score = round(proba * 100, 2)
    
    # Add category information for obesity
    if model_name == "obesity":
        if user_input["BMI"] < 18.5:
            obesity_category = "Underweight"
        elif 18.5 <= user_input["BMI"] < 24.9:
            obesity_category = "Normal"
        elif 25 <= user_input["BMI"] < 29.9:
            obesity_category = "Overweight"
        else:
            obesity_category = "Obese"
        
        predictions[model_name + "_risk_percent"] = {
            "percentage": prediction_score,
            "category": obesity_category
        }
    else:
        # For other diseases, include whether it's the risk of having or not having the disease
        if proba > 50:
            belongingness = "Risk of having " + model_name
        else:
            belongingness = "Risk of not having " + model_name
        
        predictions[model_name + "_risk_percent"] = {
            "percentage": prediction_score,
            "belongingness": belongingness
        }

# Save results to JSON
with open("health_predictions.json", "w") as outfile:
    json.dump(predictions, outfile, indent=4)

print("\nPrediction results saved in health_predictions.json")
