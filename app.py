from flask import Flask, render_template, request, jsonify

import pandas as pd
import numpy as np
import joblib
import sqlite3
import logging
import shap

app = Flask(__name__)

# ==========================================
# Load Model & SHAP Explainer
# ==========================================

model = joblib.load(
    "models/diabetes_model.pkl"
)

explainer = joblib.load(
    "explainability/shap_explainer.pkl"
)

# ==========================================
# Logging Configuration
# ==========================================

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================
# Database Function
# ==========================================

def save_prediction(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c,
    glucose,
    probability,
    prediction
):

    conn = sqlite3.connect(
        "database/diabetes.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions(
            gender,
            age,
            hypertension,
            heart_disease,
            smoking_history,
            bmi,
            hba1c,
            glucose,
            probability,
            prediction
        )
        VALUES(
            ?,?,?,?,?,?,?,?,?,?
        )
        """,
        (
            gender,
            age,
            hypertension,
            heart_disease,
            smoking_history,
            bmi,
            hba1c,
            glucose,
            probability,
            prediction
        )
    )

    conn.commit()
    conn.close()


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Prediction Route
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        gender = request.form["gender"]
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        smoking_history = request.form["smoking_history"]
        bmi = float(request.form["bmi"])
        hba1c = float(request.form["hba1c"])
        glucose = float(request.form["glucose"])

        data = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [glucose]
        })

        # ==========================
        # Prediction
        # ==========================

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0][1]

        # ==========================
        # SHAP Explainability
        # ==========================

        processed_data = (
            model.named_steps["preprocessor"]
            .transform(data)
        )

        feature_names = (
            model.named_steps["preprocessor"]
            .get_feature_names_out()
        )

        shap_values = explainer.shap_values(
            processed_data
        )

        if isinstance(shap_values, list):

            importance = np.abs(
                shap_values[1][0]
            )

        else:

            importance = np.abs(
                shap_values
            )

            if len(importance.shape) == 3:

                importance = importance[0, :, 1]

            elif len(importance.shape) == 2:

                importance = importance[0]

        top_indices = (
            importance.argsort()[-5:][::-1]
        )

        top_features = []

        for idx in top_indices:

            feature_name = (
                feature_names[idx]
                .replace("num__", "")
                .replace("cat__", "")
            )

            top_features.append({
                "feature": feature_name,
                "impact": round(
                    float(importance[idx]),
                    4
                )
            })

        # ==========================
        # Risk Result
        # ==========================

        if prediction == 1:
            result = "High Diabetes Risk"
        else:
            result = "Low Diabetes Risk"

        # ==========================
        # Save Prediction
        # ==========================

        save_prediction(
            gender,
            age,
            hypertension,
            heart_disease,
            smoking_history,
            bmi,
            hba1c,
            glucose,
            float(probability),
            result
        )

        # ==========================
        # Logging
        # ==========================

        logging.info(
            f"{gender} | {age} | {result} | {probability}"
        )

        return render_template(
            "index.html",
            prediction=result,
            probability=round(
                probability * 100,
                2
            ),
            top_features=top_features
        )

    except Exception as e:

        logging.error(str(e))

        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            probability=0,
            top_features=[]
        )


# ==========================================
# History Page
# ==========================================

@app.route("/history")
def history():

    conn = sqlite3.connect(
        "database/diabetes.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        rows=rows
    )


# ==========================================
# API Endpoint
# ==========================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    try:

        data = request.get_json()

        sample = pd.DataFrame({
            "gender": [data["gender"]],
            "age": [data["age"]],
            "hypertension": [data["hypertension"]],
            "heart_disease": [data["heart_disease"]],
            "smoking_history": [data["smoking_history"]],
            "bmi": [data["bmi"]],
            "HbA1c_level": [data["hba1c"]],
            "blood_glucose_level": [data["glucose"]]
        })

        prediction = model.predict(
            sample
        )[0]

        probability = model.predict_proba(
            sample
        )[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(
                probability * 100,
                2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


# ==========================================
# Run App
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )