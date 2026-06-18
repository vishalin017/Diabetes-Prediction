import joblib
import shap

# Load trained pipeline
model = joblib.load(
    "models/diabetes_model.pkl"
)

# Extract Random Forest
rf_model = model.named_steps["classifier"]

# Create SHAP explainer
explainer = shap.TreeExplainer(
    rf_model
)

# Save explainer
joblib.dump(
    explainer,
    "explainability/shap_explainer.pkl"
)

print("SHAP Explainer Saved")