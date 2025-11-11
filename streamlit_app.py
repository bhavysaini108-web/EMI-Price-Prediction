import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("xgboost_classifier_model.pkl")
    classifier_features = joblib.load("classifier_features.pkl")
    regressor = joblib.load("xgboost_regressor_model.pkl")
    regressor_features = joblib.load("regressor_features.pkl")
    return classifier, regressor, classifier_features, regressor_features

classifier, regressor, classifier_features, regressor_features = load_models()

# -----------------------------
# App UI
# -----------------------------
st.title("Financial Risk Assessment Platform")
st.write("Enter your data to get classification and regression predictions.")

# Example of more complete user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
monthly_salary = st.number_input("Monthly Salary", min_value=0, value=50000)
requested_amount = st.number_input("Requested Loan Amount", min_value=0, value=10000)
requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, value=12)
# ... add all other required features here

# Combine into dict
user_input = {
    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "monthly_salary": monthly_salary,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
    # add remaining features with default values if not collected from user
}

# Prepare input DataFrame
input_df = pd.DataFrame([user_input])
input_class = input_df.reindex(columns=classifier_features, fill_value=0)
input_reg = input_df.reindex(columns=regressor_features, fill_value=0)

# -----------------------------
# Predictions
# -----------------------------
if st.button("Predict"):
    class_pred = classifier.predict(input_class)[0]
    reg_pred = regressor.predict(input_reg)[0]

    st.subheader("Predictions")
    st.write(f"Classification Risk Level: **{class_pred}**")
    st.write(f"Predicted Numeric Value (Regression): **{reg_pred:.2f}**")



