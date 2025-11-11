
import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("xgboost_classifier_model.pkl")  #  saved XGBoost classification model
    regressor = joblib.load("xgboost_regressor_model.pkl")    #  saved XGBoost regression model
    return classifier, regressor

classifier, regressor = load_models()

# -----------------------------
# App UI
# -----------------------------
st.title("Financial Risk Assessment Platform")
st.write("Enter your data to get classification and regression predictions.")

# Example user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=1000, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)

# Prepare input for models
input_df = pd.DataFrame([[age, income, loan_amount]], columns=["age", "income", "loan_amount"])

# -----------------------------
# Predictions
# -----------------------------
if st.button("Predict"):
    # Classification prediction
    class_pred = classifier.predict(input_df)[0]

    # Regression prediction
    reg_pred = regressor.predict(input_df)[0]

    st.subheader("Predictions")
    st.write(f"Classification Risk Level: **{class_pred}**")
    st.write(f"Predicted Numeric Value (Regression): **{reg_pred:.2f}**")

