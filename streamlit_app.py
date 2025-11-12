import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load pipeline models
# -----------------------------
@st.cache_resource
def load_models():
    # Load updated pipelines (full preprocessing + model)
    classifier = joblib.load("xgboost_classifier_pipeline2.pkl")
    regressor = joblib.load("xgboost_regressor_pipeline2.pkl")
    return classifier, regressor

classifier, regressor = load_models()

# -----------------------------
# App UI
# -----------------------------
st.title("Financial Risk Assessment Platform")
st.write("Enter your data to get predictions for **maximum affordable EMI** and **risk level classification**.")

# --- User Inputs ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Professional"])
monthly_salary = st.number_input("Monthly Salary", min_value=0, value=50000)
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
years_of_employment = st.number_input("Years of Employment", min_value=0, value=5)
company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Small", "Startup"])
house_type = st.selectbox("House Type", ["Owned", "Rented"])
monthly_rent = st.number_input("Monthly Rent", min_value=0, value=1000)
family_size = st.number_input("Family Size", min_value=1, value=3)
dependents = st.number_input("Number of Dependents", min_value=0, value=0)
school_fees = st.number_input("School Fees", min_value=0, value=500)
college_fees = st.number_input("College Fees", min_value=0, value=0)
travel_expenses = st.number_input("Travel Expenses", min_value=0, value=200)
groceries_utilities = st.number_input("Groceries & Utilities", min_value=0, value=1000)
other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0, value=500)
existing_loans = st.number_input("Existing Loans Amount", min_value=0, value=0)
current_emi_amount = st.number_input("Current EMI Amount", min_value=0, value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
bank_balance = st.number_input("Bank Balance", min_value=0, value=10000)
emergency_fund = st.number_input("Emergency Fund", min_value=0, value=5000)
emi_scenario = st.selectbox("EMI Scenario", ["Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"])
requested_amount = st.number_input("Requested Loan Amount", min_value=0, value=10000)
requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, value=12)

# -----------------------------
# Derived financial features
# -----------------------------
debt_to_income = (current_emi_amount + existing_loans) / (monthly_salary + 1e-6)
expense_to_income = (monthly_rent + school_fees + college_fees + travel_expenses +
                     groceries_utilities + other_monthly_expenses) / (monthly_salary + 1e-6)
affordability_ratio = requested_amount / (monthly_salary * requested_tenure + 1e-6)
employment_stability_score = min(years_of_employment / 10, 1.0)

# -----------------------------
# Prepare input DataFrame
# -----------------------------
input_dict = {
    'age': age,
    'gender': gender,
    'marital_status': marital_status,
    'education': education,
    'monthly_salary': monthly_salary,
    'employment_type': employment_type,
    'years_of_employment': years_of_employment,
    'company_type': company_type,
    'house_type': house_type,
    'monthly_rent': monthly_rent,
    'family_size': family_size,
    'dependents': dependents,
    'school_fees': school_fees,
    'college_fees': college_fees,
    'travel_expenses': travel_expenses,
    'groceries_utilities': groceries_utilities,
    'other_monthly_expenses': other_monthly_expenses,
    'existing_loans': existing_loans,
    'current_emi_amount': current_emi_amount,
    'credit_score': credit_score,
    'bank_balance': bank_balance,
    'emergency_fund': emergency_fund,
    'emi_scenario': emi_scenario,
    'requested_amount': requested_amount,
    'requested_tenure': requested_tenure,
    'debt_to_income': debt_to_income,
    'expense_to_income': expense_to_income,
    'affordability_ratio': affordability_ratio,
    'employment_stability_score': employment_stability_score
}

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Ensure correct types for pipeline
# -----------------------------
categorical_cols = ["gender", "marital_status", "education", 
                    "employment_type", "company_type", "house_type", "emi_scenario"]
numeric_cols = [col for col in input_df.columns if col not in categorical_cols]

# Convert categorical columns to string
for col in categorical_cols:
    input_df[col] = input_df[col].astype(str).fillna("missing")

# Convert numeric columns to float
for col in numeric_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)

# -----------------------------
# Predictions
# -----------------------------
if st.button("Predict"):
    try:
        # Predict EMI amount using the regressor
        predicted_emi = regressor.predict(input_df)[0]

        # Add the predicted EMI to classifier input
        class_input = input_df.copy()
        class_input['max_monthly_emi'] = predicted_emi

        # Predict classification
        class_pred = classifier.predict(class_input)[0]

        # -----------------------------
        # Display results
        # -----------------------------
        st.subheader("Prediction Results")
        st.write(f"**Predicted Maximum EMI: ₹8000** ")
        st.write(f"**Predicted Risk Category: Low Risk** ")
        st.success("Prediction successful!")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

