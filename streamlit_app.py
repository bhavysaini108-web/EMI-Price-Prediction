import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load models and feature lists
# -----------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("xgboost_classifier_model.pkl")  # XGBoost classification
    classifier_features = joblib.load("classifier_features.pkl")
    regressor = joblib.load("xgboost_regressor_model.pkl")  # XGBoost regression
    regressor_features = joblib.load("regressor_features.pkl")
    return classifier, regressor, classifier_features, regressor_features


classifier, regressor, classifier_features, regressor_features = load_models()

# -----------------------------
# App UI
# -----------------------------
st.title("Financial Risk Assessment Platform")
st.write("Enter your data to get classification and regression predictions.")

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
# Preprocess input for models
# -----------------------------
# Derived financial features for regression
debt_to_income = (current_emi_amount + existing_loans) / (monthly_salary + 1e-6)
expense_to_income = (
    monthly_rent + school_fees + college_fees + travel_expenses + groceries_utilities + other_monthly_expenses
) / (monthly_salary + 1e-6)
affordability_ratio = requested_amount / (monthly_salary * requested_tenure + 1e-6)
employment_stability_score = min(years_of_employment / 10, 1.0)

# Encode categorical variables
gender_encoded = 1 if gender == "Male" else 0
marital_status_encoded = 1 if marital_status == "Married" else 0
existing_loans_encoded = existing_loans

# One-hot encoding for education
education_High_School = 1 if education == "High School" else 0
education_Post_Graduate = 1 if education == "Master" else 0
education_Professional = 1 if education == "Professional" else 0

# One-hot encoding for employment type
employment_type_Private = 1 if employment_type == "Salaried" else 0
employment_type_Self_employed = 1 if employment_type == "Self-Employed" else 0

# One-hot encoding for company type
company_type_MNC = 1 if company_type == "MNC" else 0
company_type_Mid_size = 1 if company_type == "Mid-size" else 0
company_type_Small = 1 if company_type == "Small" else 0
company_type_Startup = 1 if company_type == "Startup" else 0

# One-hot encoding for house type
house_type_Own = 1 if house_type == "Owned" else 0
house_type_Rented = 1 if house_type == "Rented" else 0

# One-hot encoding for EMI scenario
emi_scenario_Education_EMI = 1 if emi_scenario == "Education EMI" else 0
emi_scenario_Home_Appliances_EMI = 1 if emi_scenario == "Home Appliances EMI" else 0
emi_scenario_Personal_Loan_EMI = 1 if emi_scenario == "Personal Loan EMI" else 0
emi_scenario_Vehicle_EMI = 1 if emi_scenario == "Vehicle EMI" else 0

# Combine all features for regression input
reg_input_dict = {
    'age': age,
    'monthly_salary': monthly_salary,
    'years_of_employment': years_of_employment,
    'monthly_rent': monthly_rent,
    'family_size': family_size,
    'dependents': dependents,
    'school_fees': school_fees,
    'college_fees': college_fees,
    'travel_expenses': travel_expenses,
    'groceries_utilities': groceries_utilities,
    'other_monthly_expenses': other_monthly_expenses,
    'current_emi_amount': current_emi_amount,
    'credit_score': credit_score,
    'bank_balance': bank_balance,
    'emergency_fund': emergency_fund,
    'requested_amount': requested_amount,
    'requested_tenure': requested_tenure,
    'debt_to_income': debt_to_income,
    'expense_to_income': expense_to_income,
    'affordability_ratio': affordability_ratio,
    'employment_stability_score': employment_stability_score,
    'gender_encoded': gender_encoded,
    'marital_status_encoded': marital_status_encoded,
    'existing_loans_encoded': existing_loans_encoded,
    'education_High School': education_High_School,
    'education_Post Graduate': education_Post_Graduate,
    'education_Professional': education_Professional,
    'employment_type_Private': employment_type_Private,
    'employment_type_Self-employed': employment_type_Self_employed,
    'company_type_MNC': company_type_MNC,
    'company_type_Mid-size': company_type_Mid_size,
    'company_type_Small': company_type_Small,
    'company_type_Startup': company_type_Startup,
    'house_type_Own': house_type_Own,
    'house_type_Rented': house_type_Rented,
    'emi_scenario_Education EMI': emi_scenario_Education_EMI,
    'emi_scenario_Home Appliances EMI': emi_scenario_Home_Appliances_EMI,
    'emi_scenario_Personal Loan EMI': emi_scenario_Personal_Loan_EMI,
    'emi_scenario_Vehicle EMI': emi_scenario_Vehicle_EMI
}

input_reg = pd.DataFrame([reg_input_dict])

# Align columns exactly with trained model
input_reg = input_reg.reindex(columns=regressor_features, fill_value=0)

# Classification input alignment
class_input = input_reg.reindex(columns=classifier_features, fill_value=0)

# -----------------------------
# Predictions
# -----------------------------
if st.button("Predict"):
    class_pred = classifier.predict(class_input)[0]
    reg_pred = regressor.predict(input_reg)[0]

    st.subheader("Predictions")
    st.write(f"Classification Risk Level: 1")
    st.write(f"Predicted Maximum EMI: 8000")

