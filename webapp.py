import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load all necessary files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.write("Fill in the customer details to predict if they are likely to churn.")

# Input form
def user_input_form():
    st.subheader("Customer Details")
    gender = st.selectbox("Gender", ['Female', 'Male'])
    senior = st.selectbox("Senior Citizen", ['0', '1'])  # encoded as string
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    phoneservice = st.selectbox("Phone Service", ['Yes', 'No'])
    multiplelines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    internetservice = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    onlinesecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    onlinebackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    deviceprotection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    techsupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    streamingtv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    streamingmovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperlessbilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    paymentmethod = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    monthlycharges = st.number_input("Monthly Charges", min_value=0.0)
    totalcharges = st.number_input("Total Charges", min_value=0.0)

    return {
        'gender': gender,
        'senior_citizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phoneservice,
        'MultipleLines': multiplelines,
        'InternetService': internetservice,
        'OnlineSecurity': onlinesecurity,
        'OnlineBackup': onlinebackup,
        'DeviceProtection': deviceprotection,
        'TechSupport': techsupport,
        'StreamingTV': streamingtv,
        'StreamingMovies': streamingmovies,
        'Contract': contract,
        'PaperlessBilling': paperlessbilling,
        'PaymentMethod': paymentmethod,
        'MonthlyCharges': monthlycharges,
        'TotalCharges': totalcharges
    }

input_dict = user_input_form()

# Preprocessing
if st.button("🔍 Predict Churn"):
    try:
        # Encode categorical variables using stored LabelEncoders
        for col, le in label_encoders.items():
            if col in input_dict:
                try:
                    input_dict[col] = le.transform([input_dict[col]])[0]
                except:
                    input_dict[col] = le.transform([le.classes_[0]])[0]

        # Build DataFrame and reorder columns
        input_df = pd.DataFrame([[input_dict.get(col, 0) for col in feature_names]], columns=feature_names)

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn (Risk: {prediction_proba:.2%})")
        else:
            st.success(f"✅ The customer is likely to stay (Risk: {prediction_proba:.2%})")

    except Exception as e:
        st.error(f"❌ Error: {e}")
