import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the model and preprocessors
model = tf.keras.models.load_model('churn_model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered", page_icon="💳")

# Title
st.markdown("<h1 style='text-align: center;'>💳 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether a customer is likely to leave your bank based on their profile.</p>", unsafe_allow_html=True)
st.write("---")

# Input form
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 35)
        balance = st.number_input('💰 Balance', value=50000.0, step=1000.0)
        credit_score = st.number_input('📊 Credit Score', value=650.0, step=10.0)

    with col2:
        estimated_salary = st.number_input('🧾 Estimated Salary', value=50000.0, step=1000.0)
        tenure = st.slider('⏳ Tenure (years)', 0, 10, 3)
        num_of_products = st.selectbox('📦 Number of Products', [1, 2, 3, 4])
        has_cr_card = st.selectbox('💳 Has Credit Card?', ['Yes', 'No'])
        is_active_member = st.selectbox('🟢 Active Member?', ['Yes', 'No'])

    submit = st.form_submit_button("🔍 Predict Churn")

# Prediction logic
if submit:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale features
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    probability = float(prediction[0][0])

    st.write("---")
    st.subheader("📈 Churn Probability")
    st.progress(int(probability * 100))
    st.write(f"**Probability:** `{probability:.2%}`")

    if probability > 0.5:
        st.error("❌ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **likely to stay**.")

    st.caption("This prediction is based on historical data and model inference.")
