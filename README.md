# 💳 Bank Customer Churn Prediction App

A beautiful, interactive Streamlit web app to predict whether a customer will churn based on their profile. This application uses a trained **TensorFlow model** along with **scikit-learn encoders and scalers** to make real-time predictions.

---

## 🚀 Features

- Predicts the **likelihood of customer churn** for a bank.
- Interactive UI built with **Streamlit**.
- Accepts customer details like age, geography, balance, salary, etc.
- Uses a trained **Deep Learning model (TensorFlow)** and encodes categorical variables using `LabelEncoder` and `OneHotEncoder`.
- Clean aesthetic with emoji-enhanced layout, progress bar, and intuitive forms.

---

## 📊 Model & Data Preprocessing

- The model (`model.h5`) was trained on a cleaned customer churn dataset.
- Categorical variables:
  - `Gender`: Label encoded
  - `Geography`: One-hot encoded
- Numerical features were scaled using `StandardScaler`.

---

## 📁 File Structure

📦 customer-churn-app   
├── app.py # Streamlit application code  
├── model.h5 # Trained Keras model  
├── scaler.pkl # StandardScaler used for feature scaling  
├── label_encoder_gender.pkl # LabelEncoder for gender  
├── onehot_encoder_geo.pkl # OneHotEncoder for geography  
└── README.md # Project overview and instructions  

## 🖥️ Run the App Locally

1. **Clone the repository**
```bash
git clone https://github.com/your-username/customer-churn-app.git
cd customer-churn-app
```
2. **Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # on Windows use venv\Scripts\activate
pip install -r requirements.txt
```
3. **Run the Streamlit app**
```bash
stteamlit run app.py
```

