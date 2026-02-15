import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.title("Breast Cancer Classification App")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "kNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(f"model/{model_choice}.pkl")

    X = scaler.transform(X)
    y_pred = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))
