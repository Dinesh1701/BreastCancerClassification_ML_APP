import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# -------------------------------
# Title
# -------------------------------
st.title("Breast Cancer Classification App")
st.write("Upload a CSV file to test the trained ML models.")

# -------------------------------
# Upload CSV File
# -------------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

# -------------------------------
# Model Selection
# -------------------------------
model_choice = st.selectbox(
    "Select a Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------------------
# If File Uploaded
# -------------------------------
if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    try:
        # Drop ID column
        if "id" in df.columns:
            df = df.drop("id", axis=1)

        # Remove empty columns if any
        df = df.dropna(axis=1, how="all")

        # Convert target column
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

        # Remove missing values if any
        df = df.dropna()

        # Separate features and target
        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"]

        # Load scaler and model
        scaler = joblib.load("model/scaler.pkl")
        model = joblib.load(f"model/{model_choice}.pkl")

        # Scale features
        X_scaled = scaler.transform(X)

        # Predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # -------------------------------
        # Evaluation Metrics
        # -------------------------------
        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.write("Accuracy:", round(accuracy, 4))
        st.write("AUC Score:", round(auc, 4))
        st.write("Precision:", round(precision, 4))
        st.write("Recall:", round(recall, 4))
        st.write("F1 Score:", round(f1, 4))
        st.write("MCC Score:", round(mcc, 4))

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        st.write(cm)

    except Exception as e:
        st.error("Error in processing file. Please upload correct dataset format.")
        st.write(e)
