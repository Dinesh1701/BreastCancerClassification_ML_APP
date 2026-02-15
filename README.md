a. Problem Statement

    The objective of this project is to build and compare multiple Machine Learning classification models to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous).
    
    The goal is to evaluate and compare the performance of different ML algorithms using multiple evaluation metrics and deploy the best-performing model using a Streamlit web application.

b. Dataset Description 

    The dataset used in this project is the Breast Cancer Wisconsin Diagnostic Dataset, obtained from Kaggle (UCI repository).
    
    Dataset Details:
    
    Total Instances: 569
    
    Total Features: 30 numerical features
    
    Target Variable: diagnosis
    
    M → Malignant (1)
    
    B → Benign (0)
    
    Data Type: Numerical
    
    Problem Type: Binary Classification
    
    Preprocessing Steps:
    
    Removed unnecessary id column
    
    Removed empty columns (if any)
    
    Converted categorical target (M/B) into numeric values (1/0)
    
    Checked and handled missing values
    
    Applied Standard Scaling before model training

c. Models Used and Evaluation Metrics 

    The following six classification models were implemented on the same dataset:
    
    •	Logistic Regression
    
    •	Decision Tree Classifier
    
    •	K-Nearest Neighbor (kNN)
    
    •	Naive Bayes (Gaussian)
    
    •	Random Forest (Ensemble)
    
    •	XGBoost (Ensemble)
    
    •	Evaluation Metrics Used:
    
    •	Accuracy
    
    •	AUC Score
    
    •	Precision
    
    •	Recall
    
    •	F1 Score
    
    •	Matthews Correlation Coefficient (MCC)
    | ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
    | ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
    | Logistic Regression | 0.9737   | 0.9974 | 0.9762    | 0.9535 | 0.9647 | 0.9439 |
    | Decision Tree       | 0.9298   | 0.9253 | 0.9070    | 0.9070 | 0.9070 | 0.8506 |
    | kNN                 | 0.9474   | 0.9820 | 0.9302    | 0.9302 | 0.9302 | 0.8880 |
    | Naive Bayes         | 0.9649   | 0.9974 | 0.9756    | 0.9302 | 0.9524 | 0.9253 |
    | Random Forest       | 0.9649   | 0.9956 | 0.9756    | 0.9302 | 0.9524 | 0.9253 |
    | XGBoost             | 0.9561   | 0.9908 | 0.9524    | 0.9302 | 0.9412 | 0.9064 |



d. Observations on Model Performance 
    •	ML Model Name	Observation about Model Performance
    •	Logistic Regression	Achieved the highest accuracy and AUC score. The dataset appears nearly linearly separable, making logistic regression highly effective.
    •	Decision Tree	Lower performance compared to other models. Likely slight overfitting due to tree-based structure.
    •	kNN	Performed well but slightly lower than logistic regression. Sensitive to feature scaling.
    •	Naive Bayes	Strong performance despite independence assumption. Worked well on numerical dataset.
    •	Random Forest	Stable and robust performance due to ensemble averaging. Reduced overfitting compared to Decision Tree.
    •	XGBoost	Strong boosting performance, but slightly lower than Logistic Regression for this dataset.


Streamlit Application Features

    •	The application was deployed using Streamlit Community Cloud.
    
    •	The Streamlit App Includes:
    
    •	Dataset Upload Option (CSV format)
    
    •	Model Selection Dropdown
    
    •	Display of Evaluation Metrics
    
    •	Confusion Matrix Output
    
    •	Interactive Web Interface

📁 Project Structure
    project-folder/
    │-- app.py
    │-- train_models.py
    │-- requirements.txt
    │-- README.md
    │-- model/
        │-- Logistic Regression.pkl
        │-- Decision Tree.pkl
        │-- kNN.pkl
        │-- Naive Bayes.pkl
        │-- Random Forest.pkl
        │-- XGBoost.pkl
        │-- scaler.pkl

 Technologies Used

    •	Python
    
    •	Scikit-learn
    
    •	XGBoost
    
    •	Pandas
    
    •	NumPy
    
    •	Streamlit
    
    •	Joblib

 Deployment

    The application is deployed using Streamlit Community Cloud and is accessible via the live link provided in the submission.
