import streamlit as st
import pandas as pd

st.title("Student Health Risk Prediction")

uploaded_file = st.file_uploader("student_health_data.csv", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])
    
    model = joblib.load(f"model/{model_choice.replace(' ', '_').lower()}.pkl")
    
    
    data = data.drop(columns=["Student_ID"], errors="ignore")
    data = data.dropna(axis=1, how="all")
    
    
    X = data.drop("Health_Risk_Level", axis=1)
    y = data["Health_Risk_Level"]
    
    
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    
    y_pred = model.predict(X)
    
    st.write("Predictions:", y_pred[:10])