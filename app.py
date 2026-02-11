import streamlit as st
import pandas as pd
import pickle

st.title("Student Health Risk Prediction")

uploaded_file = st.file_uploader("Upload student health data CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Model selection dropdown
    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )
    
    # Load model using pickle
    model_filename = f"model/{model_choice.replace(' ', '_').lower()}.pkl"
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    
    # Drop ID and empty columns
    data = data.drop(columns=["Student_ID"], errors="ignore")
    data = data.dropna(axis=1, how="all")
    
    # Separate features and target
    if "Health_Risk_Level" in data.columns:
        X = data.drop("Health_Risk_Level", axis=1)
        y = data["Health_Risk_Level"]
    else:
        X = data
    
    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    
    # Make predictions
    y_pred = model.predict(X)
    
    st.write("Predictions (first 10):", y_pred[:10])