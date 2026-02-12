import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

st.title("Student Health Risk Prediction")

uploaded_file = st.file_uploader("Upload student health data CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    # Build filename (must match exactly your committed .pkl files)
    model_filename = f"{model_choice.replace(' ', '_').lower()}.pkl"

    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            model = pickle.load(f)

        # Preprocess
        data = data.drop(columns=["Student_ID"], errors="ignore")
        data = data.dropna(axis=1, how="all")

        if "Health_Risk_Level" in data.columns:
            X = data.drop("Health_Risk_Level", axis=1)
            y = data["Health_Risk_Level"]
        else:
            X = data
            y = None

        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(0)

        # Predictions
        y_pred = model.predict(X)

        # Map numeric predictions back to labels
        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        y_pred_labels = [label_map.get(val, val) for val in y_pred]

        st.write("Predictions (first 10):", y_pred_labels[:10])

        # If true labels exist, show evaluation metrics
        if y is not None:
            # Encode true labels
            y_true = pd.factorize(y)[0]

            # Accuracy, Precision, Recall, F1, MCC
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="macro")
            rec = recall_score(y_true, y_pred, average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")
            mcc = matthews_corrcoef(y_true, y_pred)

            # AUC (multi-class One-vs-Rest)
            y_true_bin = label_binarize(y_true, classes=[0,1,2])
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)
                auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
            else:
                auc = None

            st.subheader("Evaluation Metrics")
            st.write(f"Accuracy: {acc:.3f}")
            st.write(f"Precision: {prec:.3f}")
            st.write(f"Recall: {rec:.3f}")
            st.write(f"F1 Score: {f1:.3f}")
            st.write(f"MCC: {mcc:.3f}")
            if auc is not None:
                st.write(f"AUC: {auc:.3f}")
            else:
                st.write("AUC: Not available for this model")

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Low", "Moderate", "High"],
                        yticklabels=["Low", "Moderate", "High"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.subheader("Confusion Matrix")
            st.pyplot(fig)
    else:
        st.error(f"Model file not found: {model_filename}. Please check your repo.")