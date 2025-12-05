import streamlit as st
import pandas as pd
from model_utils import load_data, preprocess_data, train_models, evaluate_model

st.title("Model Training & Evaluation")

df = load_data()
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, le, scaler = preprocess_data(df)

log_reg, knn, rf = train_models(X_train_scaled, X_train, y_train)

model_choice = st.selectbox(
    "Select a model to evaluate:",
    ("Logistic Regression", "KNN", "Random Forest")
)

if model_choice == "Logistic Regression":
    acc, report = evaluate_model(log_reg, X_test_scaled, y_test)
elif model_choice == "KNN":
    acc, report = evaluate_model(knn, X_test_scaled, y_test)
else:
    acc, report = evaluate_model(rf, X_test, y_test)

st.write("### Accuracy:", round(acc, 3))

st.write("### Classification Report:")
st.write(pd.DataFrame(report).transpose())
