import streamlit as st
from model_utils import load_data, preprocess_data

st.title("Data Preprocessing")

df = load_data()

st.write("### Checking Missing Values")
st.write(df.isna().sum())

st.write("### Encoding & Scaling")
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, le, scaler = preprocess_data(df)

st.write("Data preprocessing completed successfully!")
