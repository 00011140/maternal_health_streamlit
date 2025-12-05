import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model_utils import load_data

st.title("Exploratory Data Analysis")

df = load_data()

st.write("### Dataset Preview")
st.write(df.head())

st.write("### Summary Statistics")
st.write(df.describe())

st.write("### Distribution of Target Variable")
fig, ax = plt.subplots()
sns.countplot(data=df, x="RiskLevel", ax=ax)
st.pyplot(fig)

st.write("### Correlation Heatmap")
numeric_df = df.select_dtypes(include='number')

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)