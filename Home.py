import streamlit as st

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Preprocessing", "Models"])

# Navigation logic
if page == "Home":
    st.title("Maternal Health Risk Prediction App")
    st.write("This Streamlit application demonstrates the workflow of:")
    st.write("""
    - Dataset loading  
    - Exploratory Data Analysis (EDA)  
    - Data preprocessing  
    - Model training and evaluation  
    """)
elif page == "EDA":
    import EDA
elif page == "Preprocessing":
    import Preprocessing
elif page == "Models":
    import Models
