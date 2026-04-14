import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Delhi AQI Regional Impact Analysis")

st.write("""
This project analyzes regional contributions from Haryana and Punjab 
to Delhi Air Quality Index (AQI) levels using statistical and machine learning techniques.
""")

st.header("Project Highlights")

st.write("""
- Statistical analysis of regional AQI
- Lag based pollution analysis
- Feature ablation study
- Machine learning models
""")

st.header("Project Files")

st.write("See full analysis in the notebook below:")
st.write("Aqi_data_prediction.ipynb")

st.success("Project Developed by Asrar Farooq")
