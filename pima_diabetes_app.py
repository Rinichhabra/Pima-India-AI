import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the Naive Bayes model
model = pickle.load(open(r'C:\Users\rinis\OneDrive\Desktop\naive_bayes_model.pkl', 'rb'))


# App title
st.title('Diabetes Prediction Web App')
st.write("This app predicts whether a person is likely to have diabetes based on diagnostic measurements.")

# Input fields with max values
st.header('Enter Patient Details:')
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=140, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=79)
bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=0, max_value=90, value=33)

# Predict button
if st.button('Predict'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Output
    if prediction[0] == 1:
        st.error('The person is likely to have diabetes.')
    else:
        st.success('The person is unlikely to have diabetes.')

    # Show probability
    st.subheader('Prediction Confidence:')
    st.write(f"🔹 Probability of NOT having diabetes: **{probability[0][0]:.2f}**")
    st.write(f"🔹 Probability of HAVING diabetes: **{probability[0][1]:.2f}**")
