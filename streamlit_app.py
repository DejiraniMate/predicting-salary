
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_regressor_model.pkl')

# Load the label encoders
encoders = {}
for col in ['Gender', 'Education', 'Job_Title', 'Location']:
    encoders[col] = joblib.load(f'{col}_label_encoder.pkl')

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for features
education = st.selectbox('Education', encoders['Education'].classes_)
experience = st.slider('Experience (Years)', 0, 30, 5)
location = st.selectbox('Location', encoders['Location'].classes_)
job_title = st.selectbox('Job Title', encoders['Job_Title'].classes_)
age = st.slider('Age', 18, 70, 30)
gender = st.selectbox('Gender', encoders['Gender'].classes_)

if st.button('Predict Salary'):
    # Create a DataFrame from the input
    input_data = pd.DataFrame({
        'Education': [education],
        'Experience': [experience],
        'Location': [location],
        'Job_Title': [job_title],
        'Age': [age],
        'Gender': [gender]
    })

    # Apply label encoding to the input data
    for col in ['Gender', 'Education', 'Job_Title', 'Location']:
        input_data[col] = encoders[col].transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: ${prediction:,.2f}')
