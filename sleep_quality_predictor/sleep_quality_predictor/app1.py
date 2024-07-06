import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model and other necessary objects
classifier = joblib.load(r'C:\Users\wailb\Desktop\FINAL PROJECT\app\app_SQ\sleep_quality_predictor\classifier.pkl')
label_encoders = joblib.load(r'C:\Users\wailb\Desktop\FINAL PROJECT\app\app_SQ\sleep_quality_predictor\label_encoders.pkl')
scaler = joblib.load(r'C:\Users\wailb\Desktop\FINAL PROJECT\app\app_SQ\sleep_quality_predictor\scaler.pkl')

# Function to get the closest valid choice for numeric inputs
def get_closest_valid_choice(user_input, valid_choices):
    closest_choice = min(valid_choices, key=lambda x: abs(float(x) - float(user_input)))
    return closest_choice

# Function to get valid input with Streamlit widgets
def get_valid_input(prompt, valid_choices, is_numeric=False):
    if is_numeric:
        user_input = st.selectbox(prompt, valid_choices)
        return user_input
    else:
        user_input = st.selectbox(prompt, valid_choices)
        return user_input

def main():
    st.title("Sleep Quality Predictor")

    # Valid choices for each input
    valid_ages = [i for i in range(27, 60)]
    valid_genders = ['Male', 'Female']
    valid_occupations = ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager']
    valid_sleep_durations = [6.1, 6.2, 5.9, 6.3, 7.8, 6.0, 6.5, 7.6, 7.7, 7.9, 6.4, 7.5, 7.2, 5.8, 6.7, 7.3, 7.4, 7.1, 6.6, 6.9, 8.0, 6.8, 8.1, 8.3, 8.5, 8.4, 8.2]
    valid_physical_activities = [i for i in range(0, 101)]
    valid_stress_levels = [i for i in range(1, 11)]
    valid_bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
    valid_heart_rates = [i for i in range(50, 101)]
    valid_daily_steps = [4200, 10000, 3000, 3500, 8000, 4000, 4100, 6800, 5000, 7000, 5500, 5200, 5600, 3300, 4800, 7500, 7300, 6200, 6000, 3700]
    valid_sleep_disorders = ['None', 'Sleep Apnea', 'Insomnia']

    # Collect user inputs with validation
    age = get_valid_input("How old are you?", valid_ages, is_numeric=True)
    gender = get_valid_input("What's your gender?", valid_genders)
    occupation = get_valid_input("What's your occupation?", valid_occupations)
    sleep_duration = get_valid_input("How many hours do you sleep per day?", valid_sleep_durations, is_numeric=True)
    physical_activity = get_valid_input("What's your physical activity level? (0-100)", valid_physical_activities, is_numeric=True)
    stress_level = get_valid_input("What's your stress level? (1-10)", valid_stress_levels, is_numeric=True)
    bmi_category = get_valid_input("What's your BMI category?", valid_bmi_categories)
    heart_rate = get_valid_input("What's your average heart rate?", valid_heart_rates, is_numeric=True)
    daily_steps = get_valid_input("How many steps do you take daily?", valid_daily_steps, is_numeric=True)
    sleep_disorder = get_valid_input("Do you have any sleep disorder?", valid_sleep_disorders)

    if st.button("Predict Sleep Quality"):
        # Encode categorical variables
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        occupation_encoded = label_encoders['Occupation'].transform([occupation])[0]
        bmi_category_encoded = label_encoders['BMI Category'].transform([bmi_category])[0]
        sleep_disorder_encoded = label_encoders['Sleep Disorder'].transform([sleep_disorder])[0]

        # Combine all features
        input_data = [age, sleep_duration, physical_activity, stress_level, heart_rate, daily_steps, gender_encoded, occupation_encoded, bmi_category_encoded, sleep_disorder_encoded]
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)
        
        # Predict sleep quality using the trained classifier
        predicted_quality = classifier.predict(input_data)[0]
        
        st.success(f'Predicted Quality of Sleep: {predicted_quality}')

if __name__ == '__main__':
    main()
