import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
pipeline = joblib.load("depression_model_pipeline.pkl")  # Ensure this matches your saved pipeline filename

# Title and instructions
st.title("Student Depression Predictor")
st.markdown("Enter the details below to predict the risk of depression:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100)
study_hours = st.slider("Study Hours per Day", 0, 15, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0, 15, 7)
internet_access = st.selectbox("Internet Access", ["Yes", "No"])

# Map inputs to the model's expected format
# If your model expects numerical encoding, do the encoding here
input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Study Hours": [study_hours],
    "Sleep Hours": [sleep_hours],
    "Internet Access": [internet_access]
})

# Optional: encode categorical variables if your pipeline doesn't handle it
# For example, if your pipeline includes OneHotEncoder or similar, skip this step

# Prediction
if st.button("Predict"):
    try:
        # Make prediction
        prediction = pipeline.predict(input_data)
        probabilities = pipeline.predict_proba(input_data)

        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.write("**Depression Risk:** Yes")
        else:
            st.write("**Depression Risk:** No")
        st.write(f"**Probability of No Depression:** {probabilities[0][0]:.2f}")
        st.write(f"**Probability of Depression:** {probabilities[0][1]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
