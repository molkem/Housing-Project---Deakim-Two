import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title of the application
st.title('Paris Real Estate Price Prediction')

st.write("""
This application predicts real estate prices in Paris based on various property features.
""")

# Load the trained model and column names (replace with actual paths if saved)
try:
    # Make sure 'ridge_model.pkl' and 'model_columns.pkl' are in the same directory as app.py
    model = joblib.load('ridge_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'ridge_model.pkl' and 'model_columns.pkl' are in the same directory as app.py")
    st.stop()


# Create input fields for the features
st.sidebar.header('Property Features')

def user_input_features():
    squareMeters = st.sidebar.slider('Square Meters', 10, 100000, 50000)
    numberOfRooms = st.sidebar.slider('Number of Rooms', 1, 100, 5)
    hasYard = st.sidebar.selectbox('Has Yard', (0, 1))
    hasPool = st.sidebar.selectbox('Has Pool', (0, 1))
    floors = st.sidebar.slider('Floors', 1, 100, 10)
    cityCode = st.sidebar.slider('City Code', 1000, 100000, 75000)
    cityPartRange = st.sidebar.slider('City Part Range', 1, 10, 5)
    numPrevOwners = st.sidebar.slider('Number of Previous Owners', 0, 10, 1)
    made = st.sidebar.slider('Year Built', 1900, 2024, 2000)
    isNewBuilt = st.sidebar.selectbox('Is New Built', (0, 1))
    hasStormProtector = st.sidebar.selectbox('Has Storm Protector', (0, 1))
    basement = st.sidebar.slider('Basement Size (sqm)', 0, 10000, 100)
    attic = st.sidebar.slider('Attic Size (sqm)', 0, 10000, 100)
    garage = st.sidebar.slider('Garage Size (sqm)', 0, 1000, 50)
    hasStorageRoom = st.sidebar.selectbox('Has Storage Room', (0, 1))
    hasGuestRoom = st.sidebar.slider('Number of Guest Rooms', 0, 10, 1)

    # Create the engineered features required by the model
    # Ensure these match the features used during training, excluding the target 'price'
    data = {'squareMeters': squareMeters,
            'numberOfRooms': numberOfRooms,
            'hasYard': hasYard,
            'hasPool': hasPool,
            'floors': floors,
            'cityCode': cityCode,
            'cityPartRange': cityPartRange,
            'numPrevOwners': numPrevOwners,
            'made': made,
            'isNewBuilt': isNewBuilt,
            'hasStormProtector': hasStormProtector,
            'basement': basement,
            'attic': attic,
            'garage': garage,
            'hasStorageRoom': hasStorageRoom,
            'hasGuestRoom': hasGuestRoom,
            # These engineered features were in the training data
            'price_per_square_meter': 0, # This was engineered during training, not a user input for prediction
            'rooms_per_square_meter': numberOfRooms / squareMeters if squareMeters > 0 else 0,
            'floors_per_square_meter': floors / squareMeters if squareMeters > 0 else 0
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure the input features are in the same order and have the same names as the training data
# Use the loaded model_columns to ensure consistency
try:
    input_df = input_df[model_columns]
except KeyError as e:
    st.error(f"Error: Input features do not match model features. Missing feature: {e}")
    st.stop()


st.subheader('User Input Features')
st.write(input_df)

# Make prediction
if st.button('Predict Price'):
    try:
        prediction = model.predict(input_df)
        st.subheader('Predicted Property Price')
        st.write(f"€{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("Note: 'price_per_square_meter' was an engineered feature during training and is not a user input for prediction. Its value for prediction input is arbitrary (set to 0 here) to match the feature structure the model expects.")