import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")

try:
    model = joblib.load(model_path)
    load_error = None
except Exception as exc:  # keep the app usable even if the model artifact is missing/corrupt
    model = None
    load_error = str(exc)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction")
st.write("Fill the customer details below to predict if they'll purchase a travel package")

if load_error:
    st.error(
        "Could not load the trained model "
        f"(`{os.path.basename(model_path)}`). Make sure the GitHub Actions "
        "pipeline has run at least once and committed the model file.\n\n"
        f"Details: {load_error}"
    )

# Collect user input
Age = st.slider("Age", 18, 70, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Number of Trips", 1, 22, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 3, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=22000.0, step=500.0)

# ----------------------------
# Prepare input data
# The column names/order here must match the feature names used when the
# pipeline was trained (see tourism_project/model_building/train.py). The
# saved pipeline includes its own preprocessing (scaling + one-hot encoding),
# so raw values are passed straight through.
# ----------------------------
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Same threshold used during training/evaluation (see train.py)
classification_threshold = 0.45

# Predict button
if st.button("Predict", disabled=model is None):
    try:
        prob = model.predict_proba(input_data)[0, 1]
        pred = int(prob >= classification_threshold)
        result = "will purchase the Wellness Tourism Package" if pred == 1 else "is unlikely to purchase the Wellness Tourism Package"
        if pred == 1:
            st.success(f"Prediction: Customer **{result}**")
        else:
            st.info(f"Prediction: Customer **{result}**")
        st.write(f"Predicted probability of purchase: **{prob:.1%}**")
    except Exception as exc:
        st.error(f"Something went wrong while making the prediction: {exc}")
