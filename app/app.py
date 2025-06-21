import joblib
import streamlit as st
import pandas as pd

from utils import get_ordinal_mappings, get_ethnicity_booleans

@st.cache_resource
def load_model():
    """Loads the model and scaler."""
    model_path = '../pipeline/model.pkl'
    scaler_path = '../pipeline/scaler.pkl'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load the model
model, scaler = load_model()

# Get the mappings
ordinal_map = get_ordinal_mappings()

st.title("Food Security Status Predictor")
st.markdown("""
Predict the food security level of a household based on key characteristics.
""")

# Sidebar content
with st.sidebar:
    st.header(":information_source: **About this App**")
    st.markdown("""
This model predicts a household's food security score based on survey data.

Developed using **XGBoost**, trained on data from **Nairobi informal settlements (2014)**.
""")

# Inputs
hhedu = st.selectbox("Education Level", list(ordinal_map['hhedu'].keys()))
hhhage = st.slider("Age of Househould Head", min_value=18, max_value=120, value=35)
hhsize = st.slider("Househould Size", min_value=1, max_value=20, value=5)
u05 = st.slider("Number of Childer Under 5", min_value=0, max_value=10, value=1)
windex3 = st.selectbox("Wealth Tertile of Household", list(ordinal_map['windex3'].keys()))
windex5 = st.selectbox("Wealth Quintile of Household", list(ordinal_map['windex5'].keys()))
site_viwandani = st.selectbox("Site Viwandani?", ["Yes", "No"]) == "Yes"
hhhsex_male = st.selectbox("Household Head is Male?", ["Yes", "No"]) == "Yes"
povline_yes = st.selectbox("Below Poverty Line?", ["Yes", "No"]) == "Yes"
ethnicity = st.selectbox("Ethnicity", ["Kikuyu", "Kisii", "Luhya", "Luo", "Other"])

# Get ethnicity values
eth_vals = get_ethnicity_booleans(ethnicity)

# Input data for ML model
input_data = {
    "hhedu": ordinal_map['hhedu'][hhedu],
    "hhhage": hhhage,
    "hhsize": hhsize,
    "u05": u05,
    "windex3": ordinal_map['windex3'][windex3],
    "windex5": ordinal_map['windex5'][windex5],
    "site_viwandani": site_viwandani,
    "hhhsex_male": hhhsex_male,
    "hhethnic_Kikuyu": eth_vals[0],
    "hhethnic_Kisii": eth_vals[1],
    "hhethnic_Luhya": eth_vals[2],
    "hhethnic_Luo": eth_vals[3],
    "hhethnic_Other": eth_vals[4],
    "povline_yes": int(povline_yes),
    "dependency_ratio": u05/hhsize
}

input_df = pd.DataFrame([input_data])

# Preprocess the input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict FS Score"):
    prediction = model.predict(input_scaled)[0]
    probas = model.predict_proba(input_scaled)[0] * 100
    probas_dict = enumerate(probas)
    probas_df = pd.DataFrame(
        probas_dict, columns=["Class", "Probability (%)"]
    )

    # FS Score labels
    fs_labels = {
        0: "Secure",
        1: "Moderately insecure",
        2: "Moderately insecure",
        3: "Severely insecure",
        4: "Severely insecure"
    }

    st.subheader(f"Predicted FS_Score: {prediction} - {fs_labels[prediction]}")

    st.markdown("### Probability per Class")
    st.markdown("#### Probability Table")
    st.table(probas_df)

    st.markdown("#### Probability Chart")
    st.bar_chart(probas_df.set_index("Class"))