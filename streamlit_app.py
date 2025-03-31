import streamlit as st
import pandas as pd
import base64
import re

# Load profile data from external CSV
@st.cache_data
def laad_profielen():
    return pd.read_csv("https://raw.githubusercontent.com/KrisBrabander/SteelCalc/refs/heads/main/alle_staalprofielen.csv")

profielen_df = laad_profielen()

# Extract the profile type based on naming conventions
profielen_df['Type'] = profielen_df['Profiel'].apply(lambda x: re.split(r'\s(?=\d|Ø)', x)[0])
profielen_dict = profielen_df.set_index('Profiel')['Gewicht_per_meter'].to_dict()

st.set_page_config(page_title="SteelCalc Pro", page_icon="", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stApp {
        max-width: 900px;
        margin: auto;
        font-family: "Segoe UI", sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("SteelCalc Pro – Steel Profile Weight Calculator")
st.write("A precise and professional tool to calculate the weight of structural steel profiles.")

# Profile type and selection
type_options = sorted(profielen_df['Type'].dropna().unique())
selected_type = st.selectbox("Select profile category:", type_options)
filtered_profiles = profielen_df[profielen_df['Type'] == selected_type]['Profiel'].tolist()

if filtered_profiles:
    selected_profile = st.selectbox("Select profile:", filtered_profiles)
else:
    st.warning("No profiles found for this category.")
    selected_profile = None

lengte = st.number_input("Length (m):", min_value=0.0, step=0.1, value=1.0)

if selected_profile:
    if "plaat" in selected_profile.lower():
        breedte = st.number_input("Width (m):", min_value=0.0, step=0.1, value=1.0)
        gewicht = profielen_dict[selected_profile] * lengte * breedte
    else:
        gewicht = profielen_dict[selected_profile] * lengte

    st.markdown(f"**Estimated weight:** {gewicht:.2f} kg")

st.markdown("---")
st.subheader("About SteelCalc Pro")
st.write("""
SteelCalc Pro is designed for engineers, estimators, and professionals who require accurate weight calculations for steel profiles. 
The tool is optimized for reliability and clarity.

- Over 800 standard profiles
- Plate support with surface calculation
- Optimized for quick project estimates and planning
- Web-based, no installation required
""")

st.caption("SteelCalc Pro © 2025")
