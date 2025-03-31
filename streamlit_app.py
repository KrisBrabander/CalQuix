import streamlit as st
import pandas as pd
import base64
import re

# Laad uitgebreide bibliotheek vanuit extern CSV-bestand
@st.cache_data
def laad_profielen():
    return pd.read_csv("https://raw.githubusercontent.com/KrisBrabander/SteelCalc/refs/heads/main/alle_staalprofielen.csv")

profielen_df = laad_profielen()

# Betere type-extractie: neem alles voor de eerste cijfer of maat
profielen_df['Type'] = profielen_df['Profiel'].apply(lambda x: re.split(r'\s(?=\d|Ø)', x)[0])
profielen_dict = profielen_df.set_index('Profiel')['Gewicht_per_meter'].to_dict()

st.set_page_config(page_title="SteelCalc Pro", page_icon="🧱", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 800px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧱 SteelCalc Pro – Professional Steel Weight Calculator")
st.write("Effortlessly calculate the exact weight of steel profiles with precision and speed.")

# Keuze via selectbox per type en dan het profiel binnen dat type
type_options = sorted(profielen_df['Type'].dropna().unique())

selected_type = st.selectbox("📁 Select Profile Type", type_options)
filtered_profiles = profielen_df[profielen_df['Type'] == selected_type]['Profiel'].tolist()

if filtered_profiles:
    selected_profile = st.selectbox("🔍 Select Profile", filtered_profiles)
else:
    st.warning("No profiles available for this type.")
    selected_profile = None

lengte = st.number_input("📏 Length (meters)", min_value=0.0, step=0.1, value=1.0)

if selected_profile:
    if "plaat" in selected_profile.lower():
        breedte = st.number_input("📐 Width (meters)", min_value=0.0, step=0.1, value=1.0)
        gewicht = profielen_dict[selected_profile] * lengte * breedte
    else:
        gewicht = profielen_dict[selected_profile] * lengte

    st.markdown(f"""
    ### 🧮 Estimated Weight: `{gewicht:.2f} kg`
    """)

st.markdown("---")
st.subheader("📚 What is SteelCalc Pro?")
st.write("""
SteelCalc Pro is a high-accuracy steel weight calculator for professionals in construction,
engineering, metalworking, and logistics. It supports hundreds of profiles and instantly
computes total weight based on input dimensions.

✔️ Over 800 profiles included  
✔️ Plate support with width and length  
✔️ Fast, responsive, mobile-friendly interface  
✔️ Ideal for estimates, logistics, and purchasing
""")

st.caption("Made for professionals. Built to save time.")
