import streamlit as st
import pandas as pd
import base64

# Laad uitgebreide bibliotheek vanuit extern CSV-bestand
@st.cache_data
def laad_profielen():
    return pd.read_csv("https://raw.githubusercontent.com/example/steel-profiles/master/profielen.csv")

profielen_df = laad_profielen()
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

col1, col2 = st.columns(2)
with col1:
    profiel = st.selectbox("🔍 Select Profile", list(profielen_dict.keys()))
with col2:
    lengte = st.number_input("📏 Length (meters)", min_value=0.0, step=0.1, value=1.0)

# Speciale berekening voor platen
if "plaat" in profiel.lower():
    breedte = st.number_input("📐 Width (meters)", min_value=0.0, step=0.1, value=1.0)
    gewicht = profielen_dict[profiel] * lengte * breedte
else:
    gewicht = profielen_dict[profiel] * lengte

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
