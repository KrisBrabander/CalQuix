import streamlit as st
import pandas as pd

# Laad uitgebreide bibliotheek vanuit extern CSV-bestand
@st.cache_data
def laad_profielen():
    return pd.read_csv("https://raw.githubusercontent.com/KrisBrabander/SteelCalc/refs/heads/main/alle_staalprofielen.csv")

profielen_df = laad_profielen()
profielen_dict = profielen_df.set_index('Profiel')['Gewicht_per_meter'].to_dict()

st.title("Staalgewicht Calculator")

profiel = st.selectbox("Selecteer het profiel", list(profielen_dict.keys()))
lengte = st.number_input("Lengte (in meters)", min_value=0.0, step=0.1, value=1.0)

# Speciale berekening voor platen
if "plaat" in profiel.lower():
    breedte = st.number_input("Breedte (in meters)", min_value=0.0, step=0.1, value=1.0)
    gewicht = profielen_dict[profiel] * lengte * breedte
    st.write(f"### Gewicht: {gewicht:.2f} kg")
else:
    gewicht = profielen_dict[profiel] * lengte
    st.write(f"### Gewicht: {gewicht:.2f} kg")
