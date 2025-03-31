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

st.set_page_config(page_title="Calquix", page_icon="", layout="centered")
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

st.title("Calquix")
st.write("A premium suite of structural calculation tools for engineers and estimators.")

# Tabs for different calculators
tabs = st.tabs(["Steel Weight Calculator", "Concrete Volume Calculator"])

# Steel Weight Calculator
tab1 = tabs[0]
with tab1:
    st.subheader("Steel Weight Calculator")

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
        gewicht_per_meter = profielen_dict[selected_profile]
        gewicht = gewicht_per_meter * lengte if "plaat" not in selected_profile.lower() else None

        st.markdown("**Calculation:**")
        st.code(f"Weight = {gewicht_per_meter:.2f} kg/m × {lengte:.2f} m")

        if "plaat" in selected_profile.lower():
            breedte = st.number_input("Width (m):", min_value=0.0, step=0.1, value=1.0)
            gewicht = gewicht_per_meter * lengte * breedte
            st.code(f"Weight = {gewicht_per_meter:.2f} kg/m² × {lengte:.2f} m × {breedte:.2f} m")

        st.markdown(f"**Estimated weight:** {gewicht:.2f} kg")

    st.markdown("---")
    st.subheader("About Steel Weight Calculator")
    st.write("""
    This tool is designed for professionals who require accurate weight calculations for steel profiles.
    
    - Over 800 standard profiles
    - Plate support with surface calculation
    - Formula display for transparency
    - Optimized for quick project estimates and planning
    - Web-based, no installation required
    """)

    st.caption("Calquix © 2025")

# Concrete Volume Calculator
tab2 = tabs[1]
with tab2:
    st.subheader("Concrete Volume Calculator")

    lengte = st.number_input("Length (m)", min_value=0.0, step=0.1, value=1.0, key="lengte_beton")
    breedte = st.number_input("Width (m)", min_value=0.0, step=0.1, value=1.0, key="breedte_beton")
    hoogte = st.number_input("Height (m)", min_value=0.0, step=0.1, value=0.2, key="hoogte_beton")
    betonklasse = st.selectbox("Concrete strength class:", ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C50/60"])
    stortverlies = st.slider("Pour loss (%):", 0, 20, 5)

    volume = lengte * breedte * hoogte
    volume_corr = volume * (1 + stortverlies / 100)
    gewicht = volume_corr * 2400  # Dichtheid van beton in kg/m3

    st.markdown("**Calculation:**")
    st.code(f"Volume = {lengte:.2f} × {breedte:.2f} × {hoogte:.2f} = {volume:.3f} m³")
    st.code(f"Volume incl. loss = {volume:.3f} × (1 + {stortverlies}% ) = {volume_corr:.3f} m³")
    st.code(f"Weight = {volume_corr:.3f} m³ × 2400 kg/m³ = {gewicht:.1f} kg")

    st.markdown(f"**Estimated volume (incl. loss):** {volume_corr:.3f} m³")
    st.markdown(f"**Estimated weight:** {gewicht:.1f} kg")

    st.markdown("---")
    st.subheader("About Concrete Volume Calculator")
    st.write("""
    This calculator estimates the volume and weight of concrete based on geometry and material assumptions.

    - Concrete density: 2400 kg/m³
    - Includes pour loss margin
    - Concrete class selection
    - Suitable for quick site and estimate calculations
    """)

    st.caption("Calquix © 2025")
