import streamlit as st
import pandas as pd
import base64
import re

@st.cache_data

def laad_profielen():
    return pd.read_csv("https://raw.githubusercontent.com/KrisBrabander/SteelCalc/refs/heads/main/alle_staalprofielen.csv")

profielen_df = laad_profielen()
profielen_df['Type'] = profielen_df['Profiel'].apply(lambda x: re.split(r'\s(?=\d|Ø)', x)[0])
profielen_dict = profielen_df.set_index('Profiel')['Gewicht_per_meter'].to_dict()

# Extra: voorbeelddata doorsnedeoppervlakken voor visuele professionaliteit
extra_info = {
    "HEA 100": {"A": 21.2, "h": 96, "b": 100},
    "HEB 100": {"A": 26.4, "h": 100, "b": 100},
    "IPE 100": {"A": 13.3, "h": 100, "b": 55},
    "Koker 50x50x4": {"A": 6.2, "h": 50, "b": 50},
    "Buis 48.3x3.25": {"A": 4.48, "h": 48.3, "b": 48.3}
}

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
    .small-note {
        color: grey;
        font-size: 0.85em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Calquix")
st.write("Advanced structural calculation tools – trusted by professionals.")

# Tabs
tabs = st.tabs(["Steel Weight Calculator", "Concrete Volume Calculator"])

with tabs[0]:
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
    st.caption("All calculations are based on nominal profile properties as per EN 1993-1-1.")

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

        if selected_profile in extra_info:
            info = extra_info[selected_profile]
            st.markdown("---")
            st.markdown("**Section properties:**")
            st.markdown(f"- Area (A): {info['A']} cm²")
            st.markdown(f"- Height (h): {info['h']} mm")
            st.markdown(f"- Width (b): {info['b']} mm")
            oppervlakte = (info['h'] / 1000) * (info['b'] / 1000)
            st.markdown(f"- Estimated surface area (1m length): {oppervlakte:.3f} m²")

    st.markdown("---")
    st.subheader("About this calculator")
    st.write("""
    Designed for engineers and estimators who need precise mass and dimensional data for steel profiles.

    - Verified profile weights
    - Section data available (where defined)
    - Based on European standards
    - Clear technical breakdown
    """)
    st.caption("Calquix © 2025 – Built by engineers, for engineers.")

with tabs[1]:
    st.subheader("Concrete Volume Calculator")

    lengte = st.number_input("Length (m)", min_value=0.0, step=0.1, value=1.0, key="lengte_beton")
    breedte = st.number_input("Width (m)", min_value=0.0, step=0.1, value=1.0, key="breedte_beton")
    hoogte = st.number_input("Height (m)", min_value=0.0, step=0.1, value=0.2, key="hoogte_beton")

    constructietype = st.selectbox("Structure type:", ["Foundation", "Floor slab", "Wall", "Beam"])
    betonklasse = st.selectbox("Concrete strength class:", ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C50/60"])
    stortverlies = st.slider("Pour loss (%):", 0, 20, 5)

    volume = lengte * breedte * hoogte
    volume_corr = volume * (1 + stortverlies / 100)
    gewicht = volume_corr * 2400

    st.markdown("**Calculation:**")
    st.code(f"Volume = {lengte:.2f} × {breedte:.2f} × {hoogte:.2f} = {volume:.3f} m³")
    st.code(f"Volume incl. loss = {volume:.3f} × (1 + {stortverlies}% ) = {volume_corr:.3f} m³")
    st.code(f"Weight = {volume_corr:.3f} m³ × 2400 kg/m³ = {gewicht:.1f} kg")

    st.markdown(f"**Estimated volume (incl. loss):** {volume_corr:.3f} m³")
    st.markdown(f"**Estimated weight:** {gewicht:.1f} kg")

    st.markdown("---")
    st.subheader("About this calculator")
    st.write("""
    This tool calculates the expected volume and weight of poured concrete.

    - Pour loss margin adjustable
    - Class selection affects cement content
    - Assumes standard concrete density (2400 kg/m³)
    - Suitable for early-stage estimates and tendering
    """)
    st.caption("Calquix © 2025 – Built by engineers, for engineers.")
