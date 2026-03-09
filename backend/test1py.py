# app.py
import streamlit as st
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

rivers = gpd.read_file(r"C:\Users\tudor\PycharmProjects\ML_Flood_Prediction\backend\HydroRIVERS_v10_eu.gdb", layer="HydroRIVERS_v10_eu")
romania_rivers = rivers.cx[20:30, 43:49]
romania_rivers = romania_rivers[romania_rivers['ORD_FLOW'] >= 3]

rainfall = st.slider("Rainfall (mm)", 0, 100, 20)

romania_rivers['risk'] = rainfall * np.log1p(romania_rivers['UPLAND_SKM'])

st.write("Flood risk map")

fig, ax = plt.subplots(figsize=(10,10))
romania_rivers.plot(ax=ax, linewidth=romania_rivers['ORD_FLOW']*0.3, column='risk', cmap='Reds', legend=True)
st.pyplot(fig)