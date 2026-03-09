import json
from pathlib import Path

import folium
import geopandas as gpd
from matplotlib import colors as mcolors
import numpy as np
import rasterio
import streamlit as st
from branca.element import Element
from shapely.geometry import Point
from shapely.ops import unary_union
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
DEM_PATH = ROOT / "cluj_dem.tif"
RIVERS_PATH = ROOT / "HydroRIVERS_romania.shp"

RISK_HEATMAP = mcolors.LinearSegmentedColormap.from_list(
    "risk_gyr",
    ["#2ca25f", "#fee08b", "#d73027"],
)


@st.cache_data(show_spinner=False)
def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=np.nanmean(arr))
    span = float(arr.max() - arr.min())
    if span <= 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (span + 1e-6)


@st.cache_data(show_spinner=True)
def prepare_static_features(downsample: int = 25):
    with rasterio.open(DEM_PATH) as src:
        dem_full = src.read(1)
        bounds = src.bounds
        transform = src.transform

    dem = dem_full[::downsample, ::downsample]
    rows, cols = dem.shape

    pixel_size_x = transform.a * downsample
    pixel_size_y = transform.e * downsample
    lons_grid = bounds.left + (np.arange(cols) + 0.5) * pixel_size_x
    lats_grid = bounds.top + (np.arange(rows) + 0.5) * pixel_size_y
    lon_grid, lat_grid = np.meshgrid(lons_grid, lats_grid)

    dz_dy, dz_dx = np.gradient(dem)
    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    flow = 1.0 / (1.0 + slope)

    elev_score = 1.0 - normalize(dem)
    slope_score = 1.0 - normalize(slope)
    flow_score = normalize(flow)

    if RIVERS_PATH.exists():
        rivers = gpd.read_file(RIVERS_PATH)
        rivers_roi = rivers.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    else:
        rivers_roi = gpd.GeoDataFrame(geometry=[])

    if len(rivers_roi) == 0:
        river_score = np.zeros_like(dem)
    else:
        river_union = unary_union(rivers_roi.geometry)
        river_dist = np.empty_like(dem, dtype=float)
        for i in range(rows):
            for j in range(cols):
                river_dist[i, j] = river_union.distance(Point(lon_grid[i, j], lat_grid[i, j]))
        river_score = 1.0 - normalize(river_dist)

    # Fallback rainfall susceptibility map from data we always have.
    rain_pattern = 0.6 * flow_score + 0.4 * river_score

    return {
        "bounds": bounds,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "elev_score": elev_score,
        "slope_score": slope_score,
        "flow_score": flow_score,
        "river_score": river_score,
        "rain_pattern": rain_pattern,
        "rivers_roi": rivers_roi,
    }


def compute_dynamic_risk(features: dict, rainfall_pct: int) -> np.ndarray:
    rainfall_factor = rainfall_pct / 100.0
    rain_effect = np.power(np.clip(features["rain_pattern"] * rainfall_factor, 0, 1), 0.85)

    risk = (
        (0.25 * features["slope_score"]
        + 0.25 * features["flow_score"]
        + 0.10 * 1/features["elev_score"]
        + 0.20 * features["river_score"])*0.50
        + 0.50 * rain_effect
    )
    return (np.clip(risk, 0, 1))

def build_map(features: dict, risk: np.ndarray, rainfall_pct: int, opacity: float = 0.68):
    center = [float(features["lat_grid"].mean()), float(features["lon_grid"].mean())]
    bounds = features["bounds"]

    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron", control_scale=True)

    # Keep legal provider attribution, but remove Leaflet prefix text for a cleaner look.
    m.get_root().html.add_child(
        Element(
            """
<style>
.leaflet-control-attribution {
    font-size: 10px !important;
    background: rgba(255, 255, 255, 0.72) !important;
    color: #6c757d !important;
    padding: 1px 6px !important;
    border-radius: 6px 0 0 0;
}
</style>
<script>
setTimeout(function () {
  var maps = document.getElementsByClassName('leaflet-container');
  for (var i = 0; i < maps.length; i++) {
    if (maps[i]._leaflet_map && maps[i]._leaflet_map.attributionControl) {
      maps[i]._leaflet_map.attributionControl.setPrefix('');
    }
  }
}, 0);
</script>
"""
        )
    )

    rgba = (RISK_HEATMAP(risk) * 255).astype(np.uint8)
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=opacity,
        name=f"Flood risk ({rainfall_pct}%)",
        mercator_project=False,
        interactive=False,
    ).add_to(m)

    rivers_roi = features["rivers_roi"]
    if len(rivers_roi) > 0:
        folium.GeoJson(
            rivers_roi.to_json(),
            name="Rivers",
            style_function=lambda x: {"color": "#1864ab", "weight": 1.2, "opacity": 0.75},
        ).add_to(m)

    # Centered custom legend to avoid Leaflet top-right control placement.
    m.get_root().html.add_child(
        Element(
            """
<style>
.risk-legend {
    position: fixed;
    left: 50%;
    bottom: 14px;
    transform: translateX(-50%);
    z-index: 9999;
    background: rgba(255, 255, 255, 0.95);
    padding: 8px 10px;
    border: 1px solid #ced4da;
    border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.18);
    font-family: Arial, sans-serif;
    color: #212529;
}
.risk-legend .caption {
    font-size: 12px;
    margin-bottom: 6px;
}
.risk-legend .bar {
    width: 260px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid #adb5bd;
    background: linear-gradient(to right, #2ca25f 0%, #fee08b 50%, #d73027 100%);
}
.risk-legend .ticks {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin-top: 3px;
}
</style>
<div class="risk-legend">
  <div class="caption">Relative flood risk (0.0 to 1.0)</div>
  <div class="bar"></div>
  <div class="ticks"><span>0.0</span><span>0.5</span><span>1.0</span></div>
</div>
"""
        )
    )

    folium.LayerControl(collapsed=True).add_to(m)
    return m


def main():
    st.set_page_config(page_title="Flood Risk Simulator", layout="wide")
    st.markdown(
        """
<style>
[data-testid="stAppDeployButton"],
.stAppDeployButton,
button[title="Deploy"] {
    display: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
    st.title("Flood Risk Simulator")
    st.caption("Risk reacts to rainfall while preserving terrain and river structure from your local data.")

    downsample = st.sidebar.slider("Grid downsample", min_value=10, max_value=60, value=25, step=5)
    rainfall_pct = st.sidebar.slider("Rainfall (%)", min_value=0, max_value=200, value=100, step=10)

    with st.spinner("Preparing terrain and river features..."):
        features = prepare_static_features(downsample=downsample)

    risk = compute_dynamic_risk(features, rainfall_pct)
    m = build_map(features, risk, rainfall_pct)

    col1, col2 = st.columns([3, 1])
    with col1:
        st_folium(m, width=1100, height=720)
    with col2:
        st.metric("Rainfall", f"{rainfall_pct}%")
        st.metric("Mean risk", f"{risk.mean():.3f}")
        st.metric("Max risk", f"{risk.max():.3f}")
        if st.button("Export current view to HTML"):
            out_file = ROOT / "cluj_flood_risk_streamlit_export.html"
            m.save(str(out_file))
            st.success(f"Saved: {out_file.name}")


if __name__ == "__main__":
    main()