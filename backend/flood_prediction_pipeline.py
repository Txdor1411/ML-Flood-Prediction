"""
Flood Prediction Pipeline for Romania
======================================
Uses HydroRIVERS data with a Random Forest model to predict flood risk
per river segment under varying rainfall conditions.

Outputs:
  - flood_risk_map.html  : interactive Folium map with rainfall slider
"""

import json
import math
import os

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Load river data
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_PATH = os.path.join(SCRIPT_DIR, "HydroRIVERS_romania.shp")

print("Loading HydroRIVERS Romania ...")
rivers = gpd.read_file(SHP_PATH)
# ORD_FLOW: 1=world's largest rivers ... 9=small streams.
# Keep orders 3-6 (major & medium rivers) for interactive map performance.
rivers = rivers[rivers["ORD_FLOW"] <= 6].copy()
rivers = rivers.to_crs(epsg=4326)

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(gdf: gpd.GeoDataFrame, rainfall_mm: float = 0.0) -> pd.DataFrame:
    """
    Build feature matrix from HydroRIVERS attributes and a rainfall value.

    Features:
      log_upland    : log(1 + upstream catchment area in km²)
      log_discharge : log(1 + average discharge in m³/s)
      log_catch     : log(1 + local catchment area in km²)
      log_length    : log(1 + segment length in km)
      ord_flow      : Strahler flow order (river size class)
      dist_ratio    : DIST_DN_KM / (DIST_UP_KM + 1)  (proximity to outlet)
      rainfall_mm   : external rainfall parameter (mm)
      rain_x_up     : interaction term: rainfall × log_upland
    """
    df = pd.DataFrame()
    df["log_upland"]    = np.log1p(gdf["UPLAND_SKM"])
    df["log_discharge"] = np.log1p(gdf["DIS_AV_CMS"])
    df["log_catch"]     = np.log1p(gdf["CATCH_SKM"])
    df["log_length"]    = np.log1p(gdf["LENGTH_KM"])
    df["ord_flow"]      = gdf["ORD_FLOW"].astype(float)
    df["dist_ratio"]    = gdf["DIST_DN_KM"] / (gdf["DIST_UP_KM"] + 1.0)
    df["rainfall_mm"]   = float(rainfall_mm)
    df["rain_x_up"]     = df["rainfall_mm"] * df["log_upland"]
    return df


def compute_physics_risk(gdf: gpd.GeoDataFrame, rainfall_mm: float) -> np.ndarray:
    """
    Physics-inspired flood risk score in [0, 1].

    Risk captures:
      - upstream catchment size  (larger → more runoff)
      - average discharge        (higher baseline flow → faster overbank)
      - rainfall                 (direct driver)
      - stream order             (smaller streams flood locally faster)
    """
    # Normalize heavy drivers to [0,1] range seen in training data
    max_up  = gdf["UPLAND_SKM"].max()
    max_dis = gdf["DIS_AV_CMS"].max()
    max_ord = gdf["ORD_FLOW"].max()

    up_norm  = gdf["UPLAND_SKM"] / (max_up  + 1e-9)
    dis_norm = gdf["DIS_AV_CMS"] / (max_dis + 1e-9)
    ord_inv  = 1.0 - gdf["ORD_FLOW"] / (max_ord + 1e-9)  # small order → local flash-flood risk
    rain_norm = float(rainfall_mm) / 200.0                # normalize to 200 mm max

    risk = (
        0.40 * up_norm
        + 0.25 * dis_norm
        + 0.20 * rain_norm * (up_norm + dis_norm) / 2.0
        + 0.15 * ord_inv
    )
    return np.clip(risk.values, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Build & train the pipeline (target = physics risk at median rainfall)
# ---------------------------------------------------------------------------
TRAIN_RAINFALL = 30.0   # mm – representative training scenario

print("Engineering features ...")
X_train = engineer_features(rivers, rainfall_mm=TRAIN_RAINFALL)
y_train = compute_physics_risk(rivers, rainfall_mm=TRAIN_RAINFALL)

print("Training Random Forest flood-risk model ...")
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf",     RandomForestRegressor(
                   n_estimators=200,
                   max_depth=8,
                   min_samples_leaf=5,
                   n_jobs=-1,
                   random_state=42,
               )),
])
model_pipeline.fit(X_train, y_train)

feature_names = list(X_train.columns)
importances   = model_pipeline.named_steps["rf"].feature_importances_
print("\nFeature importances:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:18s}  {imp:.3f}")


def predict_risk(rainfall_mm: float) -> np.ndarray:
    """Return predicted flood risk scores for all river segments."""
    X = engineer_features(rivers, rainfall_mm=rainfall_mm)
    return np.clip(model_pipeline.predict(X), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Build Folium map
# ---------------------------------------------------------------------------
RAINFALL_SCENARIOS = list(range(0, 201, 10))   # 0, 10, 20, ... 200 mm

print("\nGenerating flood risk predictions for all rainfall scenarios ...")
all_risks: dict[int, list[float]] = {
    r: [round(v, 3) for v in predict_risk(r).tolist()]
    for r in RAINFALL_SCENARIOS
}

# Build GeoJSON per river segment (only attributes we need)
# seg_id is the positional index (0-based) so it aligns with predict_risk arrays
features_json = []
pos = 0
for _, row in rivers.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        pos += 1
        continue
    try:
        geom_dict = geom.__geo_interface__
    except AttributeError:
        print(f"Warning: skipping segment at position {pos} — geometry has no __geo_interface__")
        pos += 1
        continue
    features_json.append({
        "type": "Feature",
        "geometry": geom_dict,
        "properties": {
            "seg_id":    pos,
            "ord_flow":  int(row["ORD_FLOW"]),
            "upland_km": round(float(row["UPLAND_SKM"]), 1),
            "dis_cms":   round(float(row["DIS_AV_CMS"]), 2),
        },
    })
    pos += 1

# Embed risk arrays keyed by rainfall
risk_lookup: dict[str, list[float]] = {}
for r, vals in all_risks.items():
    risk_lookup[str(r)] = vals

print("Building Folium map ...")
m = folium.Map(location=[46.0, 25.0], zoom_start=7, tiles="CartoDB positron")

# Embed data into the map HTML via JavaScript
geojson_str   = json.dumps({"type": "FeatureCollection", "features": features_json})
risk_json_str = json.dumps(risk_lookup)

# Colour scale: green → yellow → red (0 → 1)
colormap = LinearColormap(
    colors=["#00cc44", "#ffff00", "#ff6600", "#cc0000"],
    vmin=0.0, vmax=1.0,
    caption="Flood Risk (0 = low, 1 = high)",
)
colormap.add_to(m)

js_script = f"""
<script>
(function() {{
  var geojsonData  = {geojson_str};
  var riskLookup   = {risk_json_str};
  var currentRain  = 30;

  function riskToColor(r) {{
    if (r < 0.25) return '#00cc44';
    if (r < 0.50) return '#ffff00';
    if (r < 0.75) return '#ff6600';
    return '#cc0000';
  }}

  function riskToWidth(r, ordFlow) {{
    return 0.5 + r * 3.0 + ordFlow * 0.15;
  }}

  var riverLayer = null;

  function getRisks(rain) {{
    var key = String(Math.round(rain / 10) * 10);
    return riskLookup[key] || riskLookup['30'];
  }}

  function drawRivers(rain) {{
    if (riverLayer) {{ riverLayer.remove(); }}
    var risks = getRisks(rain);
    riverLayer = L.geoJSON(geojsonData, {{
      style: function(feature) {{
        var pos  = feature.properties.seg_id;
        var risk = risks[pos] !== undefined ? risks[pos] : 0.0;
        return {{
          color:   riskToColor(risk),
          weight:  riskToWidth(risk, feature.properties.ord_flow),
          opacity: 0.85,
        }};
      }},
      onEachFeature: function(feature, layer) {{
        var pos  = feature.properties.seg_id;
        var risk = risks[pos] !== undefined ? risks[pos] : 0.0;
        layer.bindTooltip(
          '<b>Flood Risk: ' + (risk * 100).toFixed(1) + '%</b><br>' +
          'Upstream area: ' + feature.properties.upland_km + ' km²<br>' +
          'Mean discharge: ' + feature.properties.dis_cms + ' m³/s<br>' +
          'Stream order: ' + feature.properties.ord_flow,
          {{sticky: true}}
        );
      }},
    }}).addTo(window._map);
  }}

  // Wait for Leaflet map to be ready
  function init() {{
    var mapDivs = document.querySelectorAll('.folium-map');
    if (!mapDivs.length) {{ setTimeout(init, 200); return; }}
    // Retrieve the Leaflet map object created by Folium
    for (var key in window) {{
      if (window[key] && window[key]._leaflet_id !== undefined &&
          typeof window[key].addLayer === 'function') {{
        window._map = window[key];
        break;
      }}
    }}
    if (!window._map) {{ setTimeout(init, 200); return; }}
    drawRivers(currentRain);
  }}
  init();

  // Slider events
  window.updateFloodMap = function(val) {{
    currentRain = parseInt(val);
    document.getElementById('rain-label').textContent = currentRain + ' mm';
    drawRivers(currentRain);
  }};
}})();
</script>
"""

slider_html = """
<div style="
    position: fixed;
    top: 12px; left: 55px;
    z-index: 9999;
    background: rgba(255,255,255,0.93);
    padding: 10px 14px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    font-family: Arial, sans-serif;
    font-size: 13px;
    min-width: 220px;
">
  <b>&#127783;&#65039; Flood Prediction &mdash; Romania</b><br>
  <label>Rainfall: <span id="rain-label">30 mm</span></label><br>
  <input type="range" min="0" max="200" step="10" value="30"
         style="width:190px; margin-top:4px;"
         oninput="updateFloodMap(this.value)">
  <br><small style="color:#555;">Random Forest model · HydroRIVERS data</small>
</div>
"""

m.get_root().html.add_child(folium.Element(slider_html))
m.get_root().html.add_child(folium.Element(js_script))

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
out_path = os.path.join(SCRIPT_DIR, "flood_risk_map.html")
m.save(out_path)
print(f"\n✅  Flood risk map saved → {out_path}")
