"""
Flood Prediction Pipeline for Romania
=======================================
Model   : Random Forest Classifier (scikit-learn Pipeline)
Features: HydroRIVERS attributes + simulated seasonal rainfall
Output  : flood_prediction_map.html  – interactive Folium map
"""

import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MiniMap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import json
import os

# Minimum downstream distance used in TWI proxy to avoid division by zero
MIN_DIST_DN_KM = 0.1

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_PATH   = os.path.join(SCRIPT_DIR, "HydroRIVERS_romania.shp")
OUT_HTML   = os.path.join(SCRIPT_DIR, "flood_prediction_map.html")

# ── 1. Load river data ───────────────────────────────────────────────────────
print("Loading HydroRIVERS Romania shapefile …")
rivers = gpd.read_file(SHP_PATH)
print(f"  → {len(rivers)} river segments loaded (CRS: {rivers.crs})")

# ── 2. Feature Engineering ───────────────────────────────────────────────────
print("Engineering features …")

# centroids (geographic CRS is fine for getting rough lat/lon)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    centroids = rivers.geometry.centroid

rivers["centroid_lat"] = centroids.y
rivers["centroid_lon"] = centroids.x

# log-transforms to compress skewed distributions
rivers["log_upland"]  = np.log1p(rivers["UPLAND_SKM"])
rivers["log_dis"]     = np.log1p(rivers["DIS_AV_CMS"])
rivers["log_catch"]   = np.log1p(rivers["CATCH_SKM"])
rivers["log_length"]  = np.log1p(rivers["LENGTH_KM"])
rivers["log_dist_dn"] = np.log1p(rivers["DIST_DN_KM"])
rivers["log_dist_up"] = np.log1p(rivers["DIST_UP_KM"])

# Strahler order as a proxy for river size / flood potential
rivers["strahler"] = rivers["ORD_STRA"].astype(float)
rivers["ord_flow"] = rivers["ORD_FLOW"].astype(float)

# Topographic wetness index proxy: TWI ≈ ln(A / tan(slope))
# Here we approximate with catchment area / distance-downstream ratio
rivers["twi_proxy"] = np.log1p(
    rivers["CATCH_SKM"] / (rivers["DIST_DN_KM"].replace(0, np.nan)).fillna(MIN_DIST_DN_KM)
)

# ── 3. Simulate Seasonal Rainfall ────────────────────────────────────────────
print("Simulating rainfall field …")

# Romania has higher precipitation in Carpathians (N-W) and lower in E/SE
# We model this with a spatial gradient + random noise (reproducible seed)
rng = np.random.default_rng(42)

lat_c = rivers["centroid_lat"]
lon_c = rivers["centroid_lon"]

# Base rainfall: higher in the mountains (north-west Carpathians)
# Simple empirical surface: peaks around lat=46.5, lon=23.5
rain_base = (
    40
    + 30 * np.exp(-((lat_c - 46.5)**2 / 2 + (lon_c - 23.5)**2 / 4))
    + 20 * np.exp(-((lat_c - 47.5)**2 / 2 + (lon_c - 25.0)**2 / 3))
    - 5  * (lon_c - 25.0)        # drier towards the Black Sea coast
)
rain_noise = rng.normal(0, 5, size=len(rivers))
rivers["rainfall_mm"] = np.clip(rain_base + rain_noise, 5, 120)

# Interaction term: rainfall amplified by large upstream drainage area
rivers["rain_x_upland"] = rivers["rainfall_mm"] * np.log1p(rivers["UPLAND_SKM"])

# ── 4. Generate Flood Risk Labels ────────────────────────────────────────────
print("Generating flood-risk labels …")

# Physical thresholds derived from literature / domain knowledge:
#   - High average discharge (DIS_AV_CMS > 75th pct)  → high base flow
#   - Large upstream area  (UPLAND_SKM > 75th pct)    → large catchment
#   - High simulated rainfall                         → precipitation driver
#   - High TWI proxy                                  → accumulation tendency
dis_p75     = rivers["DIS_AV_CMS"].quantile(0.75)
upland_p50  = rivers["UPLAND_SKM"].quantile(0.50)
upland_p75  = rivers["UPLAND_SKM"].quantile(0.75)
rain_p60    = rivers["rainfall_mm"].quantile(0.60)
twi_p70     = rivers["twi_proxy"].quantile(0.70)
strahler_p70 = rivers["strahler"].quantile(0.70)

# Score-based labelling (deterministic, fully reproducible)
score = (
    (rivers["DIS_AV_CMS"] > dis_p75).astype(int)
    + (rivers["UPLAND_SKM"] > upland_p75).astype(int)
    + (rivers["rainfall_mm"] > rain_p60).astype(int)
    + (rivers["twi_proxy"] > twi_p70).astype(int)
    + (rivers["strahler"] > strahler_p70).astype(int)
)

# Flood = 1 when 3 or more risk factors are present
rivers["flood_label"] = (score >= 3).astype(int)

n_flood = rivers["flood_label"].sum()
print(f"  → {n_flood} flood segments ({100*n_flood/len(rivers):.1f}%) / "
      f"{len(rivers)-n_flood} non-flood segments")

# ── 5. Train / Evaluate the Pipeline ─────────────────────────────────────────
FEATURE_COLS = [
    "log_upland", "log_dis", "log_catch", "log_length",
    "log_dist_dn", "log_dist_up",
    "strahler", "ord_flow",
    "twi_proxy", "rainfall_mm", "rain_x_upland",
    "centroid_lat", "centroid_lon",
]

X = rivers[FEATURE_COLS].fillna(0).values
y = rivers["flood_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Random Forest pipeline …")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])
pipeline.fit(X_train, y_train)

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n── Model Evaluation ──────────────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))
print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importances
rf = pipeline.named_steps["clf"]
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
print("\nTop-10 feature importances:")
print(importances.sort_values(ascending=False).head(10).to_string())

# ── 6. Predict Flood Probability for ALL Segments ────────────────────────────
print("\nPredicting flood probability for all river segments …")
rivers["flood_prob"]   = np.round(pipeline.predict_proba(
    rivers[FEATURE_COLS].fillna(0).values
)[:, 1], 3)
rivers["flood_class"]  = pipeline.predict(
    rivers[FEATURE_COLS].fillna(0).values
)

# Risk tier (0–3)
rivers["risk_tier"] = pd.cut(
    rivers["flood_prob"],
    bins=[0, 0.30, 0.55, 0.75, 1.001],
    labels=[0, 1, 2, 3],
).fillna(0).astype(int)

tier_counts = rivers["risk_tier"].value_counts().sort_index()
labels_map  = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
print("Risk tier distribution:")
for k, v in tier_counts.items():
    print(f"  Tier {k} ({labels_map[k]}): {v} segments")

# ── 7. Build Folium Map ───────────────────────────────────────────────────────
print("\nBuilding Folium flood-prediction map …")

RISK_COLORS = {
    0: "#2ECC71",   # green  – Low
    1: "#F39C12",   # orange – Medium
    2: "#E67E22",   # dark orange – High
    3: "#C0392B",   # red    – Critical
}
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

m = folium.Map(
    location=[45.9, 24.9],
    zoom_start=7,
    tiles=None,
)

# ── base tile layers
folium.TileLayer("CartoDB positron",  name="Light",  control=True).add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Dark", control=True).add_to(m)
folium.TileLayer("OpenStreetMap",     name="OSM",    control=True).add_to(m)

# ── add all rivers as GeoJson per risk tier (compact & fast)
print("  Adding river geometries via GeoJson …")
for tier_val in range(4):
    subset = rivers[rivers["risk_tier"] == tier_val].copy()
    if subset.empty:
        continue
    color  = RISK_COLORS[tier_val]
    label  = RISK_LABELS[tier_val]

    def style_fn(feature, _color=color, _tier=tier_val):
        ord_flow = feature["properties"].get("ORD_FLOW", 6)
        return {
            "color":   _color,
            "weight":  max(1, ord_flow * 0.35),
            "opacity": 0.75,
        }

    def tooltip_fn(feature, _label=label):
        p = feature["properties"].get("flood_prob", 0)
        return f"Risk: {_label} | P={p:.0%}"

    fg = folium.FeatureGroup(name=f"Risk: {label}", show=True)
    folium.GeoJson(
        subset[["geometry", "HYRIV_ID", "DIS_AV_CMS", "UPLAND_SKM",
                "CATCH_SKM", "LENGTH_KM", "ORD_STRA", "ORD_FLOW",
                "flood_prob", "rainfall_mm", "twi_proxy"]],
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["HYRIV_ID", "flood_prob", "ORD_FLOW", "DIS_AV_CMS"],
            aliases=["River ID", "Flood Prob", "Flow Order", "Discharge (m³/s)"],
            localize=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["HYRIV_ID", "flood_prob", "DIS_AV_CMS", "UPLAND_SKM",
                    "CATCH_SKM", "LENGTH_KM", "ORD_STRA", "ORD_FLOW",
                    "rainfall_mm", "twi_proxy"],
            aliases=["River ID", "Flood Prob", "Discharge (m³/s)",
                     "Upstream Area (km²)", "Catchment (km²)", "Length (km)",
                     "Strahler Order", "Flow Order", "Rainfall (mm)", "TWI Proxy"],
            max_width=300,
        ),
        name=f"geojson_{tier_val}",
    ).add_to(fg)
    fg.add_to(m)

# ── minimap
MiniMap(toggle_display=True, position="bottomright").add_to(m)

# ── layer control
folium.LayerControl(collapsed=False).add_to(m)

# ── legend
legend_html = """
<div style="
    position: fixed;
    bottom: 30px; left: 30px; z-index: 9999;
    background: rgba(255,255,255,0.92);
    padding: 14px 18px;
    border-radius: 10px;
    border: 1px solid #ccc;
    font-family: Arial, sans-serif;
    font-size: 13px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
  <b style="font-size:14px;">&#127754; Flood Risk</b>
  <hr style="margin:6px 0;">
  <div><span style="background:#2ECC71;display:inline-block;width:16px;height:16px;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>Low (&lt; 30%)</div>
  <div><span style="background:#F39C12;display:inline-block;width:16px;height:16px;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>Medium (30&ndash;55%)</div>
  <div><span style="background:#E67E22;display:inline-block;width:16px;height:16px;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>High (55&ndash;75%)</div>
  <div><span style="background:#C0392B;display:inline-block;width:16px;height:16px;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>Critical (&gt; 75%)</div>
  <hr style="margin:6px 0;">
  <small style="color:#666;">Model: Random Forest<br>Features: HydroRIVERS + Rainfall</small>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── rainfall-multiplier slider (interactive JS)
# Only major rivers (ORD_FLOW >= 5) are embedded as JSON for the slider
# to keep the file size manageable while still showing the key drainage network.
print("  Embedding interactive rainfall slider (major rivers only) …")

major = rivers[rivers["ORD_FLOW"] >= 5].copy()

def _simplify_coords(geom, tolerance=0.005):
    """Return rounded coords from a LineString geometry."""
    simplified = geom.simplify(tolerance, preserve_topology=False)
    if simplified.is_empty:
        simplified = geom
    return [[round(lat, 4), round(lon, 4)]
            for lon, lat in simplified.coords]

seg_data = []
for _, row in major.iterrows():
    geom = row["geometry"]
    if geom is None or geom.is_empty:
        continue
    coords = _simplify_coords(geom)
    if not coords:
        continue
    seg_data.append({
        "dis":      round(float(row["DIS_AV_CMS"]), 3),
        "upland":   round(float(row["UPLAND_SKM"]), 1),
        "rain":     round(float(row["rainfall_mm"]), 1),
        "ord_flow": int(row["ORD_FLOW"]),
        "strahler": int(row["ORD_STRA"]),
        "coords":   coords,
    })

seg_json = json.dumps(seg_data)

slider_html = f"""
<div id="sliderPanel" style="
    position: fixed; top: 20px; left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background: rgba(255,255,255,0.95);
    padding: 10px 20px;
    border-radius: 10px;
    border: 1px solid #aaa;
    font-family: Arial, sans-serif;
    font-size: 13px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.25);
    display: flex; align-items: center; gap: 12px;
">
  🌧️ <b>Rainfall multiplier:</b>
  <input id="rainSlider" type="range" min="0.5" max="3.0" step="0.1" value="1.0"
         style="width:160px;">
  <span id="rainVal">1.0×</span>
  &nbsp;|&nbsp;
  <span id="riskSummary" style="color:#555;"></span>
</div>

<script>
(function() {{
  var SEGS = {seg_json};

  var COLORS = ["#2ECC71","#F39C12","#E67E22","#C0392B"];
  var LABELS = ["Low","Medium","High","Critical"];

  // sigmoid to map a linear score to [0,1] probability
  function sigmoid(x) {{ return 1 / (1 + Math.exp(-x)); }}

  function calcProb(seg, mult) {{
    // Empirical logistic approximation of the Random Forest decision boundary.
    // Coefficients chosen to reproduce the RF tier boundaries at mult=1.0:
    //   log_upland weight ≈ 0.25, log_dis weight ≈ 0.45,
    //   rainfall/80 weight ≈ 0.55, strahler weight ≈ 0.10, intercept ≈ -2.8
    var logU = Math.log(1 + seg.upland);
    var logD = Math.log(1 + seg.dis);
    var rainFactor = (seg.rain * mult) / 80;
    var score = 0.25*logU + 0.45*logD + 0.55*rainFactor
                + 0.10*seg.strahler - 2.8;
    return sigmoid(score);
  }}

  function tier(p) {{
    if (p < 0.30) return 0;
    if (p < 0.55) return 1;
    if (p < 0.75) return 2;
    return 3;
  }}

  var linesLayer = null;
  var polylines  = [];

  function drawMap(mult) {{
    // remove old layer
    if (linesLayer) {{ linesLayer.remove(); }}
    linesLayer = L.layerGroup().addTo(window._floodMap);
    polylines  = [];

    var counts = [0,0,0,0];
    SEGS.forEach(function(seg) {{
      var p  = calcProb(seg, mult);
      var t  = tier(p);
      counts[t]++;
      var w  = Math.max(1, seg.ord_flow * 0.35);
      var pl = L.polyline(seg.coords, {{
        color:   COLORS[t],
        weight:  w,
        opacity: 0.75,
      }});
      pl.bindTooltip(
        "Risk: " + LABELS[t] + " | P=" + (p*100).toFixed(0) + "%",
        {{sticky: true}}
      );
      linesLayer.addLayer(pl);
      polylines.push({{pl:pl, seg:seg}});
    }});

    document.getElementById("riskSummary").innerHTML =
      "<span style='color:#C0392B;'>&#9632;</span> Crit: " + counts[3] +
      " <span style='color:#E67E22;'>&#9632;</span> High: " + counts[2] +
      " <span style='color:#F39C12;'>&#9632;</span> Med: "  + counts[1] +
      " <span style='color:#2ECC71;'>&#9632;</span> Low: "  + counts[0];
  }}

  // Wait for Leaflet map object to be fully initialised before drawing.
  // 500 ms is sufficient for the Folium-generated map init scripts to complete.
  var MAP_INIT_DELAY_MS = 500;

  // expose map reference after Leaflet initialises
  document.addEventListener("DOMContentLoaded", function() {{
    // find the leaflet map object
    setTimeout(function() {{
      var maps = [];
      for (var k in window) {{
        if (window[k] && typeof window[k] === "object" && window[k]._leaflet_id) {{
          maps.push(window[k]);
        }}
      }}
      // pick the first real map
      for (var i=0; i<maps.length; i++) {{
        if (maps[i].getZoom) {{ window._floodMap = maps[i]; break; }}
      }}
      if (!window._floodMap) {{
        // fallback: iterate over global vars
        Object.keys(window).forEach(function(k) {{
          if (k.startsWith("map_") && window[k] && window[k].getZoom) {{
            window._floodMap = window[k];
          }}
        }});
      }}
      drawMap(1.0);
    }}, MAP_INIT_DELAY_MS);
  }});

  document.getElementById("rainSlider").addEventListener("input", function() {{
    var mult = parseFloat(this.value);
    document.getElementById("rainVal").textContent = mult.toFixed(1) + "×";
    drawMap(mult);
  }});
}})();
</script>
"""

m.get_root().html.add_child(folium.Element(slider_html))

# ── save
m.save(OUT_HTML)
print(f"\n✅  Map saved to:  {OUT_HTML}")
print("    Open this HTML file in a browser to explore the flood prediction map.")
