from pathlib import Path

import folium
import geopandas as gpd
from matplotlib import colors as mcolors
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEM_PATH = ROOT / "romania_dem.tif"
RIVERS_PATH = ROOT / "HydroRIVERS_romania.shp"
OUT_PATH = PROJECT_ROOT / "app" / "FloodGuardMobile" / "assets" / "maps" / "romania_flood_risk_full.html"

RISK_HEATMAP = mcolors.LinearSegmentedColormap.from_list(
    "risk_gyr",
    ["#2ca25f", "#fee08b", "#d73027"],
)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=np.nanmean(arr))
    span = float(arr.max() - arr.min())
    if span <= 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (span + 1e-6)


def compute_country_risk(downsample: int = 25) -> tuple[np.ndarray, rasterio.coords.BoundingBox, gpd.GeoDataFrame]:
    if not DEM_PATH.exists():
        raise FileNotFoundError(f"Missing DEM raster: {DEM_PATH}")

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
        if "ORD_FLOW" in rivers.columns:
            rivers = rivers[rivers["ORD_FLOW"] >= 4]
        rivers_roi = rivers.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    else:
        rivers_roi = gpd.GeoDataFrame(geometry=[])

    if len(rivers_roi) == 0:
        river_score = np.zeros_like(dem)
    else:
        # Vectorized distance computation: much faster than per-cell Python loops.
        river_union = rivers_roi.geometry.union_all()
        grid_points = gpd.GeoSeries(
            gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
            crs=rivers_roi.crs,
        )
        river_dist = grid_points.distance(river_union).to_numpy().reshape(rows, cols)
        river_score = 1.0 - normalize(river_dist)

    rain_pattern = 0.6 * flow_score + 0.4 * river_score
    rainfall_factor = 1.58
    rain_effect = np.power(np.clip(rain_pattern * rainfall_factor, 0, 1), 0.85)

    risk = (
        (
            0.25 * slope_score
            + 0.25 * flow_score
            + 0.10 * np.clip(elev_score, 0, 1)
            + 0.20 * river_score
        )
        * 0.50
        + 0.50 * rain_effect
    )
    return np.clip(risk, 0, 1), bounds, rivers_roi


def build_map(risk: np.ndarray, bounds: rasterio.coords.BoundingBox, rivers_roi: gpd.GeoDataFrame) -> folium.Map:
    center = [float((bounds.bottom + bounds.top) / 2.0), float((bounds.left + bounds.right) / 2.0)]

    m = folium.Map(
        location=center,
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=False,
        zoom_control=False,
        attribution_control=False,
    )

    rgba = (RISK_HEATMAP(risk) * 255).astype(np.uint8)
    risk_overlay = folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.68,
        name="Flood risk",
        mercator_project=False,
        interactive=False,
    )
    risk_overlay.add_to(m)

    rivers_layer = None
    if len(rivers_roi) > 0:
        rivers_layer = folium.GeoJson(
            rivers_roi.to_json(),
            name="Rivers",
            style_function=lambda _: {"color": "#1864ab", "weight": 0.9, "opacity": 0.65},
        )
        rivers_layer.add_to(m)

    m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])

    map_name = m.get_name()
    risk_name = risk_overlay.get_name()
    rivers_name = rivers_layer.get_name() if rivers_layer is not None else None

    toggle_script = f"""
<style>
.map-toggles {{
    position: fixed;
    top: 120px;
    right: 12px;
    z-index: 10001;
    background: rgba(15, 23, 42, 0.9);
    color: #e9ecef;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 10px;
    padding: 8px 10px;
    font-family: Arial, sans-serif;
    font-size: 12px;
    backdrop-filter: blur(2px);
}}
.map-toggles label {{
    display: block;
    margin: 4px 0;
    cursor: pointer;
}}
.map-toggles .toggle-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 6px;
    font-weight: 600;
}}
.map-toggles .collapse-btn {{
    background: rgba(255,255,255,0.12);
    color: #f8f9fa;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    cursor: pointer;
}}
.map-toggles.collapsed .toggle-body {{
    display: none;
}}
</style>
<div class=\"map-toggles\">
    <div class=\"toggle-header\">
        <span>Map Layers</span>
        <button type=\"button\" id=\"toggle-panel\" class=\"collapse-btn\" aria-expanded=\"true\">Hide</button>
    </div>
    <div class=\"toggle-body\">
        <label><input type=\"checkbox\" id=\"toggle-risk\" checked> Show risk colors</label>
        <label><input type=\"checkbox\" id=\"toggle-rivers\" checked> Show rivers</label>
    </div>
</div>
<script>
(function() {{
    var panel = document.querySelector('.map-toggles');
    var panelBtn = document.getElementById('toggle-panel');
    var mapRef = {map_name};
    var riskLayer = {risk_name};
    var riversLayer = {rivers_name if rivers_name is not None else 'null'};

    var riskToggle = document.getElementById('toggle-risk');
    var riverToggle = document.getElementById('toggle-rivers');

    panelBtn.addEventListener('click', function() {{
        var isCollapsed = panel.classList.toggle('collapsed');
        panelBtn.textContent = isCollapsed ? 'Show' : 'Hide';
        panelBtn.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
    }});

    riskToggle.addEventListener('change', function() {{
        console.log('Risk toggle:', this.checked, 'Layer:', riskLayer);
        if (this.checked) {{
            mapRef.addLayer(riskLayer);
        }} else {{
            mapRef.removeLayer(riskLayer);
        }}
    }});

    riverToggle.addEventListener('change', function() {{
        console.log('River toggle:', this.checked, 'Layer:', riversLayer);
        if (!riversLayer) return;
        if (this.checked) {{
            mapRef.addLayer(riversLayer);
        }} else {{
            mapRef.removeLayer(riversLayer);
        }}
    }});
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(toggle_script))

    m.get_root().html.add_child(
        folium.Element(
            """
<style>
.risk-legend {
  position: fixed;
    left: 50%;
    top: 22px;
  transform: translateX(-50%);
  z-index: 9999;
    background: rgba(15, 23, 42, 0.92);
    color: #f1f3f5;
    padding: 8px 12px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.28);
  font-family: Arial, sans-serif;
}
.risk-legend .caption {
  font-size: 12px;
  margin-bottom: 6px;
    text-align: center;
}
.risk-legend .bar {
  width: 260px;
  height: 14px;
  border-radius: 4px;
    border: 1px solid rgba(255,255,255,0.5);
  background: linear-gradient(to right, #2ca25f 0%, #fee08b 50%, #d73027 100%);
}
.risk-legend .ticks {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  margin-top: 3px;
}
</style>
<div class=\"risk-legend\">
  <div class=\"caption\">Relative flood risk (0.0 to 1.0)</div>
  <div class=\"bar\"></div>
  <div class=\"ticks\"><span>0.0</span><span>0.5</span><span>1.0</span></div>
</div>
"""
        )
    )

    return m


def main() -> None:
    risk, bounds, rivers_roi = compute_country_risk()
    m = build_map(risk, bounds, rivers_roi)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_PATH))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

