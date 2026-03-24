from pathlib import Path

import folium
import geopandas as gpd
from matplotlib import colors as mcolors
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

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
        # Raster distance transform on the coarse grid is significantly faster.
        coarse_transform = transform * Affine.scale(downsample, downsample)
        river_mask = rasterize(
            [(geom, 1) for geom in rivers_roi.geometry if geom is not None and not geom.is_empty],
            out_shape=(rows, cols),
            transform=coarse_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
        river_dist = distance_transform_edt(river_mask == 0)
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
        min_zoom=6,
        max_zoom=18,
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

    # Add max bounds restriction script only
    bounds_script = f"""
<script>
(function() {{
    var mapRef = null;
    var riskRef = null;
    var riversRef = null;
    var tries = 0;
    var maxTries = 80;

    function setMapBounds() {{
        if (!mapRef) return;
        var maxBounds = L.latLngBounds(
            L.latLng({bounds.bottom}, {bounds.left}),
            L.latLng({bounds.top}, {bounds.right})
        );
        mapRef.setMaxBounds(maxBounds);
    }}

    function initWhenReady() {{
        mapRef = window['{map_name}'] || null;
        riskRef = window['{risk_name}'] || null;
        riversRef = window['{rivers_name}'] || null;

        if (mapRef) {{
            window.__floodguardMap = mapRef;
            window.__floodguardRiskLayer = riskRef;
            window.__floodguardRiversLayer = riversRef;
            setMapBounds();
            return;
        }}

        tries += 1;
        if (tries < maxTries) setTimeout(initWhenReady, 50);
    }}

    initWhenReady();
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(bounds_script))

    return m


def main() -> None:
    risk, bounds, rivers_roi = compute_country_risk()
    m = build_map(risk, bounds, rivers_roi)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_PATH))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
