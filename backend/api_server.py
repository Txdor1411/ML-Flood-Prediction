from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import geopandas as gpd
import httpx
import numpy as np
import rasterio
from affine import Affine
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parent
DEM_PATH = ROOT / "romania_dem.tif"
RIVERS_PATH = ROOT / "HydroRIVERS_romania.shp"
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ROMANIA_BOUNDS = {
    "min_lat": 43.5,
    "max_lat": 48.4,
    "min_lon": 20.0,
    "max_lon": 30.1,
}
SAFE_HUBS = [
    {"name": "Cluj-Napoca Emergency Hub", "lat": 46.7712, "lon": 23.6236},
    {"name": "Brasov Civic Shelter", "lat": 45.6579, "lon": 25.6012},
    {"name": "Bucharest North Shelter", "lat": 44.4949, "lon": 26.0800},
    {"name": "Iasi Regional Safety Point", "lat": 47.1585, "lon": 27.6014},
    {"name": "Timisoara West Emergency Point", "lat": 45.7489, "lon": 21.2087},
    {"name": "Constanta Coastal Evac Center", "lat": 44.1598, "lon": 28.6348},
]


class GeocodeSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=128)
    limit: int = Field(default=6, ge=1, le=10)


class GeocodeResult(BaseModel):
    name: str
    display_name: str
    lat: float
    lon: float


class GeocodeSearchResponse(BaseModel):
    results: list[GeocodeResult]


class LocationInput(BaseModel):
    lat: float
    lon: float


class RiskPointRequest(LocationInput):
    rainfall_pct: int = Field(default=70, ge=0, le=200)


class RiskPointResponse(BaseModel):
    risk_score: float
    risk_band: Literal["low", "medium", "high", "extreme"]
    recommendation: str
    nearby_rivers: list[str]


class RouteCard(BaseModel):
    id: str
    title: str
    path: str
    status: Literal["Clear", "Caution", "Priority"]
    distance: str
    eta: str


class RoutesRequest(LocationInput):
    rainfall_pct: int = Field(default=70, ge=0, le=200)


class RoutesResponse(BaseModel):
    routes: list[RouteCard]


class SituationSummaryRequest(LocationInput):
    location_name: str | None = None
    rainfall_pct: int = Field(default=70, ge=0, le=200)


class SituationSummaryResponse(BaseModel):
    location_name: str
    risk_score: float
    risk_band: Literal["low", "medium", "high", "extreme"]
    summary: str
    nearby_rivers: list[str]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=3000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=12)
    location_name: str | None = None


class ChatResponse(BaseModel):
    text: str


app = FastAPI(title="FloodGuard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=4)
def _openai_client() -> OpenAI | None:
    if not OPENAI_KEY:
        return None
    return OpenAI(api_key=OPENAI_KEY)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=np.nanmean(arr))
    span = float(arr.max() - arr.min())
    if span <= 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (span + 1e-6)


def _in_romania(lat: float, lon: float) -> bool:
    return (
        ROMANIA_BOUNDS["min_lat"] <= lat <= ROMANIA_BOUNDS["max_lat"]
        and ROMANIA_BOUNDS["min_lon"] <= lon <= ROMANIA_BOUNDS["max_lon"]
    )


@lru_cache(maxsize=40)
def _prepare_features(center_lat: float, center_lon: float, side_km: float, downsample: int) -> dict:
    if not DEM_PATH.exists():
        raise FileNotFoundError(f"Missing DEM raster: {DEM_PATH}")

    with rasterio.open(DEM_PATH) as src:
        full_bounds = src.bounds

        half_side_deg_lat = (side_km / 2.0) / 111.0
        cos_lat = max(np.cos(np.radians(center_lat)), 1e-6)
        half_side_deg_lon = (side_km / 2.0) / (111.0 * cos_lat)

        left = max(center_lon - half_side_deg_lon, full_bounds.left)
        right = min(center_lon + half_side_deg_lon, full_bounds.right)
        bottom = max(center_lat - half_side_deg_lat, full_bounds.bottom)
        top = min(center_lat + half_side_deg_lat, full_bounds.top)

        if left >= right or bottom >= top:
            raise ValueError("Selected area is outside DEM bounds")

        window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
        dem_full = src.read(1, window=window)
        transform = src.window_transform(window)
        bounds = rasterio.coords.BoundingBox(left=left, bottom=bottom, right=right, top=top)

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

    elev_score = 1.0 - _normalize(dem)
    slope_score = 1.0 - _normalize(slope)
    flow_score = _normalize(flow)

    if RIVERS_PATH.exists():
        rivers = gpd.read_file(RIVERS_PATH)
        if "ORD_FLOW" in rivers.columns:
            rivers = rivers[rivers["ORD_FLOW"] >= 3]
        rivers_roi = rivers.cx[bounds.left : bounds.right, bounds.bottom : bounds.top]
    else:
        rivers_roi = gpd.GeoDataFrame(geometry=[])

    if len(rivers_roi) == 0:
        river_score = np.zeros_like(dem)
    else:
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
        river_score = 1.0 - _normalize(river_dist)

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


def _compute_risk(features: dict, rainfall_pct: int) -> np.ndarray:
    rainfall_factor = rainfall_pct / 100.0
    rain_effect = np.power(np.clip(features["rain_pattern"] * rainfall_factor, 0, 1), 0.85)
    risk = (
        (
            0.25 * features["slope_score"]
            + 0.25 * features["flow_score"]
            + 0.10 * np.clip(features["elev_score"], 0, 1)
            + 0.20 * features["river_score"]
        )
        * 0.50
        + 0.50 * rain_effect
    )
    return np.clip(risk, 0, 1)


def _risk_band(score: float) -> Literal["low", "medium", "high", "extreme"]:
    if score < 0.35:
        return "low"
    if score < 0.55:
        return "medium"
    if score < 0.75:
        return "high"
    return "extreme"


def _risk_recommendation(band: str) -> str:
    if band == "low":
        return "Conditions are currently stable. Keep monitoring local weather updates."
    if band == "medium":
        return "Stay prepared, avoid low underpasses, and monitor nearby water levels."
    if band == "high":
        return "Prepare to move to higher ground and keep an evacuation kit ready."
    return "Move away from flood-prone zones now and follow local authority instructions."


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _nearest_river_names(rivers: gpd.GeoDataFrame, lat: float, lon: float, limit: int = 3) -> list[str]:
    if rivers is None or len(rivers) == 0:
        return []

    point = gpd.GeoSeries.from_xy([lon], [lat], crs="EPSG:4326").iloc[0]
    distances = rivers.geometry.distance(point)
    nearest = rivers.assign(_d=distances).sort_values("_d").head(limit)

    names: list[str] = []
    for _, row in nearest.iterrows():
        for candidate in ("NAME", "RIVER", "HYRIV_ID"):
            value = row.get(candidate)
            if value is not None and str(value).strip() and str(value) != "nan":
                names.append(str(value))
                break
    return names


def _risk_at_location(lat: float, lon: float, rainfall_pct: int) -> tuple[float, str, list[str]]:
    features = _prepare_features(round(lat, 2), round(lon, 2), side_km=120.0, downsample=220)
    risk_grid = _compute_risk(features, rainfall_pct)

    lat_idx = int(np.abs(features["lat_grid"][:, 0] - lat).argmin())
    lon_idx = int(np.abs(features["lon_grid"][0, :] - lon).argmin())
    score = float(risk_grid[lat_idx, lon_idx])
    band = _risk_band(score)
    nearby = _nearest_river_names(features["rivers_roi"], lat, lon)
    return score, band, nearby


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "openai_configured": bool(OPENAI_KEY)}


@app.post("/api/geocode/search", response_model=GeocodeSearchResponse)
async def geocode_search(payload: GeocodeSearchRequest) -> GeocodeSearchResponse:
    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "format": "jsonv2",
                "q": payload.query,
                "countrycodes": "ro",
                "addressdetails": "1",
                "limit": payload.limit,
            },
            headers={"User-Agent": "FloodGuardAPI/1.0"},
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Geocoding provider unavailable")

    raw = response.json()
    results: list[GeocodeResult] = []
    for item in raw:
        lat = float(item.get("lat", 0.0))
        lon = float(item.get("lon", 0.0))
        if not _in_romania(lat, lon):
            continue
        display_name = str(item.get("display_name", "Unknown place"))
        short_name = display_name.split(",")[0].strip() or "Unknown place"
        results.append(
            GeocodeResult(name=short_name, display_name=display_name, lat=lat, lon=lon)
        )
    return GeocodeSearchResponse(results=results)


@app.post("/api/risk/point", response_model=RiskPointResponse)
def risk_point(payload: RiskPointRequest) -> RiskPointResponse:
    if not _in_romania(payload.lat, payload.lon):
        raise HTTPException(status_code=400, detail="Location must be inside Romania bounds")

    try:
        score, band, nearby = _risk_at_location(payload.lat, payload.lon, payload.rainfall_pct)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Risk computation failed: {exc}") from exc

    return RiskPointResponse(
        risk_score=round(score, 4),
        risk_band=band,
        recommendation=_risk_recommendation(band),
        nearby_rivers=nearby,
    )


@app.post("/api/routes/cards", response_model=RoutesResponse)
def route_cards(payload: RoutesRequest) -> RoutesResponse:
    if not _in_romania(payload.lat, payload.lon):
        raise HTTPException(status_code=400, detail="Location must be inside Romania bounds")

    score, band, _ = _risk_at_location(payload.lat, payload.lon, payload.rainfall_pct)

    sorted_hubs = sorted(
        SAFE_HUBS,
        key=lambda h: _haversine_km(payload.lat, payload.lon, h["lat"], h["lon"]),
    )[:3]

    status = "Clear"
    if band in {"high", "extreme"}:
        status = "Priority"
    elif band == "medium":
        status = "Caution"

    routes: list[RouteCard] = []
    for idx, hub in enumerate(sorted_hubs, start=1):
        distance_km = _haversine_km(payload.lat, payload.lon, hub["lat"], hub["lon"])
        eta_min = max(8, int(round((distance_km / 55.0) * 60)))
        routes.append(
            RouteCard(
                id=str(idx),
                title=f"Route {idx}",
                path=f"Selected place -> {hub['name']}",
                status=status if idx == 1 else ("Caution" if status == "Priority" else "Clear"),
                distance=f"{distance_km:.1f} km",
                eta=f"Est. {eta_min} min",
            )
        )

    return RoutesResponse(routes=routes)


def _build_situation_prompt(location_name: str, risk_score: float, risk_band: str, nearby_rivers: list[str]) -> str:
    rivers = ", ".join(nearby_rivers) if nearby_rivers else "no major mapped river nearby"
    return (
        "You are FloodGuard AI. Provide a concise flood situation summary in plain language. "
        "Respond in 2-4 short sentences. Include immediate actions if risk is high. "
        f"Location: {location_name}. Risk score: {risk_score:.2f}. Risk band: {risk_band}. Nearby rivers: {rivers}."
    )


@app.post("/api/situation/summary", response_model=SituationSummaryResponse)
def situation_summary(payload: SituationSummaryRequest) -> SituationSummaryResponse:
    if not _in_romania(payload.lat, payload.lon):
        raise HTTPException(status_code=400, detail="Location must be inside Romania bounds")

    score, band, nearby = _risk_at_location(payload.lat, payload.lon, payload.rainfall_pct)
    location_name = payload.location_name or f"{payload.lat:.4f}, {payload.lon:.4f}"

    default_summary = (
        f"Flood risk is currently {band} for {location_name}. "
        f"Risk score is {score:.2f}. {_risk_recommendation(band)}"
    )

    client = _openai_client()
    if client is None:
        return SituationSummaryResponse(
            location_name=location_name,
            risk_score=round(score, 4),
            risk_band=band,
            summary=default_summary,
            nearby_rivers=nearby,
        )

    prompt = _build_situation_prompt(location_name, score, band, nearby)
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You are a flood safety assistant. Be concise and actionable.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        ai_text = completion.choices[0].message.content or default_summary
    except Exception:  # noqa: BLE001
        ai_text = default_summary

    return SituationSummaryResponse(
        location_name=location_name,
        risk_score=round(score, 4),
        risk_band=band,
        summary=ai_text,
        nearby_rivers=nearby,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    client = _openai_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on backend. Add it to backend/.env.",
        )

    system_prompt = (
        "You are FloodGuard AI. Give concise, practical flood safety guidance and evacuation advice. "
        "Do not invent emergency numbers. If uncertain, say what is unknown."
    )
    if payload.location_name:
        system_prompt += f" Current selected location: {payload.location_name}."

    messages = [{"role": "system", "content": system_prompt}]
    for message in payload.messages:
        if message.role == "system":
            continue
        messages.append({"role": message.role, "content": message.content})

    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0.2,
            messages=messages,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc

    text = completion.choices[0].message.content or ""
    if not text.strip():
        text = "I could not generate a response. Please try again."

    return ChatResponse(text=text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
