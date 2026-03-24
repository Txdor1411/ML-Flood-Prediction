# FloodGuard API Setup

This FastAPI service powers:
- Location search (Nominatim proxy)
- Flood risk lookup per selected location
- Location-specific evacuation route cards
- Situation summaries using OpenAI GPT-4.1-mini
- Chat proxy using OpenAI GPT-4.1-mini

## 1) Install dependencies

```powershell
c:/Users/tudor/PycharmProjects/ML_Flood_Prediction/.venv/Scripts/python.exe -m pip install -r backend/requirements-api.txt
```

## 2) Add environment values

Copy backend/.env.example to backend/.env and set:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Only OPENAI_API_KEY is required for AI endpoints.

## 3) Run the API

```powershell
c:/Users/tudor/PycharmProjects/ML_Flood_Prediction/.venv/Scripts/python.exe -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000 --reload
```

## 4) Check health

Open:
- http://localhost:8000/api/health
- http://localhost:8000/docs

## API endpoints

- POST /api/geocode/search
- POST /api/risk/point
- POST /api/routes/cards
- POST /api/situation/summary
- POST /api/chat
