from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .core.database import create_tables
from .core.config import get_settings
from .routers import auth, documents, study

settings = get_settings()

app = FastAPI(
    title="Learnly AI",
    description="AI-powered study tool API",
    version="1.0.0",
)

# ── CORS (adjust origins for production) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(study.router)

# ── Serve frontend static files (production) ─────────────────────────────────
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


@app.on_event("startup")
def startup():
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    create_tables()


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.APP_NAME}
