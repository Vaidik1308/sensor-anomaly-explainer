"""
main.py — FastAPI Application Entry Point

Sensor Anomaly Explanation Generator for Marine & Oil Industry IoT Platform Operators.
This is the main entry point that configures CORS, loads the RAG knowledge base,
and registers all API routes.

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes import router
from rag_engine import load_vector_store

# ── Load environment variables from .env file ───────────────────────────────
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ── Application lifespan: load resources on startup ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS vector store when the server starts."""
    print("[Startup] Loading marine/oil anomaly knowledge base...")
    try:
        load_vector_store()
        print("[Startup] Knowledge base loaded successfully.")
    except FileNotFoundError as e:
        print(f"[Startup] WARNING: {e}")
        print("[Startup] The API will start, but /explain will fail until the index is built.")
    yield
    print("[Shutdown] Sensor Anomaly Explainer shutting down.")


# ── Create FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="Sensor Anomaly Explanation Generator",
    description=(
        "RAG-powered AI that explains marine and oil industry IoT sensor anomalies "
        "in plain language for frontline vessel and platform operators."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware — allow frontend to call the API ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ─────────────────────────────────────────────────────────
app.include_router(router)

# ── Serve frontend static files ─────────────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
