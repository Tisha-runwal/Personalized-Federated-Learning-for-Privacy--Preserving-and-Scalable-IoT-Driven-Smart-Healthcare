"""FastAPI application entry-point for PFL-HCare dashboard backend."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.db import init_db
from server.routes.training import router as training_router
from server.routes.metrics import router as metrics_router
from server.routes.datasets import router as datasets_router
from server.ws.live import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database on startup."""
    await init_db()
    yield


app = FastAPI(
    title="PFL-HCare API",
    description="Backend for the Personalized Federated Learning Healthcare dashboard.",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev-server and any CRA dev-server
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(metrics_router, prefix="/api/metrics", tags=["metrics"])
app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])
app.include_router(ws_router, tags=["websocket"])


@app.get("/", tags=["health"])
async def root() -> dict:
    return {"status": "ok", "service": "pfl-hcare"}
