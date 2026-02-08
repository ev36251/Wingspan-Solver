"""FastAPI entry point for the Wingspan Solver API."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import EXCEL_FILE
from backend.data.registries import load_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data registries on startup."""
    load_all(EXCEL_FILE)
    yield


app = FastAPI(
    title="Wingspan Solver API",
    version="0.1.0",
    description="API for the Wingspan board game solver",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from backend.api.routes_data import router as data_router
from backend.api.routes_game import router as game_router
from backend.api.routes_solver import router as solver_router
from backend.api.routes_setup import router as setup_router

app.include_router(data_router, prefix="/api", tags=["data"])
app.include_router(game_router, prefix="/api/games", tags=["games"])
app.include_router(solver_router, prefix="/api/games", tags=["solver"])
app.include_router(setup_router, prefix="/api", tags=["setup"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
