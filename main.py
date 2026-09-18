"""
FastAPI backend for the Eco Lifestyle Agent.

Endpoints:
  POST /api/chat          — Send a message, get an answer
  POST /api/reset         — Clear conversation history
  GET  /api/health        — Health check
  GET  /api/topics        — List knowledge base topics
  GET  /                  — Serve the chat UI (index.html)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from agent import get_agent

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Eco Lifestyle Agent",
    description="RAG-powered AI assistant for sustainable living guidance",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of context chunks to retrieve")


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    context_chunks: int
    response_time_ms: int


class ResetResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_model: str
    vector_store_path: str


class TopicItem(BaseModel):
    filename: str
    topic: str


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-warm the RAG pipeline and agent on startup."""
    print("[API] Warming up Eco Lifestyle Agent…")
    get_agent()
    print("[API] Agent ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    return HealthResponse(
        status="ok",
        llm_provider=config.LLM_PROVIDER,
        embedding_model=config.EMBEDDING_MODEL,
        vector_store_path=config.VECTOR_STORE_PATH,
    )


@app.get("/api/topics", response_model=List[TopicItem], tags=["knowledge"])
async def list_topics():
    """Return the list of topics available in the knowledge base."""
    kb_path = Path("data/kb")
    topics = []
    topic_map = {
        "01_reduce_plastic.txt": "Reducing Plastic Use",
        "02_energy_saving.txt": "Home Energy Saving & Renewables",
        "03_sustainable_travel.txt": "Sustainable Travel & Transport",
        "04_recycling_waste.txt": "Recycling, Waste & Composting",
        "05_sustainable_food.txt": "Sustainable Diet & Food Choices",
        "06_eco_products.txt": "Eco-Friendly Products & Shopping",
        "07_water_conservation.txt": "Water Conservation",
        "08_government_schemes.txt": "Government Schemes & Grants",
        "09_biodiversity_garden.txt": "Biodiversity & Wildlife Gardening",
        "10_carbon_footprint.txt": "Personal Carbon Footprint",
    }
    for filename, topic in topic_map.items():
        if (kb_path / filename).exists():
            topics.append(TopicItem(filename=filename, topic=topic))
    return topics


@app.post("/api/chat", response_model=ChatResponse, tags=["agent"])
async def chat(request: ChatRequest):
    """Send a message to the Eco Lifestyle Agent and receive an answer."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    agent = get_agent()
    start = time.perf_counter()
    try:
        result = agent.chat(request.message.strip(), top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        context_chunks=result["context_chunks"],
        response_time_ms=elapsed_ms,
    )


@app.post("/api/reset", response_model=ResetResponse, tags=["agent"])
async def reset_conversation():
    """Clear the agent's conversation history."""
    get_agent().reset_history()
    return ResetResponse(status="Conversation history cleared.")


# ── Static files / SPA fallback ───────────────────────────────────────────────
_STATIC = Path("frontend")

if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC / "static")), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index = _STATIC / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"message": "Eco Lifestyle Agent API is running. See /docs."})
else:
    @app.get("/", include_in_schema=False)
    async def serve_root():
        return JSONResponse({"message": "Eco Lifestyle Agent API is running. See /docs."})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
