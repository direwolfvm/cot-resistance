"""FastAPI app: chat API + static web UI.

Run:  uvicorn server.main:app --reload
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config_setup, pipeline
from .model import load_backend
from .transcript import Transcript

app = FastAPI(title="cot-resistance", description="Authenticated role tags PoC")

# Opt-in local key-setup page (ENABLE_KEY_SETUP=1). Registered before the
# static mount so its routes take precedence.
config_setup.register(app)

SESSIONS: dict[str, Transcript] = {}
BACKEND = load_backend()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    defense_on: bool = True


@app.post("/api/session")
def new_session() -> dict:
    session_id = secrets.token_urlsafe(16)
    SESSIONS[session_id] = Transcript()
    return {"session_id": session_id, "backend": BACKEND.name}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    transcript = SESSIONS.get(req.session_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="unknown session")
    trace = pipeline.handle_message(transcript, BACKEND, req.message, req.defense_on)
    return {"trace": trace, "transcript": pipeline.transcript_view(transcript)}


@app.get("/api/transcript/{session_id}")
def get_transcript(session_id: str) -> dict:
    transcript = SESSIONS.get(session_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"transcript": pipeline.transcript_view(transcript)}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="web")
