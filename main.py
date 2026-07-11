# -*- coding: utf-8 -*-
"""Russian AI Assistant — FastAPI web server"""

import os
import uuid
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nlp_engine import SimpleNLP
from api_clients import AnthropicClient, OpenAIClient

app = FastAPI(title="Русский AI Помощник")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory state ───
nlp = SimpleNLP()
sessions: dict[str, dict] = {}
MAX_HISTORY = 50


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    api_mode: str = "offline"
    api_type: str = "anthropic"
    api_key: str = ""


class ChatResponse(BaseModel):
    reply: str
    session_id: str


def get_session(sid: str) -> dict:
    if sid not in sessions:
        sessions[sid] = {"history": []}
    return sessions[sid]


@app.get("/")
def index():
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    sid = req.session_id or uuid.uuid4().hex[:12]
    session = get_session(sid)

    # Add user message
    session["history"].append({"role": "user", "content": req.message})

    # Get response
    if req.api_mode == "online" and req.api_key:
        if req.api_type == "anthropic":
            client = AnthropicClient(req.api_key)
        else:
            client = OpenAIClient(req.api_key)
        reply = client.send_message(session["history"][-20:])
    else:
        time.sleep(0.3)
        reply = nlp.get_response(req.message) or "Извините, я не совсем понял."

    session["history"].append({"role": "assistant", "content": reply})

    # Trim history
    if len(session["history"]) > MAX_HISTORY:
        session["history"] = session["history"][-MAX_HISTORY:]

    return ChatResponse(reply=reply, session_id=sid)


@app.post("/api/clear")
def clear_session(data: dict):
    sid = data.get("session_id", "")
    if sid in sessions:
        del sessions[sid]
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"status": "ok"}
