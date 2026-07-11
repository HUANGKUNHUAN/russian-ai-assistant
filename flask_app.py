# -*- coding: utf-8 -*-
"""Russian AI Assistant — Flask web app for PythonAnywhere"""

import os
import json
import time
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory

from nlp_engine import SimpleNLP
from api_clients import AnthropicClient, OpenAIClient

app = Flask(__name__)
nlp = SimpleNLP()

# ─── In-memory chat sessions ───
sessions: dict[str, list] = {}
MAX_HISTORY = 50

# ─── Serve frontend ───
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ─── Chat API ───
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "Message is required"}), 400

    message = data["message"].strip()
    sid = data.get("session_id", uuid.uuid4().hex[:12])
    api_mode = data.get("api_mode", "offline")
    api_type = data.get("api_type", "anthropic")
    api_key = data.get("api_key", "")

    if sid not in sessions:
        sessions[sid] = []
    history = sessions[sid]

    # Add user message
    history.append({"role": "user", "content": message})

    # Get response
    if api_mode == "online" and api_key:
        if api_type == "anthropic":
            client = AnthropicClient(api_key)
        else:
            client = OpenAIClient(api_key)
        reply = client.send_message(history[-20:])
    else:
        time.sleep(0.3)
        reply = nlp.get_response(message) or "Извините, я не совсем понял ваш вопрос."

    history.append({"role": "assistant", "content": reply})

    # Trim history
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    return jsonify({"reply": reply, "session_id": sid})


# ─── Clear session ───
@app.route("/api/clear", methods=["POST"])
def clear():
    sid = request.get_json().get("session_id", "")
    sessions.pop(sid, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
