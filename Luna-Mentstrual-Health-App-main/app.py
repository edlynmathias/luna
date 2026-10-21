
import os, json, re
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from rag import KnowledgeBase, answer_question

load_dotenv(override=True)  # always use the latest .env values (Streamlit reruns keep old env vars)
app = Flask(__name__, static_folder="static")
kb = KnowledgeBase()
KB_STATUS = kb.ingest(Path("knowledge/sources.json"))

@app.after_request
def allow_local_cors(resp):
    # Lets the page work if index.html is opened directly (file://) or from another local port.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/api/chat", methods=["OPTIONS"])
def chat_preflight():
    return "", 204

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/status")
def status():
    return jsonify({"ok": True, "knowledge": KB_STATUS, "llm_configured": bool(os.getenv("LLM_API_KEY"))})

@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    question = str(body.get("message", "")).strip()
    if not question or len(question) > 1200:
        return jsonify({"error": "Please enter a question under 1,200 characters."}), 400
    try:
        history = body.get("history") if isinstance(body.get("history"), list) else []
        result = answer_question(question, kb, history=history[-12:])
        return jsonify(result)
    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify({"error": "Luna couldn't complete that answer. Please try again, or consult a qualified health professional."}), 502

if __name__ == "__main__":
    print(f"Luna running: http://127.0.0.1:{os.getenv('PORT', '5000')}  ({KB_STATUS['mode']})")
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
