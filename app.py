import os
import json
import requests
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"],
    storage_uri="memory://"
)

HF_API_KEY = os.environ.get("HF_API_KEY", "")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

SYSTEM_PROMPT = """You are ARIA (Advanced Reasoning Intelligence Assistant), a sharp, 
precise AI assistant. Respond in a helpful, structured way. Use markdown formatting 
with headers and bullet points where appropriate. Keep responses focused and actionable.
Never say you cannot help — always provide useful information."""


def query_hf(task):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task}
        ],
        "max_tokens": 512
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def get_fallback_response(task: str) -> str:
    task_lower = task.lower()
    if "study" in task_lower:
        return """## Study Plan\n\n**Week 1–2: Foundation**\n- Set clear goals and deadlines\n- Use active recall (flashcards, practice problems)\n- Study in 25-min Pomodoro blocks\n\n**Week 3–4: Deep Work**\n- Past papers and mock tests\n- Teach concepts out loud (Feynman technique)\n- Review weak areas daily\n\n**Tip:** Sleep 7–8 hours — memory consolidates during sleep."""
    elif "interview" in task_lower:
        return """## Interview Prep Guide\n\n**Technical Round**\n- Practice DSA on LeetCode (Easy → Medium)\n- Study system design basics\n- Review your projects deeply\n\n**HR Round**\n- Prepare STAR-format stories\n- Research the company thoroughly\n- Prepare 5 smart questions to ask\n\n**Day Before:** Mock interview with a friend, good sleep."""
    elif "python" in task_lower:
        return """## Python Learning Roadmap\n\n**Beginner (0–4 weeks)**\n- Syntax, loops, functions, OOP\n- Practice: HackerRank, Codecademy\n\n**Intermediate (1–3 months)**\n- File I/O, APIs, Flask, pandas\n- Build: To-do app, weather app\n\n**Advanced (3–6 months)**\n- Async, decorators, design patterns\n- Build: Full-stack project, scraper, ML model"""
    else:
        return """## Here to help!\n\nI can assist you with a wide range of tasks.\n\n**Learning & Study**\n- Create a study plan for any subject\n- Explain complex topics simply\n\n**Career & Jobs**\n- Interview preparation tips\n- Resume writing advice\n\n**Programming**\n- Python, JavaScript, Web Dev roadmaps\n- Code explanations"""


@app.route("/")
def index():
    api_status = "connected" if HF_API_KEY else "demo"
    return render_template("index.html", api_status=api_status)


@app.route("/run", methods=["POST"])
@limiter.limit("15 per minute")
def run_ai():
    data = request.get_json(silent=True) or {}
    task = data.get("task", "").strip()

    if not task:
        return jsonify({"error": "Please enter a task or question."}), 400

    if len(task) > 2000:
        return jsonify({"error": "Input too long. Keep it under 2000 characters."}), 400

    try:
        if HF_API_KEY:
            result = query_hf(task)
        else:
            result = get_fallback_response(task)

        return jsonify({"result": result, "status": "success"})

    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500


@app.route("/stream", methods=["POST"])
@limiter.limit("10 per minute")
def stream_ai():
    data = request.get_json(silent=True) or {}
    task = data.get("task", "").strip()

    if not task:
        return jsonify({"error": "Empty input"}), 400

    def generate():
        try:
            if HF_API_KEY:
                result = query_hf(task)
                words = result.split(" ")
                import time
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    time.sleep(0.03)
            else:
                import time
                text = get_fallback_response(task)
                words = text.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    time.sleep(0.04)
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/chat", methods=["POST"])
@limiter.limit("20 per minute")
def chat_ai():
    data = request.get_json(silent=True) or {}
    task = data.get("task", "").strip()

    if not task:
        return jsonify({"error": "Empty input"}), 400

    try:
        if HF_API_KEY:
            result = query_hf(task)
        else:
            result = get_fallback_response(task)

        return jsonify({"result": result, "status": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "Mistral-7B" if HF_API_KEY else "demo",
        "version": "2.0.0"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)