"""
==========================================================
  ZENITH OX - Secure Intelligent Research Assistant
  Stage 2: AI Mode Selection (6 modes)
  + Syntax-highlighted code + downloadable code files
  
  UPDATED: Groq -> Gemini 2.5 Flash (no paid API key needed)
==========================================================
"""

import os
import json
import secrets
import re
import time
import zipfile
from datetime import datetime

import requests
import numpy as np

from openai import OpenAI
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, flash, send_from_directory,
)
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# 1. ENVIRONMENT
# ----------------------------------------------------------
load_dotenv()

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_FILE  = os.path.join(BASE_DIR, "users.json")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
WRITABLE_USERS  = "/tmp/users.json"
WRITABLE_MEMORY = "/tmp/memory.json"

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")
SECRET_KEY          = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

gemini_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ----------------------------------------------------------
# 2. FLASK APP
# ----------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = SECRET_KEY
MEMORY_LIMIT = 15
TOP_K_MEMORY = 3

# ----------------------------------------------------------
# 2b. GOOGLE OAUTH
# ----------------------------------------------------------
oauth = OAuth(app)
google = None
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    google = oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def google_enabled():
    return google is not None


app.jinja_env.globals["google_enabled"] = google_enabled


# ==========================================================
# 3. AI MODES
# ==========================================================
AI_MODES = {
    "developer": {
        "name": "AI Developer",
        "emoji": "\U0001f4bb",
        "tagline": "Code generation, debugging, and explanation",
        "model": "gemini-2.5-flash",
        "system_prompt": (
            "You are an expert software developer and programming assistant. "
            "Write clean, well-documented code in any language requested. "
            "Debug and fix code issues with clear explanations. "
            "Explain complex programming concepts simply. "
            "Follow best practices and design patterns. "
            "Always provide working code examples with proper formatting. "
            "IMPORTANT: Always wrap code in markdown code fences with the language name, "
            "for example: ```python\\n...code...\\n``` or ```javascript\\n...code...\\n```. "
            "Never put code outside of fences. Separate explanations from code clearly."
        ),
        "temperature": 0.3,
        "max_tokens": 2000,
        "uses_web_search": False,
    },
    "story_writer": {
        "name": "AI Story Writer",
        "emoji": "\U0001f4d6",
        "tagline": "Creative writing, stories, poems, and scripts",
        "model": "gemini-2.5-flash",
        "system_prompt": (
            "You are a talented creative writer. "
            "Write engaging stories with vivid descriptions and compelling characters. "
            "Craft poetry with rhythm and imagery. "
            "Create scripts with authentic dialogue. "
            "Adapt your writing style to match the requested genre. "
            "Be creative, original, and evocative in your writing."
        ),
        "temperature": 0.85,
        "max_tokens": 2000,
        "uses_web_search": False,
    },
    "solve_it": {
        "name": "AI Solve It",
        "emoji": "\U0001f9ee",
        "tagline": "Math problems and step-by-step solutions",
        "model": "gemini-2.5-flash",
        "system_prompt": (
            "You are an expert mathematician and problem solver. "
            "Solve math problems step-by-step, showing all work clearly. "
            "Explain mathematical concepts with examples. "
            "Handle algebra, calculus, statistics, geometry, and more. "
            "Break complex problems into manageable numbered steps. "
            "Always verify your answers by checking the work."
        ),
        "temperature": 0.2,
        "max_tokens": 2000,
        "uses_web_search": False,
    },
    "researcher": {
        "name": "AI Researcher",
        "emoji": "\U0001f50d",
        "tagline": "Web search + memory research assistant",
        "model": "gemini-2.5-flash",
        "system_prompt": (
            "You are Zenith OX, a secure, intelligent research assistant. "
            "You answer clearly, accurately, and concisely. "
            "Use the provided past memory and web context when relevant, "
            "but never fabricate facts. If unsure, say so."
        ),
        "temperature": 0.6,
        "max_tokens": 800,
        "uses_web_search": True,
    },
    "email_writer": {
        "name": "AI Email Writer",
        "emoji": "\u2709\ufe0f",
        "tagline": "Generate professional emails ready to copy",
        "model": "gemini-2.5-flash",
        "system_prompt": (
            "You are an expert email writer. "
            "Write clear, professional, and well-structured emails. "
            "Adapt tone to context: formal, casual, follow-up, complaint, request, etc. "
            "Include appropriate Subject line, greeting, body, and sign-off. "
            "Keep emails concise yet complete. "
            "Format the output as a ready-to-copy email with Subject: and Body: clearly marked."
        ),
        "temperature": 0.5,
        "max_tokens": 1500,
        "uses_web_search": False,
    },
    "pptx_generator": {
        "name": "AI Slides Generator",
        "emoji": "\U0001f4ca",
        "tagline": "Generate downloadable PowerPoint presentations",
        "model": "gemini-2.5-flash",
        "system_prompt": (
            "You generate PowerPoint presentation content. "
            "When the user asks for a presentation, generate ONLY a valid JSON object with no extra text.\n"
            "Use this exact format:\n"
            '{"title": "Presentation Title", "slides": ['
            '{"title": "Slide Title", "bullets": ["Point 1", "Point 2", "Point 3"]}'
            "]}\n\n"
            "Rules:\n"
            "- Generate 3-10 slides based on the topic complexity\n"
            "- Each slide should have 3-5 concise bullet points\n"
            "- First slide should be an overview\n"
            "- Last slide should be a summary or conclusion\n"
            "- Output ONLY the JSON object, no markdown fences, no explanations"
        ),
        "temperature": 0.4,
        "max_tokens": 3000,
        "uses_web_search": False,
        "special_handler": "pptx",
    },
}

# Language to file extension mapping
LANG_EXTENSIONS = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "html": ".html", "css": ".css",
    "java": ".java", "c": ".c",
    "cpp": ".cpp", "c++": ".cpp", "csharp": ".cs",
    "go": ".go", "rust": ".rs", "ruby": ".rb",
    "php": ".php", "sql": ".sql", "swift": ".swift",
    "kotlin": ".kt", "dart": ".dart", "r": ".r",
    "bash": ".sh", "sh": ".sh", "shell": ".sh",
    "json": ".json", "yaml": ".yml", "yml": ".yml",
    "xml": ".xml", "markdown": ".md", "md": ".md",
    "txt": ".txt", "text": ".txt", "": ".txt",
}


# ==========================================================
# 4. STORAGE HELPERS
# ==========================================================
def load_users():
    for path in [WRITABLE_USERS, USERS_FILE]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return {}


def save_users(users):
    with open(WRITABLE_USERS, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def load_memory():
    for path in [WRITABLE_MEMORY, MEMORY_FILE]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return {}


def save_memory(memory):
    with open(WRITABLE_MEMORY, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def get_user_memory(memory_key):
    return load_memory().get(memory_key, [])


def update_user_memory(memory_key, role, content):
    mem = load_memory()
    mem.setdefault(memory_key, [])
    mem[memory_key].append({"role": role, "content": content})
    if len(mem[memory_key]) > MEMORY_LIMIT * 2:
        mem[memory_key] = mem[memory_key][-MEMORY_LIMIT * 2:]
    save_memory(mem)


# ==========================================================
# 4b. AUTH HELPERS
# ==========================================================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))


def find_user_by_email(email: str):
    email = (email or "").strip().lower()
    users = load_users()
    for key, u in users.items():
        if isinstance(u, dict) and u.get("email", "").lower() == email:
            return key, u
    return None, None


def find_user_by_google_id(gid: str):
    users = load_users()
    for key, u in users.items():
        if isinstance(u, dict) and u.get("google_id") == gid:
            return key, u
    return None, None


def display_name_from_email(email: str) -> str:
    return (email or "").split("@")[0] or "friend"


def time_based_greeting(name: str) -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        part = "Good morning"
    elif 12 <= hour < 17:
        part = "Good afternoon"
    elif 17 <= hour < 22:
        part = "Good evening"
    else:
        part = "Good night"
    return f"{part}, {name}"


# ==========================================================
# 5. VECTOR MEMORY
# ==========================================================
def retrieve_relevant_memory(memory_key, query, top_k=TOP_K_MEMORY):
    history = get_user_memory(memory_key)
    if not history:
        return ""
    pairs = []
    for i, msg in enumerate(history):
        if msg["role"] == "user":
            reply = ""
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                reply = history[i + 1]["content"]
            pairs.append((msg["content"], reply))
    if not pairs:
        return ""
    user_texts = [p[0] for p in pairs]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(user_texts + [query])
        query_vec = matrix[-1]
        past_vecs = matrix[:-1]
        sims = cosine_similarity(query_vec, past_vecs).flatten()
    except ValueError:
        return ""
    ranked = np.argsort(sims)[::-1]
    selected = [idx for idx in ranked if sims[idx] > 0][:top_k]
    if not selected:
        return ""
    lines = []
    for idx in selected:
        u, a = pairs[idx]
        lines.append(f"- User previously asked: {u}\n  You answered: {a}")
    return "\n".join(lines)


# ==========================================================
# 6. WEB SEARCH
# ==========================================================
def tavily_search(query, max_results=3):
    if not TAVILY_API_KEY:
        return ""
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query,
                  "search_depth": "basic", "max_results": max_results},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        chunks = [res.get("content", "") for res in data.get("results", [])]
        return "\n\n".join(filter(None, chunks))[:4000]
    except Exception as e:
        print("[Tavily error]", e)
        return ""


# ==========================================================
# 7. GEMINI CHAT (was: GROQ CHAT)
# ==========================================================
def ask_gemini(user_input, vector_memory, web_context, mode):
    prompt = f"User Question: {user_input}\n\n"
    if vector_memory:
        prompt += f"Relevant Past Memory:\n{vector_memory}\n\n"
    if web_context:
        prompt += f"Web Context:\n{web_context}\n\n"
    if not mode.get("special_handler"):
        prompt += "Instruction:\nProvide a clear, accurate, and helpful answer."
    try:
        resp = gemini_client.chat.completions.create(
            model=mode["model"],
            messages=[
                {"role": "system", "content": mode["system_prompt"]},
                {"role": "user",   "content": prompt},
            ],
            temperature=mode["temperature"],
            max_tokens=mode["max_tokens"],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Zenith OX error] {e}"


# ==========================================================
# 8. PPTX GENERATOR
# ==========================================================
def parse_slides_json(ai_response):
    text = ai_response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return None


def format_slides_as_text(data):
    title = data.get("title", "Presentation")
    slides = data.get("slides", [])
    if not slides:
        return None
    lines = [f"\U0001f4ca {title}\n"]
    for i, s in enumerate(slides, 1):
        lines.append(f"--- Slide {i}: {s.get('title', 'Untitled')} ---")
        for bullet in s.get("bullets", []):
            lines.append(f"  \u2022 {bullet}")
        lines.append("")
    return "\n".join(lines)


def generate_pptx(data):
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError as e:
        print(f"[PPTX import error] {e}")
        return None
    try:
        title = data.get("title", "Presentation")
        slides = data.get("slides", [])
        if not slides:
            return None
        prs = Presentation()
        title_layout = prs.slide_layouts[0]
        first = prs.slides.add_slide(title_layout)
        first.shapes.title.text = title
        if len(first.placeholders) > 1:
            first.placeholders[1].text = "Generated by Zenith OX"
        content_layout = prs.slide_layouts[1]
        slide_titles = []
        for s in slides:
            slide = prs.slides.add_slide(content_layout)
            slide.shapes.title.text = s.get("title", "Untitled")
            slide_titles.append(s.get("title", "Untitled"))
            body = slide.placeholders[1]
            tf = body.text_frame
            tf.clear()
            for j, bullet in enumerate(s.get("bullets", [])):
                if j == 0:
                    tf.text = bullet
                else:
                    p = tf.add_paragraph()
                    p.text = bullet
        filename = f"zenith_ox_{int(time.time())}.pptx"
        filepath = f"/tmp/{filename}"
        prs.save(filepath)
        return {"filename": filename, "url": f"/download/{filename}", "slides": slide_titles}
    except Exception as e:
        print(f"[PPTX creation error] {e}")
        return None


# ==========================================================
# 9. CODE FILE GENERATOR (Developer mode)
# ==========================================================
CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def extract_code_blocks(text):
    """Extract all fenced code blocks from markdown text."""
    matches = CODE_BLOCK_RE.findall(text)
    blocks = []
    for lang, code in matches:
        blocks.append({"language": (lang or "txt").lower(), "code": code.strip()})
    return blocks


def save_code_files(code_blocks):
    """Save code blocks as individual files + a zip archive. Returns download info."""
    if not code_blocks:
        return None
    timestamp = int(time.time())
    files_created = []

    for i, block in enumerate(code_blocks):
        lang = block["language"]
        ext = LANG_EXTENSIONS.get(lang, ".txt")
        if len(code_blocks) == 1:
            filename = f"code{ext}"
        else:
            filename = f"code_{i + 1}{ext}"
        filepath = f"/tmp/{filename}"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(block["code"])
        files_created.append({"filename": filename, "filepath": filepath})

    # Create zip
    zip_filename = f"zenith_code_{timestamp}.zip"
    zip_filepath = f"/tmp/{zip_filename}"
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        for fi in files_created:
            zf.write(fi["filepath"], fi["filename"])

    return {
        "files": [fi["filename"] for fi in files_created],
        "zip_filename": zip_filename,
        "zip_url": f"/download-zip/{zip_filename}",
    }


# ==========================================================
# 10. ROUTES
# ==========================================================

@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    if "ai_mode" not in session:
        return redirect(url_for("menu"))
    mode_key = session["ai_mode"]
    mode = AI_MODES.get(mode_key, AI_MODES["researcher"])
    username = session.get("display_name") or session["user_id"]
    greeting = time_based_greeting(username)
    return render_template("index.html", username=username, greeting=greeting,
                           mode=mode, mode_key=mode_key)


@app.route("/menu")
def menu():
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    username = session.get("display_name") or session["user_id"]
    greeting = time_based_greeting(username)
    return render_template("menu.html", username=username, greeting=greeting, modes=AI_MODES)


@app.route("/select-mode/<mode_key>")
def select_mode(mode_key):
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    if mode_key not in AI_MODES:
        flash("Invalid AI mode.", "error")
        return redirect(url_for("menu"))
    session["ai_mode"] = mode_key
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    email    = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm  = request.form.get("confirm") or ""
    name     = (request.form.get("name") or "").strip()
    if not valid_email(email):
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("register"))
    if len(password) < 6:
        flash("Password must be at least 6 characters.", "error")
        return redirect(url_for("register"))
    if password != confirm:
        flash("Passwords do not match.", "error")
        return redirect(url_for("register"))
    # NOTE: Your original code was cut off here.
    # Add the rest of your register route below.
    # ...


# ==========================================================
# IMPORTANT: Your original code was truncated at the register
# route. Paste the rest of your routes below this point.
# All changes above are complete - no other modifications needed.
#
# Remember: anywhere you previously called ask_groq(),
# change it to ask_gemini() with the same arguments.
# ==========================================================