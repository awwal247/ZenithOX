# 🧠 Zenith OX
### A Secure Intelligent Research Assistant with Vector Memory

Zenith OX is a Flask-based AI research assistant that combines
**email + password** and/or **Google OAuth** authentication,
**per-user vector memory** (TF-IDF + cosine similarity),
and **real-time web search** (Tavily) to power a ChatGPT-style UI.

---

## ✨ Features
| # | Feature | Tech |
|---|---|---|
| 🔐 | Email + password login (hashed with Werkzeug) | `werkzeug.security` |
| 🟦 | Google OAuth sign-in (optional) | `authlib` |
| 👋 | Time-aware greeting (morning / afternoon / evening / night) — computed in Python | `datetime` |
| 🧠 | Vector-based conversation memory  | `scikit-learn` TF-IDF + cosine similarity |
| 🌐 | Real-time web retrieval          | Tavily API (via raw `requests`) |
| 💬 | Intelligent chat responses        | Groq `llama-3.3-70b-versatile` (free) |
| 🗂 | Per-user persistent storage       | JSON files (`users.json`, `memory.json`) |
| 🎨 | Minimal-JS UI (forms are pure HTML, validation is pure Python) | Flask Jinja + flash messages |

---

## 📁 Project Structure
```
Zenith-OX/
├── api/
│   └── index.py          # Flask backend (all routes + logic, in Python)
├── templates/
│   ├── register.html     # Email + password + Google sign-up
│   ├── login.html        # Email + password + Google login
│   └── index.html        # Chat UI with greeting bar at the top
├── static/
│   ├── style.css         # Unified styles
│   └── script.js         # Minimal JS (chat only)
├── users.json            # Per-user accounts (email, hash, google_id)
├── memory.json           # Per-user chat history
├── .env.example
├── requirements.txt
├── vercel.json
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Required:
- `GROQ_API_KEY` — from <https://console.groq.com/>
- `TAVILY_API_KEY` — from <https://tavily.com/> (free tier works)
- `FLASK_SECRET_KEY` — any long random string

Optional (enables the "Continue with Google" button):
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`

> Get Google credentials at <https://console.cloud.google.com/apis/credentials>.
> Authorized redirect URI: `http://localhost:5000/auth/google/callback`
> (or your production domain).

### 3. Run
```bash
python api/index.py
```
The app listens on <http://localhost:5000>.

---

## 👤 User Journey

1. **Register** → `/register`
   - *Option A*: Type an **email + password** → account is created and you're signed in.
   - *Option B*: Click **Sign up with Google** (if configured).
   - *Option C*: You may later link Google to the same email — both methods work for the same account.
2. **Login** → `/login`, either email+password or Google.
3. **Chat** → the page opens with a greeting at the top (e.g. *"Good morning, Alice 👋"*), and Zenith OX will:
   1. Pull the **top 3 most relevant** past messages (TF-IDF + cosine).
   2. Fetch **live web context** from Tavily.
   3. Compose a prompt and call **Groq** (`llama3-8b-8192`).
   4. Stream the answer back with typing animation.
   5. Persist the conversation to `memory.json`.

---

## 🔌 API Endpoints

| Method | Route                     | Purpose |
|--------|---------------------------|---------|
| GET    | `/`                       | Chat UI + greeting (requires login) |
| GET/POST | `/register`             | Register with email + password |
| GET/POST | `/login`                | Login with email + password |
| GET    | `/login/google`           | Start Google OAuth flow |
| GET    | `/auth/google/callback`   | Google OAuth callback |
| POST   | `/chat`                   | Send a message, get AI reply |
| POST   | `/clear`                  | Wipe the current user's memory |
| GET/POST | `/logout`               | End the session |

---

## 🕒 How the greeting works
The first thing you see on the chat page is a top bar like:

> **Good morning, Alice 👋**

The text is **rendered server-side by Python**, based on the server's current hour:

| Hour (local) | Greeting |
|---|---|
| 05–11 | Good morning |
| 12–16 | Good afternoon |
| 17–21 | Good evening |
| 22–04 | Good night |

It's plain text at the top of the page — **not** a chat-bubble response.

---

## 🔒 Why this version uses less JavaScript
Forms (register, login, flash error messages) are **100% HTML + Flask**.
No `fetch`, no AJAX, no client-side validation for auth.
Only the chat input uses a small amount of JS (to keep the typing animation
and keep the page from reloading between messages).

---

## 📱 Runs on Pydroid / Termux
The project keeps its dependency list intentionally small so it works
in low-resource Android Python environments (Pydroid 3, Termux).

---

## ⚠️ Known Limitations
- JSON storage is **not scalable** — fine for a single instance / personal use only.
- TF-IDF is much lighter than true vector embeddings; upgrade to OpenAI embeddings
  (`text-embedding-3-small`) if you need semantic recall.
- No JWT; sessions rely on Flask's signed cookies.

---

## 🚀 Roadmap
- [ ] Swap JSON → SQLite / PostgreSQL
- [ ] Password-reset flow via email
- [ ] Link-account flow (let Google-only users add a password later, and vice versa)
- [ ] OpenAI embeddings for true vector memory
- [ ] Streaming (SSE) responses

---

## 🛡 Security Notes
- Never commit your real `.env` or `users.json` (see `.gitignore`).
- Passwords are stored as **salted hashes** (`werkzeug.security.generate_password_hash`).
- For production: add HTTPS, consider a real DB, and rotate `FLASK_SECRET_KEY`.
