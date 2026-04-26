# 🤖 Multi-Agent AI Web App

A web chat application powered by multiple specialized AI agents, built with **CrewAI**, **FastAPI** and **Supabase**. Each user has their own isolated space: files, database and embeddings cache completely separated.

---

## What does this app do?

The user types a message in the chat and a **router LLM** decides which agents to activate based on the nature of the query. The agents work sequentially and an analyst agent synthesizes the final response.

### Available agents

| Agent | When it's activated | Capabilities |
|---|---|---|
| 🔍 **Researcher** | Information that changes over time | Web search via SerpAPI, temporal query handling |
| 📁 **File Manager** | File operations | Read, write, list, delete files (.txt, .pdf, .docx, .json); describe images |
| 🎨 **Image Agent** | Generating new images | Generates images with DALL-E 2 |
| 🗄️ **SQL Agent** | Queries on structured data | Runs SELECT queries on the user's PostgreSQL schema; statistical analysis |
| 🧠 **Analyst** | Always (synthesis agent) | Consolidates results and returns the final response as JSON |

The router can activate **zero or more agents** depending on the query. Greetings, simple math or conceptual questions are answered directly without activating any specialized agent.

---

## Architecture

```
Frontend (HTML/JS)
       │
       ▼
FastAPI (server.py)
       │
       ├── Router LLM  →  decides which agents to activate
       │
       └── CrewAI (crew.py)
              ├── Researcher        → SerpAPI
              ├── File Manager      → Supabase Storage + RAG (pgvector)
              ├── Image Agent       → DALL-E 2 + Supabase Storage
              ├── SQL Agent         → PostgreSQL (per-user schema)
              └── Analyst           → final response
```

### Per-user storage

- **Files**: Supabase Storage — each user has their own bucket (`agent-files-{username}`)
- **Database**: PostgreSQL — each user has their own schema (`{username}`)
- **RAG cache**: pgvector in the `rag_cache` schema — persistent embeddings per file and user
- **Chat history**: `app_internal.historial` table in PostgreSQL

---

## Prerequisites

### External services

| Service | Used for | Where to get it |
|---|---|---|
| **OpenAI** | LLM (GPT-4o-mini), embeddings (text-embedding-3-small), vision, DALL-E 2 | [platform.openai.com](https://platform.openai.com) |
| **SerpAPI** | Web search for the Researcher agent | [serpapi.com](https://serpapi.com) |
| **Supabase** | File storage + PostgreSQL + pgvector | [supabase.com](https://supabase.com) |

### Python dependencies

```bash
pip install fastapi uvicorn crewai openai supabase psycopg2-binary \
            serpapi pymupdf python-docx numpy pandas pydantic python-dotenv
```

---

## Environment variables

Create a `.env` file in the project root (see `.env.example`):

```env
# App access password
APP_PASSWORD=your_secure_password

# OpenAI
OPENAI_API_KEY=sk-...

# SerpAPI
SERPAPI_API_KEY=...

# PostgreSQL (direct connection to Supabase)
PG_HOST=db.xxxxxxxxxxxx.supabase.co
PG_PORT=5432
PG_DB=postgres
PG_USER=postgres
PG_PASSWORD=...

# Supabase Storage
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJ...        # service_role key (not the anon key)
SUPABASE_BUCKET=agent-files
```

> ⚠️ Use the **service_role key** from Supabase (not the `anon key`) so the backend can create buckets and operate without RLS restrictions.

---

## Supabase setup (SQL)

Run the following commands in the **Supabase SQL Editor** before starting the app.

### 1. Enable pgvector

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Internal app schema

Holds the chat history and registered users.

```sql
CREATE SCHEMA IF NOT EXISTS app_internal;

-- Users table
CREATE TABLE IF NOT EXISTS app_internal.usuarios (
    id         SERIAL PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Chat history table
CREATE TABLE IF NOT EXISTS app_internal.historial (
    id         SERIAL PRIMARY KEY,
    username   TEXT NOT NULL,
    role       TEXT NOT NULL,   -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_historial_username
    ON app_internal.historial (username);
```

### 3. RAG cache schema (embeddings)

Per-user tables are created automatically at runtime, but the schema must exist beforehand.

```sql
CREATE SCHEMA IF NOT EXISTS rag_cache;
```

### 4. User schemas (optional — created automatically)

Every time a user uploads a CSV from the Database section, the app creates their schema automatically. To create one manually for a specific user:

```sql
-- Replace 'username' with the actual username
CREATE SCHEMA IF NOT EXISTS "username";
```

---

## Running locally

```bash
# 1. Clone the repository
git clone https://github.com/your-user/your-repo.git
cd your-repo

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# 5. Start the backend
uvicorn server:app --reload --port 8000

# 6. Open the frontend
# Open index.html in your browser or serve it with Live Server
```

For local development, set `API_URL` (in the frontend files) to `http://localhost:8000`.

---

## Deploy on Render

Start command:

```bash
uvicorn server:app --host 0.0.0.0 --port $PORT
```

Add all environment variables from `.env` in the Render dashboard → **Environment**.

Once deployed, update `API_URL` in the frontend files to the URL provided by Render.

---

## Project structure

```
├── server.py        # FastAPI app (chat, files and DB endpoints)
├── crew.py          # Agent orchestration with CrewAI
├── agents.py        # Agent definitions
├── tasks.py         # Task definitions per agent
├── tools.py         # Tools: SerpAPI, files, SQL, DALL-E, RAG
├── rag.py           # RAG pipeline: chunking, embeddings, semantic search
├── storage.py       # Abstraction layer over Supabase Storage
├── llm.py           # LLM configuration per agent
├── config.py        # Constants and configuration helpers
├── utils.py         # Utilities: JSON parsing, history formatting
├── index.html       # Main frontend (chat)
├── storage.html     # File management frontend
├── database.html    # Database management frontend (CSV upload)
└── .env.example     # Environment variables template
```

---

## Frontend features

- **Chat** (`index.html`): conversational interface with persistent history
- **Storage** (`storage.html`): upload, download and delete user files
- **Database** (`database.html`): load CSV files as tables into the user's personal schema, queryable from the chat

---

## Security notes

- SQL queries are restricted to `SELECT`. Any attempt to run `DROP`, `DELETE`, `UPDATE` or `INSERT` is blocked.
- Each user can only access their own PostgreSQL schema and Storage bucket.
- The `SUPABASE_KEY` (service_role) is never exposed to the frontend.