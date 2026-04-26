# 🤖 Multi-Agent AI Web App

A web chat application with multiple specialized AI agents, built with CrewAI, FastAPI, and Supabase. Each user has their own isolated space: completely separate files, database, and embedding cache.

---

## What does this app do?

The user types a message in the chat, and an LLM router decides which agents to activate based on the nature of the query. The agents work sequentially, and an analyst agent synthesizes the final result.

### Available Agents

| Agent | When it's activated | Capabilities |

---|---|---|

🔍 **Researcher** | Information that changes over time | Web search with SerpAPI, handling timelines |

📁 **File Manager** | File operations | Read, write, list, and delete files (.txt, .pdf, .docx, .json); describe images |

🎨 **Image Agent** | Generate new images | Generates images using DALL-E 2 |

🗄️ **SQL Agent** | Queries on structured data | Executes SELECT statements on the user's schema in PostgreSQL; statistical analysis |

🧠 **Analyst** | Always (synthesis agent) | Consolidates results and responds to the user in JSON |

The router can activate **zero or more agents** depending on the query. Greetings, simple math, or conceptual questions are answered directly without activating any specialized agents.

---

## Architecture

```
Frontend (HTML/JS)

▼
FastAPI (server.py)

│

├── Router LLM → decides which agents to activate

│

└── CrewAI (crew.py)

├── Researcher → SerpAPI

├── File Manager → Supabase Storage + RAG (pgvector)

├── Image Agent → DALL-E 2 + Supabase Storage

├── SQL Agent → PostgreSQL (user-defined schema)

└── Analyst → final response
```

### User-defined Storage

- **Files**: Supabase Storage — each User has their own bucket (`agent-files-{username}`)
- **Database**: PostgreSQL — each user has their own schema (`{username}`)
- **RAG Cache**: pgvector in the schema `rag_cache` — persistent embeddings per file and user
- **Chat History**: table `app_internal.historial` in PostgreSQL

---

## Prerequisites

### External Services

| Service | What it's used for | Where to get it |

---|---|---|

**OpenAI** | LLM (GPT-4o-mini), embeddings (text-embedding-3-small), vision, DALL-E 2 | [platform.openai.com](https://platform.openai.com) |

**SerpAPI** | Web search for the Researcher agent | [serpapi.com](https://serpapi.com) |
| **Supabase** | File storage + PostgreSQL + pgvector | [supabase.com](https://supabase.com) |

### Python Dependencies

```bash
pip install fastapi uvicorn crewai openai supabase psycopg2-binary \
serpapi pymupdf python-docx numpy pandas pydantic python-dotenv
```

---

## Environment Variables

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
SUPABASE_KEY=eyJ... # service_role key (not the anonymous key)
SUPABASE_BUCKET=agent-files
```

> ⚠️ Use the Supabase **service_role key** (not the anonymous key) so the backend can create buckets and operate without RLS restrictions.

---

## Supabase Configuration (SQL)

Run the following commands in the Supabase **SQL Editor** before starting the app.

### 1. Enable pgvector

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Internal schema of the app

Contains conversation history and registered users.

```sql
CREATE SCHEMA IF NOT EXISTS app_internal;

-- User table
CREATE TABLE IF NOT EXISTS app_internal.users ( 
id SERIAL PRIMARY KEY, 
username TEXT UNIQUE NOT NULL, 
created_at TIMESTAMPTZ DEFAULT now()
);

-- Chat history table
CREATE TABLE IF NOT EXISTS app_internal.history ( 
id SERIAL PRIMARY KEY, 
username TEXT NOT NULL, 
role TEXT NOT NULL, -- 'user' | 'assistant' 
content TEXT NOT NULL, 
created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_historial_username
ON app_internal.historial (username);

``

### 3. Schema for RAG cache (embeddings)

Individual tables for each user are created automatically at runtime, but the schema must exist.

``sql
CREATE SCHEMA IF NOT EXISTS rag_cache;

``

### 4. User schemas (optional, created automatically)

Each time a user uploads a CSV from the Database section, the app automatically creates their schema. If you want to create it manually for a particular user:

```sql
-- Replace 'username' with the actual username