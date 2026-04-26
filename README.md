# 🤖 Multi-Agent AI Web App

Una aplicación web de chat con múltiples agentes de IA especializados, construida con **CrewAI**, **FastAPI** y **Supabase**. Cada usuario tiene su propio espacio aislado: archivos, base de datos y caché de embeddings completamente separados.

---

## ¿Qué hace esta app?

El usuario escribe un mensaje en el chat y un **router LLM** decide qué agentes activar según la naturaleza de la consulta. Los agentes trabajan en secuencia y un agente analista sintetiza el resultado final.

### Agentes disponibles

| Agente | Cuándo se activa | Capacidades |
|---|---|---|
| 🔍 **Researcher** | Información que cambia con el tiempo | Búsqueda web con SerpAPI, manejo de temporalidad |
| 📁 **File Manager** | Operaciones sobre archivos | Leer, escribir, listar, eliminar archivos (.txt, .pdf, .docx, .json); describir imágenes |
| 🎨 **Image Agent** | Generación de imágenes nuevas | Genera imágenes con DALL-E 2 |
| 🗄️ **SQL Agent** | Consultas sobre datos estructurados | Ejecuta SELECT sobre el schema del usuario en PostgreSQL; análisis estadístico |
| 🧠 **Analyst** | Siempre (agente de síntesis) | Consolida los resultados y responde al usuario en JSON |

El router puede activar **cero o más agentes** dependiendo de la consulta. Saludos, matemática simple o preguntas conceptuales se responden directamente sin activar ningún agente especializado.

---

## Arquitectura

```
Frontend (HTML/JS)
       │
       ▼
FastAPI (server.py)
       │
       ├── Router LLM  →  decide qué agentes activar
       │
       └── CrewAI (crew.py)
              ├── Researcher        → SerpAPI
              ├── File Manager      → Supabase Storage + RAG (pgvector)
              ├── Image Agent       → DALL-E 2 + Supabase Storage
              ├── SQL Agent         → PostgreSQL (schema por usuario)
              └── Analyst           → respuesta final
```

### Almacenamiento por usuario

- **Archivos**: Supabase Storage — cada usuario tiene su propio bucket (`agent-files-{username}`)
- **Base de datos**: PostgreSQL — cada usuario tiene su propio schema (`{username}`)
- **Caché RAG**: pgvector en el schema `rag_cache` — embeddings persistentes por archivo y usuario
- **Historial de chat**: tabla `app_internal.historial` en PostgreSQL

---

## Requisitos previos

### Servicios externos

| Servicio | Para qué se usa | Dónde obtenerlo |
|---|---|---|
| **OpenAI** | LLM (GPT-4o-mini), embeddings (text-embedding-3-small), visión, DALL-E 2 | [platform.openai.com](https://platform.openai.com) |
| **SerpAPI** | Búsqueda web del agente Researcher | [serpapi.com](https://serpapi.com) |
| **Supabase** | Storage de archivos + PostgreSQL + pgvector | [supabase.com](https://supabase.com) |

### Dependencias Python

```bash
pip install fastapi uvicorn crewai openai supabase psycopg2-binary \
            serpapi pymupdf python-docx numpy pandas pydantic python-dotenv
```

---

## Variables de entorno

Crear un archivo `.env` en la raíz del proyecto (ver `.env.example`):

```env
# Contraseña de acceso a la app
APP_PASSWORD=tu_password_seguro

# OpenAI
OPENAI_API_KEY=sk-...

# SerpAPI
SERPAPI_API_KEY=...

# PostgreSQL (conexión directa a Supabase)
PG_HOST=db.xxxxxxxxxxxx.supabase.co
PG_PORT=5432
PG_DB=postgres
PG_USER=postgres
PG_PASSWORD=...

# Supabase Storage
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJ...        # service_role key (no la anon key)
SUPABASE_BUCKET=agent-files
```

> ⚠️ Usar la **service_role key** de Supabase (no la `anon key`) para que el backend pueda crear buckets y operar sin restricciones de RLS.

---

## Configuración de Supabase (SQL)

Ejecutar los siguientes comandos en el **SQL Editor** de Supabase antes de levantar la app.

### 1. Habilitar pgvector

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Schema interno de la app

Contiene el historial de conversaciones y los usuarios registrados.

```sql
CREATE SCHEMA IF NOT EXISTS app_internal;

-- Tabla de usuarios
CREATE TABLE IF NOT EXISTS app_internal.usuarios (
    id         SERIAL PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Tabla de historial de chat
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

### 3. Schema para caché RAG (embeddings)

Las tablas individuales por usuario se crean automáticamente en tiempo de ejecución, pero el schema debe existir.

```sql
CREATE SCHEMA IF NOT EXISTS rag_cache;
```

### 4. Schemas de usuario (opcionales, se crean automáticamente)

Cada vez que un usuario carga un CSV desde la sección Database, la app crea su schema automáticamente. Si querés crearlo manualmente para un usuario en particular:

```sql
-- Reemplazar 'nombre_usuario' por el username real
CREATE SCHEMA IF NOT EXISTS "nombre_usuario";
```

---

## Levantar la app localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 5. Levantar el backend
uvicorn server:app --reload --port 8000

# 6. Abrir el frontend
# Abrir index.html en el navegador o servirlo con Live Server
```

Para desarrollo local, `API_URL` (en frontend) = `http://localhost:8000`.


---

## Deploy en Render

Start command:

```bash
uvicorn server:app --host 0.0.0.0 --port $PORT
```

Agregar todas las variables de entorno del archivo `.env` en el panel de Render → **Environment**.

Para deploy en Vercel,`API_URL` (en frontend) = "URL provista por Render"

---

## Estructura del proyecto

```
├── server.py        # API FastAPI (endpoints de chat, archivos, DB)
├── crew.py          # Orquestación de agentes con CrewAI
├── agents.py        # Definición de cada agente
├── tasks.py         # Definición de tareas por agente
├── tools.py         # Herramientas: SerpAPI, archivos, SQL, DALL-E, RAG
├── rag.py           # Pipeline RAG: chunking, embeddings, búsqueda semántica
├── storage.py       # Capa de abstracción sobre Supabase Storage
├── llm.py           # Configuración de LLMs por agente
├── config.py        # Constantes y helpers de configuración
├── utils.py         # Utilidades: parseo JSON, formato de historial
├── index.html       # Frontend principal (chat)
├── storage.html     # Frontend de gestión de archivos
├── database.html    # Frontend de gestión de base de datos (CSV upload)
└── .env.example     # Plantilla de variables de entorno
```

---

## Funcionalidades del frontend

- **Chat** (`index.html`): interfaz conversacional con historial persistente
- **Storage** (`storage.html`): subir, descargar y eliminar archivos del usuario
- **Database** (`database.html`): cargar archivos CSV como tablas en el schema personal del usuario, consultables desde el chat

---

## Notas de seguridad

- Las queries SQL están restringidas a `SELECT`. Cualquier intento de `DROP`, `DELETE`, `UPDATE` o `INSERT` es bloqueado.
- Cada usuario accede únicamente a su propio schema de PostgreSQL y bucket de Storage.
- La `SUPABASE_KEY` (service_role) nunca se expone al frontend.
