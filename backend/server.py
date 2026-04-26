# server.py

# Backend FastAPI 
#
# Deploy en Render:
#   Start command: uvicorn server:app --host 0.0.0.0 --port $PORT

import os
import re
import csv
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from psycopg2.extras import execute_batch

from config import MAX_HISTORY

load_dotenv()

from crew import run_crew, route
import storage as st_layer
import psycopg2


# ─────────────────────────────────────────────────────────────
# DB INTERNA (app_internal)
# ─────────────────────────────────────────────────────────────
def _get_db_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=int(os.getenv("PG_PORT", "5432")),
        database=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        sslmode="require",
        options="-c client_encoding=UTF8",
    )

# ─────────────────────────────────────────────────────────────
# HISTORIAL DESDE DB
# ─────────────────────────────────────────────────────────────
def _load_history_from_db(username: str, limit: int = 10):
    """Carga los últimos N mensajes del historial del usuario desde PostgreSQL."""
    if not username:
        return []

    try:
        conn = _get_db_conn()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT role, content
            FROM app_internal.historial
            WHERE username = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (username, limit)
        )

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    except Exception as e:
        print(f"[HISTORIAL] Error cargando historial: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Multi-Agent AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    username: str = ""

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    agents_used: list[str]
    image_url: str | None = None

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    ok: bool
    message: str


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Valida password, registra el usuario si es nuevo y provisiona sus recursos."""
    app_password = os.environ.get("APP_PASSWORD", "")
    if not app_password:
        raise HTTPException(status_code=500, detail="APP_PASSWORD no configurada.")

    if req.password != app_password:
        return LoginResponse(ok=False, message="Password incorrecta.")

    username = req.username.strip()
    if not username:
        return LoginResponse(ok=False, message="El nombre de usuario no puede estar vacío.")

    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO app_internal.usuarios (username) VALUES (%s) ON CONFLICT (username) DO NOTHING",
            (username,)
        )
        conn.commit()
        st_layer.ensure_user_resources(username, conn)
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando usuario: {e}")

    return LoginResponse(ok=True, message="ok")


@app.get("/historial/{username}")
def get_historial(username: str):
    try:
        conn = _get_db_conn()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT role, content
            FROM app_internal.historial
            WHERE username = %s
            ORDER BY id ASC
            """,
            (username,)
        )

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return {"historial": [{"role": r[0], "content": r[1]} for r in rows]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Falta OPENAI_API_KEY en las variables de entorno.")

    history = _load_history_from_db(req.username)
    if not history and req.history:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        history = history[-MAX_HISTORY:]

    agents_used = route(req.message, history)
    if "analyst" not in agents_used:
        agents_used.append("analyst")

    try:
        result = run_crew(req.message, history=history, username=req.username)
    except Exception as e:
        result = {"answer": f"Error al ejecutar el crew: {e}", "sources": []}

    if not isinstance(result, dict):
        result = {"answer": str(result), "sources": []}

    if req.username:
        try:
            conn = _get_db_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO app_internal.historial (username, role, content) VALUES (%s, %s, %s)",
                (req.username, "user", req.message)
            )
            cur.execute(
                "INSERT INTO app_internal.historial (username, role, content) VALUES (%s, %s, %s)",
                (req.username, "assistant", result.get("answer", ""))
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[HISTORIAL] Error guardando: {e}")

    image_url = _find_image_url(result.get("answer", ""), req.username)

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        agents_used=agents_used,
        image_url=image_url,
    )


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    username: str = Query(default="", description="Username del usuario que sube el archivo"),
):
    allowed_extensions = {".txt", ".csv", ".json", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Tipo de archivo no permitido: {ext}")
    if not username:
        raise HTTPException(status_code=400, detail="Se requiere username.")

    content = await file.read()
    url = st_layer.save_local_then_upload(file.filename, content, username)
    return {"ok": True, "filename": file.filename, "url": url}


@app.get("/files/{username}")
def list_files(username: str):
    try:
        files = st_layer.list_files(username)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando archivos: {e}")


@app.delete("/files/{username}/{filename}")
def delete_file(username: str, filename: str):
    try:
        st_layer.delete_file(filename, username)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando archivo: {e}")


@app.get("/files/{username}/{filename}/download")
def download_file(username: str, filename: str):
    try:
        data = st_layer.download_file(filename, username)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error descargando archivo: {e}")


@app.get("/tables/{username}")
def list_tables(username: str):
    from config import get_schema_name
    conn = None
    cur = None
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        schema = get_schema_name(username)

        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s
            ORDER BY table_name
        """, (schema,))

        tables = [r[0] for r in cur.fetchall()]
        return {"tables": tables, "schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando tablas: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()


@app.get("/tables/{username}/{table_name}")
def get_table_data(
    username: str,
    table_name: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
):
    from config import get_schema_name
    conn = None
    cur = None
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        schema = get_schema_name(username)

        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (schema, table_name))
        if cur.fetchone()[0] == 0:
            raise HTTPException(status_code=404, detail=f"Tabla '{table_name}' no encontrada.")

        cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{table_name}"')
        total = cur.fetchone()[0]
        total_pages = max(1, -(-total // page_size))  # ceil division
        offset = (page - 1) * page_size

        cur.execute(f'SELECT * FROM "{schema}"."{table_name}" LIMIT %s OFFSET %s', (page_size, offset))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

        return {
            "columns": columns,
            "rows": [list(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo datos: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()


@app.delete("/tables/{username}/{table_name}")
def drop_table(username: str, table_name: str):
    from config import get_schema_name
    conn = None
    cur = None
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        schema = get_schema_name(username)

        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (schema, table_name))
        if cur.fetchone()[0] == 0:
            raise HTTPException(status_code=404, detail=f"Tabla '{table_name}' no encontrada.")

        cur.execute(f'DROP TABLE "{schema}"."{table_name}"')
        conn.commit()

        return {"status": "dropped", "table": table_name, "schema": schema}
    except HTTPException:
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error eliminando tabla: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()


@app.get("/tables/{username}/{table_name}/export")
def export_table_csv(username: str, table_name: str):
    from config import get_schema_name
    conn = None
    cur = None
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        schema = get_schema_name(username)

        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (schema, table_name))
        if cur.fetchone()[0] == 0:
            raise HTTPException(status_code=404, detail="Tabla no encontrada")

        cur.execute(f'SELECT * FROM "{schema}"."{table_name}"')
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        writer.writerows(rows)
        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={table_name}.csv"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando CSV: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()


@app.post("/tables/{username}/{table_name}/import")
async def import_table_csv(username: str, table_name: str, file: UploadFile = File(...)):
    from config import get_schema_name

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser .csv")

    conn = None
    cur = None

    try:
        content = await file.read()

        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = content.decode("latin-1")

        reader = csv.DictReader(io.StringIO(decoded))
        rows = list(reader)

        if not rows:
            raise HTTPException(status_code=400, detail="El CSV está vacío o no tiene filas de datos.")

        conn = _get_db_conn()
        cur = conn.cursor()
        schema = get_schema_name(username)
        csv_columns = list(rows[0].keys())

        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (schema, table_name))
        table_exists = cur.fetchone()[0] > 0

        if not table_exists:
            col_definitions = _infer_columns(csv_columns, rows)
            cols_ddl = ", ".join(f'"{col}" {dtype}' for col, dtype in col_definitions.items())
            cur.execute(f'CREATE TABLE "{schema}"."{table_name}" ({cols_ddl})')
            conn.commit()
            print(f"[IMPORT] Tabla creada: {schema}.{table_name} — columnas: {col_definitions}")
            insert_columns = csv_columns
        else:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
            """, (schema, table_name))
            db_columns = [r[0] for r in cur.fetchall()]

            insert_columns = [c for c in csv_columns if c in db_columns]
            ignored = [c for c in csv_columns if c not in db_columns]

            if not insert_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Ninguna columna del CSV coincide con la tabla. Columnas en DB: {db_columns}"
                )
            if ignored:
                print(f"[IMPORT] Columnas del CSV ignoradas (no existen en la tabla): {ignored}")

        cleaned_rows = [
            {k: (v if v != "" else None) for k, v in row.items() if k in insert_columns}
            for row in rows
        ]

        cols_str = ", ".join(f'"{c}"' for c in insert_columns)
        placeholders = ", ".join(f"%({c})s" for c in insert_columns)
        insert_query = f'INSERT INTO "{schema}"."{table_name}" ({cols_str}) VALUES ({placeholders})'

        execute_batch(cur, insert_query, cleaned_rows)
        conn.commit()

        return {
            "ok": True,
            "rows_inserted": len(cleaned_rows),
            "table_created": not table_exists,
        }

    except HTTPException:
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error importando CSV: {str(e)}")
    finally:
        if cur: cur.close()
        if conn: conn.close()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _infer_columns(columns: list[str], rows: list[dict]) -> dict:
    col_types = {}
    for col in columns:
        values = [row[col] for row in rows if row.get(col, "") != ""]

        if not values:
            col_types[col] = "TEXT"
            continue

        bool_vals = {"true", "false", "1", "0", "yes", "no", "t", "f"}
        if all(v.strip().lower() in bool_vals for v in values):
            col_types[col] = "BOOLEAN"
            continue

        try:
            [int(v) for v in values]
            col_types[col] = "INTEGER"
            continue
        except ValueError:
            pass

        try:
            [float(v.replace(",", ".")) for v in values]
            col_types[col] = "NUMERIC"
            continue
        except ValueError:
            pass

        col_types[col] = "TEXT"

    return col_types


def _find_image_url(answer: str, username: str = "") -> str | None:
    if not answer:
        return None

    match = re.search(r"[\w\-]+\.png", answer, re.IGNORECASE)
    if not match:
        return None

    filename = match.group(0)

    from config import SUPABASE_URL, get_bucket_name
    if SUPABASE_URL and username:
        bucket = get_bucket_name(username)
        return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"

    return None