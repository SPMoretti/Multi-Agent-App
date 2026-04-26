# rag.py
# Lógica RAG: chunking, embeddings, búsqueda semántica y respuesta.
# Los embeddings se persisten en pgvector (Supabase Postgres) por usuario,

import os
import hashlib
import numpy as np
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, CHAR_THRESHOLD


# ─────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide el texto en chunks de tamaño aproximado.
    Usa palabras como unidad para no cortar en mitad de una palabra.
    ~1 token ≈ 4 caracteres en inglés/español.
    """
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    chunks = []
    start = 0

    while start < len(text):
        end = start + char_size

        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut != -1:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - char_overlap
        if start >= len(text):
            break

    return chunks


# ─────────────────────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────────────────────
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de chunks usando text-embedding-3-small."""
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )

    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    """Genera el embedding para la query del usuario."""
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )

    return response.data[0].embedding


# ─────────────────────────────────────────────────────────────
# HASH
# ─────────────────────────────────────────────────────────────
def _file_hash(filepath: str) -> str:
    """Calcula el hash MD5 del archivo para detectar cambios."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────
# PGVECTOR — conexión
# ─────────────────────────────────────────────────────────────
def _get_vector_conn():
    """Conexión a Supabase Postgres para las tablas de embeddings."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=int(os.getenv("PG_PORT", "5432")),
        database=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        sslmode="require",
        options="-c client_encoding=UTF8",
    )
    return conn


def _vector_table(username: str) -> str:
    """Nombre de la tabla de embeddings para el usuario en el schema rag_cache."""
    from config import normalize_username
    norm = normalize_username(username) if username else "default"
    return f"rag_cache.embeddings_{norm}"


def _ensure_vector_table(conn, username: str):
    """
    Crea el schema rag_cache y la tabla de embeddings del usuario si no existen.
    Requiere que la extensión pgvector esté habilitada en Supabase (viene por defecto).
    """
    table = _vector_table(username)
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS rag_cache")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id          SERIAL PRIMARY KEY,
            filename    TEXT NOT NULL,
            file_hash   TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text  TEXT NOT NULL,
            embedding   vector(1536),
            created_at  TIMESTAMPTZ DEFAULT now()
        )
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table.replace('.', '_')}_filename
        ON {table} (filename)
    """)
    conn.commit()
    cur.close()


# ─────────────────────────────────────────────────────────────
# CACHE — carga desde pgvector
# ─────────────────────────────────────────────────────────────
def load_cache(filename: str, filepath: str, username: str = "") -> dict | None:
    """
    Intenta cargar chunks y embeddings desde la tabla pgvector del usuario.
    Si el hash del archivo cambió, considera el caché inválido y lo descarta.
    """
    current_hash = _file_hash(filepath)

    try:
        conn = _get_vector_conn()
        _ensure_vector_table(conn, username)
        table = _vector_table(username)
        cur = conn.cursor()

        cur.execute(
            f"SELECT chunk_index, chunk_text, embedding FROM {table} "
            f"WHERE filename = %s AND file_hash = %s ORDER BY chunk_index",
            (filename, current_hash)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return None

        chunks = [r[1] for r in rows]
        # psycopg2 devuelve el vector como string "[0.1,0.2,...]" → parsear
        embeddings = [_parse_vector(r[2]) for r in rows]

        print(f"[RAG] Caché cargado desde pgvector para '{filename}' (usuario: {username or 'anónimo'})")
        return {"hash": current_hash, "chunks": chunks, "embeddings": embeddings}

    except Exception as e:
        print(f"[RAG] No se pudo cargar caché desde pgvector: {e}")
        return None


def _parse_vector(raw) -> list[float]:
    """Convierte el string de pgvector '[0.1,0.2,...]' a lista de floats."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    # Si el driver lo devuelve como objeto pgvector
    return list(raw)


# ─────────────────────────────────────────────────────────────
# CACHE — guardado en pgvector
# ─────────────────────────────────────────────────────────────
def save_cache(filename: str, filepath: str, chunks: list[str],
               embeddings: list[list[float]], username: str = ""):
    """
    Guarda chunks y embeddings en la tabla pgvector del usuario.
    Elimina primero cualquier entrada previa del mismo archivo para mantener limpieza.
    """
    current_hash = _file_hash(filepath)

    try:
        conn = _get_vector_conn()
        _ensure_vector_table(conn, username)
        table = _vector_table(username)
        cur = conn.cursor()

        # Limpiar entradas anteriores del mismo archivo
        cur.execute(f"DELETE FROM {table} WHERE filename = %s", (filename,))

        # Insertar nuevos chunks con sus embeddings
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vec_str = "[" + ",".join(str(x) for x in emb) + "]"
            cur.execute(
                f"INSERT INTO {table} (filename, file_hash, chunk_index, chunk_text, embedding) "
                f"VALUES (%s, %s, %s, %s, %s::vector)",
                (filename, current_hash, i, chunk, vec_str)
            )

        conn.commit()
        cur.close()
        conn.close()
        print(f"[RAG] Caché guardado en pgvector para '{filename}' ({len(chunks)} chunks, usuario: {username or 'anónimo'})")

    except Exception as e:
        print(f"[RAG] Advertencia: no se pudo guardar caché en pgvector: {e}")


# ─────────────────────────────────────────────────────────────
# BÚSQUEDA SEMÁNTICA
# ─────────────────────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def search_chunks(query_embedding: list[float], chunks: list[str],
                  embeddings: list[list[float]], top_k: int = TOP_K) -> list[str]:
    """Devuelve los top_k chunks más relevantes para la query."""
    scores = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(embeddings)
    ]

    scores.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in scores[:top_k]]


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────
def rag_query(text: str, query: str, filename: str, filepath: str, username: str = "") -> str:
    """
    Proceso RAG completo:
    1. Intenta cargar cache desde pgvector del usuario
    2. Si no hay cache → chunkea y embeddea → guarda en pgvector
    3. Busca chunks relevantes por similitud coseno
    4. Genera respuesta con el contexto recuperado
    """
    # 1. Intentar cargar cache
    cache = load_cache(filename, filepath, username)

    if cache:
        chunks = cache["chunks"]
        embeddings = cache["embeddings"]
    else:
        # 2. Chunkear y embeddear
        print(f"[RAG] Generando embeddings para '{filename}'...")
        chunks = chunk_text(text)
        if not chunks:
            return "El archivo no contiene texto procesable."

        embeddings = embed_chunks(chunks)
        save_cache(filename, filepath, chunks, embeddings, username)
        print(f"[RAG] Caché guardado en pgvector para '{filename}' ({len(chunks)} chunks)")

    # 3. Buscar chunks relevantes
    query_embedding = embed_query(query)
    relevant_chunks = search_chunks(query_embedding, chunks, embeddings)

    # 4. Generar respuesta con contexto
    context = "\n\n---\n\n".join(relevant_chunks)

    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sos un asistente que responde preguntas basándose EXCLUSIVAMENTE "
                    "en el contexto provisto. Si la respuesta no está en el contexto, "
                    "decilo explícitamente. No inventes información."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Contexto del documento:\n\n{context}\n\n"
                    f"Pregunta: {query}"
                )
            }
        ],
        max_tokens=1024,
        temperature=0.0,
    )

    return response.choices[0].message.content
