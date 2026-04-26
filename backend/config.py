# config.py

import os
import re
import tempfile


# ─────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE USERNAME
# ─────────────────────────────────────────────────────────────
def normalize_username(username: str) -> str:
    """
    Sanitiza el username para usarlo como nombre de bucket, schema o carpeta.
    Preserva mayúsculas y minúsculas: "user" y "USER" son usuarios distintos.
    Solo reemplaza caracteres inválidos (no alfanuméricos) por "_".
    Ej: "Juan García!" → "Juan_Garc_a"
    """
    name = username.strip()
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)  # caracteres inválidos → _ (sin .lower())
    name = re.sub(r'_+', '_', name)              # colapsa underscores múltiples
    name = name.strip('_')                       # saca underscores al inicio/fin
    return name or "default"


def get_bucket_name(username: str) -> str:
    """Nombre del bucket de Supabase Storage para el usuario."""
    return f"agent-files-{normalize_username(username)}"


def get_files_dir(username: str) -> str:
    """Carpeta local para el usuario (versión local)."""
    return f"agent_files_{normalize_username(username)}"


def get_schema_name(username: str) -> str:
    """Nombre del schema PostgreSQL para el usuario."""
    return normalize_username(username)


def get_cache_dir(username: str) -> str:
    """Carpeta de caché RAG para el usuario."""
    return os.path.join(get_files_dir(username), ".rag_cache")


# ─────────────────────────────────────────────────────────────
# ARCHIVOS
# ─────────────────────────────────────────────────────────────
TMP_DIR   = os.environ.get("TMP_DIR", tempfile.gettempdir())

# ─────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────
MODEL_ANALYST      = "gpt-4o-mini"
MODEL_FILE_MANAGER = "gpt-4o-mini"
MODEL_RESEARCHER   = "gpt-4o-mini"
MODEL_ROUTER       = "gpt-4o-mini"
MAX_HISTORY        = 10

# ─────────────────────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 100
TOP_K          = 5
CHAR_THRESHOLD = 10_000

# ─────────────────────────────────────────────────────────────
# SUPABASE STORAGE
# ─────────────────────────────────────────────────────────────
SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")   # service_role key
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "agent-files")  # fallback sin usuario