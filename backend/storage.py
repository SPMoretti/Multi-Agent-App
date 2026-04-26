# storage.py

# Capa de almacenamiento — opera exclusivamente sobre Supabase Storage.
# Render usa TMP_DIR como caché local de sesión para evitar descargas repetidas.

import os
from config import (
    TMP_DIR, SUPABASE_URL, SUPABASE_KEY,
    get_bucket_name, get_schema_name, normalize_username,
)

os.makedirs(TMP_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# CLIENTE SUPABASE (lazy init)
# ─────────────────────────────────────────────────────────────
_supabase = None

def _get_supabase():
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL y SUPABASE_KEY son requeridas. "
                "Configurarlas como variables de entorno en Render."
            )
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def _use_supabase() -> bool:
    """Valida que las variables de Supabase estén configuradas. Lanza error si no."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL y SUPABASE_KEY son requeridas. "
            "Configurarlas como variables de entorno en Render."
        )
    return True


# ─────────────────────────────────────────────────────────────
# CREACIÓN AUTOMÁTICA DE RECURSOS POR USUARIO
# ─────────────────────────────────────────────────────────────
_provisioned_users: set = set()  # caché en memoria para no repetir en cada request


def ensure_user_resources(username: str, pg_conn=None):
    """
    Crea (si no existen) el bucket de Supabase Storage y el schema de PostgreSQL
    para el usuario. Se llama una vez por sesión gracias al caché en memoria.
    """
    norm = normalize_username(username)
    if norm in _provisioned_users:
        return

    _ensure_bucket(username)

    if pg_conn:
        _ensure_schema(username, pg_conn)

    _provisioned_users.add(norm)


def _ensure_bucket(username: str):
    """Crea el bucket del usuario si no existe."""
    try:
        sb = _get_supabase()
        bucket_name = get_bucket_name(username)
        existing = [b.name for b in sb.storage.list_buckets()]
        if bucket_name not in existing:
            sb.storage.create_bucket(bucket_name, options={"public": False})
            print(f"[STORAGE] Bucket creado: {bucket_name}")
        else:
            print(f"[STORAGE] Bucket ya existe: {bucket_name}")
    except Exception as e:
        print(f"[STORAGE] Advertencia al crear bucket: {e}")


def _ensure_schema(username: str, conn):
    """Crea el schema PostgreSQL del usuario si no existe."""
    schema = get_schema_name(username)
    try:
        cur = conn.cursor()
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.commit()
        cur.close()
        print(f"[DB] Schema creado/verificado: {schema}")
    except Exception as e:
        print(f"[DB] Advertencia al crear schema: {e}")


# ─────────────────────────────────────────────────────────────
# OPERACIONES (todas operan sobre Supabase Storage)
# ─────────────────────────────────────────────────────────────

def upload_file(filename: str, data: bytes, username: str,
                content_type: str = "application/octet-stream") -> str:
    """
    Sube un archivo al bucket del usuario en Supabase.
    Si ya existe, lo reemplaza. Devuelve la URL pública.
    """
    _use_supabase()
    sb = _get_supabase()
    bucket = get_bucket_name(username)
    try:
        sb.storage.from_(bucket).remove([filename])
    except Exception:
        pass
    sb.storage.from_(bucket).upload(filename, data, {"content-type": content_type})
    return sb.storage.from_(bucket).get_public_url(filename)


def download_file(filename: str, username: str) -> bytes:
    """Descarga un archivo del bucket del usuario. Devuelve los bytes."""
    _use_supabase()
    sb = _get_supabase()
    return sb.storage.from_(get_bucket_name(username)).download(filename)


def list_files(username: str) -> list[str]:
    """Lista los archivos del bucket del usuario, excluyendo archivos de sistema."""
    _use_supabase()
    sb = _get_supabase()
    items = sb.storage.from_(get_bucket_name(username)).list()
    return [i["name"] for i in items if not i["name"].startswith(".")]


def delete_file(filename: str, username: str) -> bool:
    """Elimina un archivo del bucket del usuario."""
    _use_supabase()
    sb = _get_supabase()
    sb.storage.from_(get_bucket_name(username)).remove([filename])
    return True


def get_local_path(filename: str, username: str) -> str:
    """
    Asegura que el archivo esté disponible localmente para lectura.
    Descarga desde Supabase a TMP_DIR/{username}/ si no está en caché local.
    Devuelve el path local.
    """
    _use_supabase()
    user_tmp = os.path.join(TMP_DIR, normalize_username(username))
    os.makedirs(user_tmp, exist_ok=True)
    tmp_path = os.path.join(user_tmp, filename)
    if not os.path.exists(tmp_path):
        data = download_file(filename, username)
        with open(tmp_path, "wb") as f:
            f.write(data)
    return tmp_path


def save_local_then_upload(filename: str, content_bytes: bytes, username: str,
                            content_type: str = "application/octet-stream") -> str:
    """
    Guarda el archivo en TMP_DIR/{username}/ (caché local de sesión)
    y lo sube a Supabase Storage. Devuelve la URL pública.
    """
    _use_supabase()
    user_tmp = os.path.join(TMP_DIR, normalize_username(username))
    os.makedirs(user_tmp, exist_ok=True)
    tmp_path = os.path.join(user_tmp, filename)
    with open(tmp_path, "wb") as f:
        f.write(content_bytes)
    return upload_file(filename, content_bytes, username, content_type)