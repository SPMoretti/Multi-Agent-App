#tasks.py

from crewai import Task
from utils import format_history


def build_research_task(agent, task_description):
    from datetime import datetime
    today = datetime.now().strftime("%d/%m/%Y")
    current_year = datetime.now().year

    return Task(
        description=(
            f"Fecha actual: {today}\n\n"
            f"Consulta del usuario: {task_description}\n\n"

            "PASOS OBLIGATORIOS:\n"
            "1. Clasificar la temporalidad: ¿es tiempo real, reciente, fecha explícita o atemporal?\n"
            f"2. Si es tiempo real o reciente → incluir '{today}' o '{current_year}' en la query\n"
            "3. Ejecutar la búsqueda con 'Search the internet'\n"
            "4. Revisar las fechas de los snippets devueltos\n"
            "5. Si los resultados no coinciden con la temporalidad → reformular la query y reintentar\n"
            "6. Devolver la información con fecha, fuente y URL\n\n"

            "PROHIBIDO:\n"
            "- Inventar datos\n"
            "- Responder sin buscar\n"
            "- Dar por válido un resultado sin revisar su fecha en consultas temporales\n"
            "- Omitir las URLs de las fuentes\n"
        ),
        agent=agent,
        expected_output=(
            "Información encontrada con fecha de publicación, nombre del sitio y URL de cada fuente "
            "en el formato [Fuente: <url>]. "
            "Si no se encontró información confiable, explicar qué se buscó y por qué no fue suficiente."
        ),
    )


def build_file_task(agent, task_description, history=None):
    history_text = format_history(history) if history else ""

    return Task(
        description=(
            f"{history_text}"
            f"Operación solicitada: {task_description}\n\n"

            "PASO 1 — IDENTIFICAR LA OPERACIÓN:\n"
            "Determiná qué quiere el usuario ANTES de actuar:\n"
            "- ¿Pregunta sobre el contenido de una imagen? → Describe image (NO escribir)\n"
            "- ¿Quiere ver el texto de un archivo (.txt, .pdf, .docx, .json)? → Read file safely\n"
            "  (pasá siempre el campo 'query' con lo que el usuario quiere saber)\n"
            "- ¿Hace una pregunta sobre un archivo grande? → Search file with RAG\n"
            "- ¿Quiere crear o guardar texto? → Write file safely (NO para imágenes ni CSV)\n"
            "- ¿Quiere ver qué archivos tiene? → List files\n"
            "- ¿Quiere borrar? → Delete file safely\n"
            "- ¿Menciona un archivo .csv? → Informar que los CSV se gestionan en la sección Database\n\n"

            "PASO 2 — REGLAS DE EJECUCIÓN:\n"
            "- Solo el nombre del archivo, sin rutas: 'notas.txt'\n"
            "- Respetar nombre y formato pedido por el usuario\n"
            "- Si no especifica nombre → nombre descriptivo\n"
            "- Si no especifica formato → .txt\n"
            "- No inventar datos técnicos\n"
            "- Si no sabés el nombre del archivo → listar primero\n\n"

            "PASO 3 — CONFIRMAR:\n"
            "Informar el resultado exacto: nombre del archivo, operación realizada, y contenido o descripción si aplica.\n"
        ),
        agent=agent,
        expected_output=(
            "Resultado detallado: si fue lectura → contenido del archivo. "
            "Si fue escritura → nombre y formato confirmados. "
            "Si fue descripción de imagen → descripción completa. "
            "Si fue listado → cantidad y nombres de archivos. "
            "Si fue un .csv → indicar que debe usarse la sección Database."
        ),
    )


def build_image_task(agent, task_description, history=None):
    history_text = format_history(history) if history else ""

    return Task(
        description=(
            f"{history_text}"
            f"Solicitud: {task_description}\n\n"

            "Debes:\n"
            "1. Entender qué imagen quiere generar el usuario\n"
            "2. Construir un prompt descriptivo en inglés\n"
            "3. Determinar el nombre del archivo (.png)\n"
            "   - Si el usuario especifica un nombre → usarlo\n"
            "   - Si no → crear un nombre descriptivo basado en el contenido\n"
            "4. Generar la imagen con 'Generate image'\n"
            "5. Confirmar nombre y ubicación del archivo generado\n\n"

            "REGLAS:\n"
            "- El prompt DEBE estar en inglés\n"
            "- El archivo se guarda siempre en agent_files/\n"
            "- El formato es siempre .png\n"
        ),
        agent=agent,
        expected_output="Confirmacion del archivo de imagen generado con su nombre exacto.",
    )


def build_synthesis_task(agent, task_description, history, context_tasks, force_json_hint=False):
    history_text = format_history(history)

    json_hint = ""
    if force_json_hint:
        json_hint = (
            "\n\nATENCIÓN - REINTENTO POR ERROR DE FORMATO:\n"
            "Tu respuesta anterior no era JSON válido.\n"
            "Respondé ÚNICAMENTE con el JSON, sin texto previo ni posterior.\n"
            "Sin markdown, sin bloques de código, sin explicaciones.\n"
        )

    return Task(
        description=(
            f"{history_text}"
            f"Consulta del usuario: {task_description}\n\n"
            f"{json_hint}"

            "PASO 1 — IDENTIFICAR LA FUENTE PRINCIPAL:\n"
            "Revisá el contexto y determiná de dónde viene la información, en este orden de prioridad:\n"
            "  1. SQL agent → máxima prioridad, usar sin recalcular ni verificar\n"
            "  2. Researcher → usar sin corregir ni expandir con conocimiento propio\n"
            "  3. File Manager → fuente válida aunque no haya SQL ni Researcher\n"
            "  4. Image Agent → informar nombre del archivo generado o descripción\n"
            "  5. Conocimiento propio → solo si ninguna fuente anterior tiene datos\n\n"

            "PASO 2 — CONSTRUIR LA RESPUESTA:\n"
            "- Responder SIEMPRE a la consulta actual, no a consultas anteriores del historial\n"
            "- El historial es solo contexto de referencia\n"
            "- Si hay datos en el contexto → usarlos. No decir 'no tengo acceso' ni 'no hay datos'\n"
            "- Si el usuario pide contenido creativo → generarlo directamente\n"
            "- Si genuinamente no hay datos → decirlo y explicar por qué\n\n"

            "PASO 3 — FUENTES:\n"
            "- Si la fuente es SQL → sources: []\n"
            "- Si la fuente es el File Manager → sources: ['archivo: nombre_del_archivo']\n"
            "- Si la fuente es el Researcher → extraer TODAS las URLs presentes en el resultado\n"
            "  del Researcher (aparecen en formato [Fuente: <url>] o como URLs directas)\n"
            "  y colocarlas en sources como lista de strings\n\n"

            "FORMATO DE RESPUESTA (OBLIGATORIO):\n"
            "Responder SOLO con JSON válido, sin texto adicional, sin markdown:\n"
            "{\n"
            '  "answer": "respuesta clara y completa",\n'
            '  "sources": [] \n'
            "}\n\n"

            "PROHIBIDO:\n"
            "- Recalcular resultados SQL\n"
            "- Inventar datos técnicos, números o hechos\n"
            "- Agregar texto fuera del JSON\n"
            "- Usar markdown o bloques de código\n"
            "- Incluir el campo 'confidence' en la respuesta\n"
        ),
        agent=agent,
        context=context_tasks if context_tasks else None,
        expected_output="JSON válido con answer y sources.",
    )


def build_sql_task(agent, task_description, history=None):
    history_text = format_history(history) if history else ""

    return Task(
        description=(
            f"{history_text}"
            f"Consulta: {task_description}\n\n"

            "PASOS:\n"
            "1. Entender la consulta considerando el historial si es continuación\n"
            "2. Verificar que las tablas y columnas necesarias existen en el schema\n"
            "   - Si no existen → responder inmediatamente sin intentar la query\n"
            "3. Generar la query SQL adecuada:\n"
            "   - Suma, promedio, máximo, mínimo → usar SUM(), AVG(), MAX(), MIN()\n"
            "   - Registros específicos → SELECT con LIMIT razonable\n"
            "4. Ejecutar con 'Query PostgreSQL safely'\n"
            "5. Si hay SQL_ERROR → analizar, corregir y reintentar (máx 3 veces)\n"
            "   - Si tras 3 intentos sigue fallando → informar el error sin inventar datos\n"
            "6. Si el usuario pide análisis avanzado (correlación, outliers, tendencia) "
            "   → ejecutar primero la query y luego pasar los datos a 'Analyze data'\n\n"

            "REGLAS:\n"
            "- Solo usar tablas y columnas que existen en el schema\n"
            "- Solo queries SELECT\n"
            "- No inventar datos si el resultado está vacío\n"
            "- No calcular manualmente lo que puede hacer una función SQL o 'Analyze data'\n"
        ),
        agent=agent,
        expected_output=(
            "JSON con este formato según el caso:\n"
            "- Consulta exitosa: {\"data\": [{...}]}\n"
            "- Sin resultados: {\"data\": [], \"message\": \"No se encontraron registros\"}\n"
            "- Tabla/columna inexistente: {\"error\": \"La tabla X no existe en el schema\"}\n"
            "- Error SQL irrecuperable: {\"error\": \"descripción del error\"}\n"
            "- Con análisis estadístico: {\"data\": [{...}], \"analysis\": {...}}"
        ),
    )
