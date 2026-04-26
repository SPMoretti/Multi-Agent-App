#crew.py

from crewai import Crew, Process
from agents import (
    build_researcher,
    build_file_manager,
    build_analyst,
    build_sql_agent,
    build_image_agent,
)
from tasks import (
    build_research_task,
    build_file_task,
    build_synthesis_task,
    build_sql_task,
    build_image_task,
)
from utils import parse_json_response


def route(task_description: str, history=None):
    from llm import get_router_llm

    history_text = ""
    if history:
        history_text = "\n".join(
            [f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
             for m in history[-6:]]
        )

    llm = get_router_llm()
    prompt = f"""
Sos un router que decide qué agentes activar para responder la consulta del usuario.

HISTORIAL RECIENTE:
{history_text}

CONSULTA ACTUAL: "{task_description}"

AGENTES DISPONIBLES Y CUÁNDO ACTIVARLOS:

"researcher"
  → Activar cuando el dato cambia con el tiempo y no puede saberse sin internet:
    noticias, precios, cotizaciones, clima, eventos recientes, resultados deportivos,
    lanzamientos, ganadores de competencias, información de personas/empresas actuales.
  → En caso de duda sobre si algo requiere internet → activar.
  ✗ NO activar para: definiciones, conceptos, matemática, historia, hechos universales
    que no cambian ("capital de un país", "quién escribió un libro", "fórmula matemática").

"file_manager"
  → Activar cuando el usuario quiere operar sobre archivos existentes:
    leer, escribir, listar, eliminar archivos de texto (.txt, .pdf, .docx, .json).
  → Activar cuando el usuario quiere VER, DESCRIBIR o preguntar sobre el CONTENIDO
    de una imagen existente (.jpg, .png, .gif).
  → Activar si el historial indica que se estaba trabajando con archivos y la consulta
    es una continuación ("ahora borralo", "guardá eso", "listá los que hay").
  ✗ NO activar para generar imágenes nuevas.
  ✗ NO activar para archivos .csv (esos se gestionan en la sección Database).

"image_agent"
  → Activar SOLO cuando el usuario pide CREAR o GENERAR una imagen nueva.
  → Señales claras: "generá", "creá una imagen", "hacé una foto de", "dibujá", "ilustrá".
  ✗ NO activar para describir o ver imágenes que ya existen → eso es "file_manager".
  ✗ NO activar si el usuario menciona una imagen con nombre de archivo existente.

"sql"
  → Activar cuando la consulta refiere a datos estructurados: tablas, registros,
    columnas, valores, estadísticas sobre la base de datos.
  → Activar si el historial indica que se estaba consultando la base y la consulta
    es una continuación ("¿y el promedio?", "filtrá por fecha", "mostrá los últimos 5").

CUÁNDO DEVOLVER [] (sin agentes):
  → Saludos, chistes, agradecimientos.
  → Matemática simple o conversión de unidades.
  → Definiciones de conceptos estables que no cambian con el tiempo.
  → Preguntas sobre la fecha/hora actual.
  → Preguntas que el modelo puede responder con total certeza sin ningún recurso externo.

REGLAS:
- Podés activar más de un agente si la consulta lo requiere.
- Nunca activar "image_agent" y "file_manager" por la misma imagen (elegir según si es generar o describir).
- Considerar el historial para resolver referencias ambiguas ("ese archivo", "la imagen que subí", "eso").

Respondé SOLO con un array JSON válido, sin texto adicional.

EJEMPLOS:
- "¿Cuánto vale el dólar hoy?" → ["researcher"]
- "Leé el archivo ventas.txt" → ["file_manager"]
- "Qué hay en la imagen logo.png" → ["file_manager"]
- "Generá una imagen de un atardecer" → ["image_agent"]
- "Dame el total de ventas del mes" → ["sql"]
- "Buscá noticias y guardá un resumen en un txt" → ["researcher", "file_manager"]
- "¿Cuánto es 15% de 200?" → []
- "¿Cuál es la capital de Francia?" → []
- "Hola, ¿cómo estás?" → []
"""

    response = llm.call(prompt)

    try:
        import json, re
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        agents = json.loads(match.group()) if match else []
        return agents
    except Exception:
        return []


def run_crew(task_description, history=None, username="", status=None):

    agents_needed = route(task_description, history)
    print(f">>> AGENTES ACTIVADOS: {agents_needed}")

    def _build_and_run(force_json_hint=False):
        agents = []
        tasks = []

        if "researcher" in agents_needed:
            r = build_researcher()
            agents.append(r)
            tasks.append(build_research_task(r, task_description))

        if "file_manager" in agents_needed:
            fm = build_file_manager(username=username)
            agents.append(fm)
            tasks.append(build_file_task(fm, task_description, history))

        if "sql" in agents_needed:
            sql_agent = build_sql_agent(username=username)
            agents.append(sql_agent)
            tasks.append(build_sql_task(sql_agent, task_description, history))

        if "image_agent" in agents_needed:
            img_agent = build_image_agent(username=username)
            agents.append(img_agent)
            tasks.append(build_image_task(img_agent, task_description, history))

        analyst = build_analyst()
        agents.append(analyst)

        tasks.append(
            build_synthesis_task(
                analyst,
                task_description,
                history,
                tasks.copy(),
                force_json_hint=force_json_hint,
            )
        )

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        return crew.kickoff()

    # Retry limpio
    max_retries = 3

    for attempt in range(max_retries):
        force_hint = attempt > 0
        raw = str(_build_and_run(force_json_hint=force_hint))
        parsed = parse_json_response(raw)

        if parsed:
            return parsed

        print(f"[CREW] Intento {attempt + 1} fallido — respuesta no era JSON válido")

    # Fallback si los 3 intentos fallan
    return {
        "answer": raw,
        "sources": []
    }
